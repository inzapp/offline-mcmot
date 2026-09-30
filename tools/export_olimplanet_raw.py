#!/usr/bin/env python3
"""Export tracking outputs in the Olymplanet sample shape.

The delivery package exposes tracking/path data and track-level gender/age
values when the improved model produced a matching result. Aggregate
demographic files are not copied into the package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path


PACKAGE_NAME = "olimplanet_raw_json_20260904-05_latest"
TIMEZONE = "Asia/Seoul"


def rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def rounded(value: float | None, digits: int = 6):
    if value is None:
        return None
    result = round(value, digits)
    return int(result) if result == int(result) else result


def time_only(value: str | None) -> str | None:
    if not value:
        return None
    value = value.replace("T", " ")
    if len(value) >= 19 and value[10] == " ":
        return value[11:19]
    return value[-8:] if len(value) >= 8 else value


def is_true(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def local_uid(row: dict[str, str]) -> str:
    nominal_start = row["recording_id"].rsplit("_", 1)[-1]
    return f"{row['date']}:{row['camera_id']}:{nominal_start}:L{row['local_id']}"


def gaze_values(row: dict[str, str]) -> tuple[float, list[float]]:
    """Convert the stored camera gaze ray to the sample's g/e fields."""
    if not is_true(row.get("gaze_valid")):
        return -1, [-1, -1]
    values = [
        float_or_none(row.get("gaze_camera_start_x")),
        float_or_none(row.get("gaze_camera_start_y")),
        float_or_none(row.get("gaze_camera_end_x")),
        float_or_none(row.get("gaze_camera_end_y")),
    ]
    if any(value is None for value in values):
        return -1, [-1, -1]
    sx, sy, ex, ey = values
    dx, dy = ex - sx, ey - sy
    if math.hypot(dx, dy) < 1e-12:
        return -1, [-1, -1]
    angle = math.degrees(math.atan2(dy, dx)) % 360.0
    return rounded(angle, 3), [rounded(sx), rounded(sy)]


def path_point(row: dict[str, str]) -> dict:
    cx = float(row["cx"])
    cy = float(row["cy"])
    height = float(row["h"])
    gaze, eye = gaze_values(row)
    return {
        "t": time_only(row["timestamp"]),
        "p": [rounded(cx), rounded(cy + height / 2)],
        # The latest MCMOT CSV does not contain the legacy z field.  Keep the
        # sample-compatible required field with its documented sentinel.
        "z": 0,
        "g": gaze,
        # Body direction is not retained in the latest output.
        "b": -1,
        "e": eye,
    }


def build_paths(root: Path) -> tuple[dict[str, list[dict]], int, int]:
    """Read observed points and keep the first point in each second per ID."""
    paths: dict[str, list[dict]] = {}
    last_second: dict[str, str] = {}
    source_rows = 0
    selected_points = 0
    track_path = root / "local" / "tracks.csv"
    for row in rows(track_path):
        source_rows += 1
        if not is_true(row.get("is_observed")):
            continue
        if not row.get("cx") or not row.get("cy") or not row.get("h"):
            continue
        uid = local_uid(row)
        second = row["timestamp"][:19]
        if last_second.get(uid) == second:
            continue
        last_second[uid] = second
        paths.setdefault(uid, []).append(path_point(row))
        selected_points += 1
    return paths, source_rows, selected_points


def attribute_uid(row: dict[str, str]) -> str | None:
    filename = Path(row.get("file", "")).stem
    parts = filename.split("_")
    if len(parts) != 4 or not parts[3].startswith("L"):
        return None
    date, camera, nominal_start, local_id = parts
    if not date or not camera or not nominal_start or not local_id[1:]:
        return None
    return f"{date}:{camera}:{nominal_start}:{local_id}"


def load_attributes(root: Path) -> dict[str, dict[str, object]]:
    """Load improved track-level attributes keyed by the exported person ID."""
    attributes: dict[str, dict[str, object]] = {}
    for row in rows(root / "attributes" / "results.csv"):
        uid = attribute_uid(row)
        if uid is None or uid in attributes:
            continue
        gender = (row.get("gender") or "").strip() or None
        attributes[uid] = {
            "gender": gender,
            "age": rounded(float_or_none(row.get("age")), 6),
        }
    return attributes


def measure(row: dict[str, str], attributes: dict[str, object] | None) -> dict:
    attributes = attributes or {}
    return {
        "exposed_time": rounded(float_or_none(row.get("track_span_seconds")), 6),
        "watched_time": rounded(float_or_none(row.get("watch_seconds")), 6),
        "attention_time": rounded(float_or_none(row.get("attention_seconds")), 6),
        "gender": attributes.get("gender"),
        "age": attributes.get("age"),
        "appear_timing": time_only(row.get("first_seen")),
    }


def track_object(
    row: dict[str, str],
    paths: dict[str, list[dict]],
    attributes_by_uid: dict[str, dict[str, object]],
) -> dict:
    uid = row["local_uid"]
    return {
        "camera_number": str(row["camera_id"]),
        "person_id": uid,
        "measure": measure(row, attributes_by_uid.get(uid)),
        "path": paths.get(uid, []),
    }


def root_header(site: str, dates: list[str], cameras: list[str], combined: bool) -> dict:
    header = OrderedDict([
        ("format_version", "1.1"),
        ("data_dates", dates),
        ("timezone", TIMEZONE),
        ("site", site),
    ])
    if combined:
        header["camera_numbers"] = cameras
    return header


def open_json_array(path: Path, header: dict) -> object:
    stream = path.open("w", encoding="utf-8", newline="\n")
    prefix = json.dumps(header, ensure_ascii=False, indent=2)
    stream.write(prefix[:-1])
    stream.write(',\n  "tracks": [\n')
    return stream


def write_json_files(
    output_root: Path,
    site: str,
    dates: list[str],
    cameras: list[str],
    people: list[dict[str, str]],
    paths: dict[str, list[dict]],
    attributes_by_uid: dict[str, dict[str, object]],
) -> dict:
    combined_dir = output_root / "combined"
    camera_dir = output_root / "by_camera"
    combined_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)

    combined_path = combined_dir / "all_cameras_raw.json"
    combined = open_json_array(combined_path, root_header(site, dates, cameras, True))
    camera_streams = {}
    camera_first = Counter()
    combined_written = 0
    camera_counts = Counter()
    camera_path_points = Counter()
    total_points = 0
    total_with_path = 0
    try:
        for camera in cameras:
            path = camera_dir / f"camera_{camera}_raw.json"
            camera_streams[camera] = open_json_array(
                path, root_header(site, dates, [camera], False)
            )
        for person in people:
            camera = str(person["camera_id"])
            payload = track_object(person, paths, attributes_by_uid)
            encoded = json.dumps(payload, ensure_ascii=False, indent=2)
            if combined_written:
                combined.write(",\n")
            if camera_first[camera]:
                camera_streams[camera].write(",\n")
            combined.write(encoded)
            camera_streams[camera].write(encoded)
            combined_written += 1
            camera_first[camera] += 1
            camera_counts[camera] += 1
            point_count = len(payload["path"])
            camera_path_points[camera] += point_count
            total_points += point_count
            total_with_path += point_count > 0
    finally:
        combined.write("\n  ]\n}\n")
        combined.close()
        for stream in camera_streams.values():
            stream.write("\n  ]\n}\n")
            stream.close()

    return {
        "combined": "combined/all_cameras_raw.json",
        "tracks": len(people),
        "tracks_with_path": total_with_path,
        "path_points": total_points,
        "tracks_by_camera": dict(camera_counts),
        "path_points_by_camera": dict(camera_path_points),
    }


def load_people(root: Path) -> list[dict[str, str]]:
    people = []
    for row in rows(root / "local" / "persons.csv"):
        row = dict(row)
        row["local_uid"] = row["local_uid"]
        people.append(row)
    return people


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_schema(package_root: Path) -> None:
    schema = {
        "format_version": "1.1",
        "root_structure": {
            "format_version": "데이터 형식 버전",
            "data_dates": "사이트 파일에 포함된 기준 날짜 배열(YYYY-MM-DD)",
            "timezone": "측정 시각 기준 시간대",
            "site": "사이트 구분자",
            "camera_numbers": "통합 파일에 포함된 카메라 번호 배열",
            "tracks": "사람별 카메라 단위 레코드 배열",
            "combined": "사이트 내 전체 카메라 통합 tracks 배열",
            "by_camera": "사이트·카메라별 tracks 배열",
        },
        "track_fields": {
            "camera_number": "카메라 번호(string)",
            "person_id": "녹화 구간을 구분하는 익명 추적 ID(string)",
            "measure": "사람 단위 tracking 측정값 객체",
            "path": "관측 tracking point의 시간순 배열(초당 최대 1점)",
        },
        "measure_fields": {
            "exposed_time": "sec; 사람이 화면에 머문 추적 시간(예측 구간 포함)",
            "watched_time": "sec; 시청 판정 누적 시간",
            "attention_time": "sec; 주목 판정 누적 시간",
            "gender": "male/female; 개선 모델 결과가 없으면 null",
            "age": "number; 개선 모델 추정 연령, 결과가 없으면 null",
            "appear_timing": "최초 등장 시각(HH:MM:SS, KST)",
        },
        "path_fields": {
            "t": "측정 시각(HH:MM:SS, KST)",
            "p": "[cx, bbox bottom] 카메라 화면 정규화 좌표",
            "z": "0 sentinel; 제공 원천에 해당 필드 없음",
            "g": "gaze camera ray의 방향각(도); 미산출 -1",
            "b": "-1 sentinel; 제공 원천에 body angle 없음",
            "e": "gaze ray 시작점(camera normalized); 미산출 [-1,-1]",
        },
        "sampling": {
            "source": "관측 위치 데이터",
            "rows": "is_observed=1인 관측 row만 사용",
            "resolution": "각 추적 ID·초(second)당 첫 관측점 1개",
        },
    }
    (package_root / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def make_description(
    package_root: Path,
    sites: list[dict],
    package_name: str,
    created_at: str,
    city: str,
) -> str:
    dates = sorted({date for site in sites for date in site["dates"]})
    lines = [
        f"# 올림플래닛 제공용 {city.capitalize()} raw 데이터 설명서",
        "",
        f"- 패키지: `{package_name}`",
        f"- 작성 시각: {created_at} (KST)",
        f"- 기준 데이터: {dates[0]} ~ {dates[-1]}, KST",
        "- 형식: UTF-8 JSON",
        "",
        "## 1. 산출물 구성",
        "",
        f"- raw 패키지: `{package_name}`",
        "- JSON은 사이트별 `combined/all_cameras_raw.json`과 `by_camera/camera_*_raw.json`으로 구성됩니다.",
        "- 성별·연령은 track별 개선 모델 결과를 연결하며, 연결 결과가 없는 track은 `null`로 표시합니다.",
        "",
        "## 2. 적용한 원천 및 최신 버전",
        "",
        "| 사이트 | 데이터 기간 | 카메라 | 추적 ID | 1초 path point | 성별·연령 매칭 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for site in sites:
        counts = site["counts"]
        attribute_matching = site["attribute_matching"]
        cameras = ", ".join(site["cameras"])
        lines.append(
            f"| `{site['site']}` | {', '.join(site['dates'])} | {cameras} | "
            f"{counts['tracks']:,} | {counts['path_points']:,} | "
            f"{attribute_matching['matched_tracks']:,}/{counts['tracks']:,} |"
        )
    lines += [
        "",
        "각 사이트에서 번호 기준으로 가장 최신인 처리 결과를 사용했으며, 두 사이트 모두 동일한 기간의 데이터를 포함합니다.",
        "",
        "샘플과 동일한 `tracks`·`measure`·`path` 구조를 유지하고, 여러 날짜를 한 파일에 담기 위해 JSON 머리말의 `data_dates`를 배열로 표기했습니다.",
        "",
        "## 3. JSON 구조",
        "",
        "각 `tracks` 항목은 한 카메라 녹화 안에서 관측된 한 사람의 추적 구간입니다. "
        "`person_id`는 서로 다른 녹화 구간에서도 구분할 수 있도록 구성한 익명 추적 ID입니다.",
        "",
        "| 필드 | 설명 |",
        "|---|---|",
        "| `camera_number` | 카메라 번호 |",
        "| `person_id` | 날짜·카메라·녹화 시작 시각·순번으로 구성한 익명 추적 ID |",
        "| `measure.exposed_time` | 화면에 머문 추적 시간(초, 예측 구간 포함) |",
        "| `measure.watched_time` | 시청 판정 누적 시간(초) |",
        "| `measure.attention_time` | 주목 판정 누적 시간(초) |",
        "| `measure.gender` | 개선 모델 성별 결과(`male`/`female`), 결과가 없으면 `null` |",
        "| `measure.age` | 개선 모델 추정 연령(세), 결과가 없으면 `null` |",
        "| `measure.appear_timing` | 최초 등장 시각(KST) |",
        "| `path` | 관측 row를 1초 단위로 줄인 시간순 좌표 |",
        "",
        "`path.p`는 카메라 화면 정규화 bbox 중심 하단 좌표입니다. `z`와 `b`는 "
        "원천 값이 없어 각각 `0`, `-1` sentinel을 사용했습니다. `g`와 `e`는 저장된 "
        "카메라 gaze ray에서 변환한 값이며, 유효하지 않으면 `-1`/`[-1,-1]`입니다.",
        "",
        "## 4. 성별·연령 데이터 전달 원칙",
        "",
        "각 track의 `measure.gender`와 `measure.age`는 개선 모델의 track별 결과를 `person_id`에 연결한 값입니다. "
        "결과 파일에서 해당 track을 찾을 수 없거나 값이 비어 있는 경우에는 해당 필드를 `null`로 표시합니다. "
        "이번 전달본에는 별도의 성별·연령 집계 파일을 포함하지 않습니다.",
        "",
        "## 5. 검증 방법",
        "",
        "- `manifest.json`에 패키지 파일별 byte 수와 SHA-256을 기록했습니다.",
        "- JSON track 수·path point 수·카메라별 집계는 manifest의 counts와 일치해야 합니다.",
        "- 성별·연령 매칭 건수는 manifest의 attribute matching 집계와 일치해야 합니다.",
        "",
        "세부 필드 정의는 패키지의 `schema.json`, 파일 목록과 검증 집계는 `manifest.json`을 참고하십시오.",
        "",
    ]
    return "\n".join(lines)


def write_package_readme(package_root: Path, description_name: str, city: str) -> None:
    text = f"""# 올림플래닛 제공용 raw JSON 패키지

자세한 전달 규칙과 필드 설명은 함께 전달한 `{description_name}`를 참고하십시오.

## 디렉터리

- `{city}_inside/`: 실내 데이터의 JSON
- `{city}_outside/`: 실외 데이터의 JSON
- `schema.json`: JSON 필드 및 성별·연령 제공 제한
- `manifest.json`: 파일별 SHA-256, track/path 집계

## 성별·연령 제한

개인 단위 JSON의 `measure.gender`, `measure.age`에는 track별 개선 모델 결과가 들어갑니다.
결과가 없는 track은 해당 값을 `null`로 표시하며, 별도의 성별·연령 집계 파일은 포함하지 않습니다.
"""
    (package_root / "README.md").write_text(text, encoding="utf-8")


def write_manifest(package_root: Path, metadata: dict, package_name: str) -> None:
    files = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(package_root).as_posix()
        files.append({
            "name": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "encoding": "UTF-8",
        })
    payload = {
        "package": package_name,
        "created_at": metadata["created_at"],
        "format": "JSON",
        "data_period": metadata["data_period"],
        "sites": metadata["sites"],
        "track_attributes": "track-level gender/age are included when matched; unmatched values are null; aggregate files are not included",
        "counts": metadata["counts"],
        "attribute_matching": metadata["attribute_matching"],
        "files": files,
    }
    (package_root / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def export_site(source_root: Path, package_root: Path, site: str) -> dict:
    people = load_people(source_root)
    dates = sorted({row["date"] for row in people if row.get("date")})
    cameras = sorted({str(row["camera_id"]) for row in people if row.get("camera_id")})
    paths, source_rows, _ = build_paths(source_root)
    people_uids = {row["local_uid"] for row in people}
    paths = {uid: points for uid, points in paths.items() if uid in people_uids}
    selected_points = sum(len(points) for points in paths.values())
    output_root = package_root / site
    attributes_by_uid = load_attributes(source_root)
    people_uids = {row["local_uid"] for row in people}
    matched_uids = people_uids & set(attributes_by_uid)
    tracks_with_gender = sum(
        attributes_by_uid[uid].get("gender") is not None for uid in matched_uids
    )
    tracks_with_age = sum(
        attributes_by_uid[uid].get("age") is not None for uid in matched_uids
    )
    json_counts = write_json_files(
        output_root, site, dates, cameras, people, paths, attributes_by_uid
    )
    return {
        "site": site,
        "source": source_root.name,
        "cameras": cameras,
        "dates": dates,
        "counts": {
            **json_counts,
            "source_track_rows": source_rows,
            "selected_path_points": selected_points,
        },
        "attribute_matching": {
            "result_rows": len(attributes_by_uid),
            "matched_tracks": len(matched_uids),
            "tracks_without_result": len(people_uids - set(attributes_by_uid)),
            "tracks_with_gender": tracks_with_gender,
            "tracks_with_age": tracks_with_age,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--city", choices=["pohang", "gumi"], default="pohang")
    parser.add_argument("--inside", type=Path, default=None)
    parser.add_argument("--outside", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    default_inside = "output_pohang_inside4" if args.city == "pohang" else "output_gumi_inside3"
    default_outside = "output_pohang_outside3" if args.city == "pohang" else "output_gumi_outside3"
    inside = (args.inside or workspace / default_inside).resolve()
    outside = (args.outside or workspace / default_outside).resolve()
    for path in (inside, outside):
        if not (path / "local" / "persons.csv").exists():
            raise SystemExit(f"missing source output: {path}")
        if not (path / "local" / "tracks.csv").exists() or not (path / "attributes" / "results.csv").exists():
            raise SystemExit(f"missing tracks or attributes output: {path}")

    source_dates = sorted({row["date"] for root in (inside, outside)
                           for row in rows(root / "local" / "persons.csv") if row.get("date")})
    if not source_dates:
        raise SystemExit("no source dates found")
    date_tag = source_dates[0].replace("-", "")
    date_tag += "-" + (source_dates[-1][-2:] if source_dates[0][:7] == source_dates[-1][:7]
                       else source_dates[-1].replace("-", ""))
    package_name = (PACKAGE_NAME if args.city == "pohang" else
                    f"olimplanet_{args.city}_raw_json_{date_tag}_latest")
    description_name = ("olimplanet_data_description_20260904-05_latest.md"
                        if args.city == "pohang" else
                        f"olimplanet_{args.city}_data_description_{date_tag}_latest.md")

    package_root = (args.output or workspace / package_name).resolve()
    if package_root.exists():
        raise SystemExit(f"output already exists; remove or choose another path: {package_root}")
    package_root.mkdir(parents=True)
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    inside_meta = export_site(inside, package_root, f"{args.city}_inside")
    outside_meta = export_site(outside, package_root, f"{args.city}_outside")
    write_schema(package_root)
    write_package_readme(package_root, description_name, args.city)
    description = make_description(package_root, [inside_meta, outside_meta], package_name, created_at, args.city)
    description_path = workspace / description_name
    description_path.write_text(description, encoding="utf-8")
    write_manifest(package_root, {
        "created_at": created_at,
        "data_period": sorted(set(inside_meta["dates"] + outside_meta["dates"])),
        "sites": [f"{args.city}_inside", f"{args.city}_outside"],
        "counts": {
            f"{args.city}_inside": inside_meta["counts"],
            f"{args.city}_outside": outside_meta["counts"],
        },
        "attribute_matching": {
            f"{args.city}_inside": inside_meta["attribute_matching"],
            f"{args.city}_outside": outside_meta["attribute_matching"],
        },
    }, package_name)
    print(json.dumps({
        "package": str(package_root),
        "description": str(description_path),
        "inside": inside_meta,
        "outside": outside_meta,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
