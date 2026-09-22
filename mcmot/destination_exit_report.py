from __future__ import annotations

import argparse
import csv
import html
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from .export_html import export_standalone_html
from .geometry import point_in_polygon


PERSON_FIELDS = ["global_id", "date", "disappeared_at", "hour", "destination",
                 "gender", "age", "age_group", "attribute_crop", "input_mode",
                 "face_status", "camera_id", "local_uid", "last_bev_x",
                 "last_bev_y", "interior_last_seen"]


def _rows(path: Path):
    # utf-8-sig accepts both ordinary UTF-8 and Excel-friendly BOM CSV files.
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _age_group(value: str) -> str:
    try:
        age = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if age < 0:
        return "invalid_negative"
    decade = int(age) // 10 * 10
    return "80+" if decade >= 80 else f"{decade}-{decade + 9}"


def _new_attributes(results_path: Path) -> dict[str, dict]:
    if not results_path.exists():
        return {}
    return {os.path.basename(row.get("file", "")): row for row in _rows(results_path)
            if row.get("status") == "ok" and row.get("file")}


def find_destination_exits(tracks_path: Path, people_path: Path,
                           routes: list[dict], results_path: Path | None = None) -> list[dict]:
    """Find people whose final observed position is a destination-side gate.

    ``start_gate`` is the facility-side gate. For each facility, the interior
    is the remaining valid BEV ROI outside that gate. A person qualifies only
    when an interior point was observed before the final facility-gate point.
    Predicted track rows are deliberately ignored.
    """
    people = {row["global_id"]: row for row in _rows(people_path)}
    attributes = _new_attributes(results_path) if results_path else {}
    states: dict[str, dict] = defaultdict(
        lambda: {"last_time": "", "last_rows": [], "interior": [""] * len(routes)})

    for row in _rows(tracks_path):
        if (not row.get("global_id") or row.get("is_observed") != "1" or
                row.get("bev_x") in (None, "") or row.get("bev_y") in (None, "")):
            continue
        gid, when = row["global_id"], row["timestamp"]
        point = (float(row["bev_x"]), float(row["bev_y"]))
        state = states[gid]
        if str(row.get("inside_bev_roi", "")).lower() in ("1", "true"):
            for index, route in enumerate(routes):
                if not point_in_polygon(point, route["start_gate"]):
                    state["interior"][index] = max(state["interior"][index], when)
        if when > state["last_time"]:
            state["last_time"], state["last_rows"] = when, [row]
        elif when == state["last_time"]:
            state["last_rows"].append(row)

    results = []
    for gid, state in states.items():
        if not state["last_time"]:
            continue
        person = people.get(gid, {})
        crop_name = os.path.basename(person.get("representative_crop_path", ""))
        attribute = attributes.get(crop_name, {})
        for index, route in enumerate(routes):
            final_rows = [row for row in state["last_rows"] if point_in_polygon(
                (float(row["bev_x"]), float(row["bev_y"])), route["start_gate"])]
            interior_time = state["interior"][index]
            if not final_rows or not interior_time or interior_time >= state["last_time"]:
                continue
            row = final_rows[0]
            moment = datetime.fromisoformat(state["last_time"])
            results.append({
                "global_id": gid,
                "date": person.get("date") or moment.strftime("%Y-%m-%d"),
                "disappeared_at": state["last_time"],
                "hour": moment.strftime("%H:00"),
                "destination": route["name"],
                "gender": attribute.get("gender") or "unknown",
                "age": attribute.get("age") or "unknown",
                "age_group": _age_group(attribute.get("age", "")),
                "attribute_crop": crop_name,
                "input_mode": attribute.get("input_mode", ""),
                "face_status": attribute.get("face_status", ""),
                "camera_id": row.get("camera_id", ""),
                "local_uid": row.get("local_uid", ""),
                "last_bev_x": row["bev_x"],
                "last_bev_y": row["bev_y"],
                "interior_last_seen": interior_time,
            })
    return sorted(results, key=lambda row: (row["date"], row["disappeared_at"],
                                            row["destination"], row["global_id"]))


def aggregate(events: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    counts = Counter(tuple(row[key] for key in dimensions) for row in events)
    totals = Counter((row["date"], row["hour"], row["destination"]) for row in events)
    result = []
    for key, count in sorted(counts.items()):
        values = dict(zip(dimensions, key))
        total = totals[(values["date"], values["hour"], values["destination"])]
        result.append({**values, "person_count": count,
                       "percent": f"{count / total * 100:.2f}" if total else "0.00"})
    return result


def area_demographics(events: list[dict], dimensions: tuple[str, ...]) -> list[dict]:
    counts = Counter(tuple(row[key] for key in dimensions) for row in events)
    destination_totals = Counter(row["destination"] for row in events)
    gender_totals = Counter((row["destination"], row["gender"]) for row in events)
    result = []
    for key, count in sorted(counts.items()):
        values = dict(zip(dimensions, key))
        total = destination_totals[values["destination"]]
        item = {**values, "person_count": count,
                "percent_of_destination": f"{count / total * 100:.2f}" if total else "0.00"}
        if "gender" in values:
            gender_total = gender_totals[(values["destination"], values["gender"])]
            item["percent_within_gender"] = (
                f"{count / gender_total * 100:.2f}" if gender_total else "0.00")
        result.append(item)
    return result


def _table(rows: list[dict], fields: list[str]) -> str:
    if not rows:
        return '<div class="empty">해당 인원이 없습니다.</div>'
    headings = {"destination": "이동 시설", "person_count": "인원", "percent": "비율(%)",
                "date": "날짜", "hour": "시간", "gender": "성별", "age": "연령"}
    head = "".join(f"<th>{html.escape(headings.get(key, key))}</th>" for key in fields)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>"
                                      for key in fields) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def generate(output_root: Path, cfg: dict) -> tuple[Path, Path]:
    routes = cfg.get("spatial", {}).get("routes", [])
    events = find_destination_exits(output_root / "global/tracks.csv",
                                    output_root / "global/persons.csv", routes,
                                    output_root / "attributes/results.csv")
    export_dir = output_root / "spatial/destination_exits"
    _write(export_dir / "persons.csv", PERSON_FIELDS, events)

    dimensions = {
        "gender.csv": ("date", "hour", "destination", "gender"),
        "age.csv": ("date", "hour", "destination", "age_group"),
        "gender_age.csv": ("date", "hour", "destination", "gender", "age_group"),
    }
    aggregates = {}
    for filename, fields in dimensions.items():
        aggregates[filename] = aggregate(events, fields)
        _write(export_dir / filename, [*fields, "person_count", "percent"], aggregates[filename])

    destination_counts = Counter(row["destination"] for row in events)
    overview = [{"destination": route["name"], "person_count": destination_counts[route["name"]]}
                for route in routes]
    _write(export_dir / "overview.csv", ["destination", "person_count"], overview)

    area_age = area_demographics(events, ("destination", "age_group"))
    area_gender_age = area_demographics(events, ("destination", "gender", "age_group"))
    _write(export_dir / "area_age_new_model.csv",
           ["destination", "age_group", "person_count", "percent_of_destination"], area_age)
    _write(export_dir / "area_gender_age_new_model.csv",
           ["destination", "gender", "age_group", "person_count",
            "percent_of_destination", "percent_within_gender"], area_gender_age)

    total = len(events)
    unknown_gender = sum(row["gender"] == "unknown" for row in events)
    unknown_age = sum(row["age"] == "unknown" for row in events)
    overall_gender = Counter(row["gender"] for row in events)
    overall_age = Counter(row["age_group"] for row in events)
    overall_cross = Counter((row["gender"], row["age_group"]) for row in events)
    gender_rows = [{"gender": key, "person_count": value,
                    "percent": f"{value / total * 100:.2f}" if total else "0.00"}
                   for key, value in sorted(overall_gender.items())]
    age_rows = [{"age": key, "person_count": value,
                 "percent": f"{value / total * 100:.2f}" if total else "0.00"}
                for key, value in sorted(overall_age.items())]
    cross_rows = [{"gender": key[0], "age": key[1], "person_count": value,
                   "percent": f"{value / total * 100:.2f}" if total else "0.00"}
                  for key, value in sorted(overall_cross.items())]
    area_sections = "".join(
        f'<section><h2>{html.escape(route["name"])} — 새 모델 상세</h2>'
        f'<p class="note">비율의 분모는 이 영역의 최종 소실 인원 {destination_counts[route["name"]]:,}명입니다. '
        f'성별×연령 표의 성별 내 비율은 해당 영역의 남성 또는 여성 인원을 각각 분모로 사용합니다.</p>'
        f'<h3>연령대</h3>{_table([row for row in area_age if row["destination"] == route["name"]], ["age_group", "person_count", "percent_of_destination"])}'
        f'<h3>성별 × 연령대</h3>{_table([row for row in area_gender_age if row["destination"] == route["name"]], ["gender", "age_group", "person_count", "percent_of_destination", "percent_within_gender"])}</section>'
        for route in routes)

    cards = "".join(
        f'<div class="card"><small>{html.escape(row["destination"])}</small><b>{row["person_count"]:,}</b><span>명</span></div>'
        for row in overview)
    dates = ", ".join(cfg.get("dates", []))
    content = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>시설 방향 이동 후 최종 소실 인원 보고서</title>
<style>:root{{--ink:#132337;--muted:#607086;--line:#dce4eb;--bg:#f3f6f8;--blue:#087ea4;--red:#ba3d31}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,Pretendard,"Noto Sans KR",sans-serif}}header{{padding:44px max(5vw,28px);color:white;background:linear-gradient(125deg,#102940,#087e91)}}header h1{{margin:0 0 10px;font-size:36px}}header p{{color:#d7edf1;line-height:1.7}}main{{max-width:1450px;margin:auto;padding:28px}}section{{background:white;border:1px solid var(--line);border-radius:16px;padding:24px;margin:20px 0}}h2{{margin-top:0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}}.card{{background:white;border:1px solid var(--line);border-radius:13px;padding:18px;display:flex;flex-direction:column;gap:5px}}.card b{{font-size:30px;color:var(--blue)}}small,.card span{{color:var(--muted)}}.note{{border-left:4px solid var(--blue);padding:12px 16px;background:#edf8fb;line-height:1.65}}.warn{{border-color:var(--red);background:#fff3f0}}.table-wrap{{overflow:auto;max-height:620px;border:1px solid var(--line);border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{position:sticky;top:0;background:#eaf2f5}}.pair{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}}.empty{{padding:24px;text-align:center;color:var(--muted)}}</style></head>
<body><header><h1>시설 방향 이동 후 최종 소실 인원 보고서</h1><p>{html.escape(cfg.get("site", ""))} · {html.escape(dates)} · Global ID 기준</p></header><main>
<div class="cards"><div class="card"><small>전체 시설 방향 소실 이벤트</small><b>{total:,}</b><span>명·시설 건수</span></div>{cards}</div>
<section><h2>산정 기준</h2><p class="note"><b>시설 gate를 제외한 나머지 유효 BEV ROI에서 관측 → 시설 gate 진입 → Global ID의 마지막 실제 관측점이 시설 gate 안에서 종료</b>한 경우만 포함했습니다. 예측 좌표는 제외했고, 카메라 간 연결된 사람은 Global ID의 최종 관측점을 사용했습니다. 전시장 항목은 현재 설정에 따라 화장실·카페 방향을 포함합니다.</p><p class="note warn">최종 소실은 시설 입장을 강하게 시사하는 관측 기준이며, 가림·추적 누락·녹화 종료와 실제 입장을 완전히 구분하지는 못합니다. 한 사람이 서로 다른 시설에서 각각 조건을 충족하면 시설별로 집계될 수 있습니다.</p></section>
<section><h2>시설별 인원</h2>{_table(overview, ["destination", "person_count"])}</section>
<section><h2>전체 성별·연령 분포 — 새 모델</h2><p class="note"><code>attributes/results.csv</code>의 개별 crop 추론값을 Global ID 대표 crop에 연결했습니다. 성별 unknown {unknown_gender:,}건, 연령 unknown {unknown_age:,}건을 임의 보정하지 않고 별도 범주로 유지했습니다.</p><div class="pair"><div><h3>성별</h3>{_table(gender_rows, ["gender", "person_count", "percent"])}</div><div><h3>연령</h3>{_table(age_rows, ["age", "person_count", "percent"])}</div></div><h3>성별 × 연령</h3>{_table(cross_rows, ["gender", "age", "person_count", "percent"])}</section>
{area_sections}
<section><h2>시간대별 성별</h2>{_table(aggregates["gender.csv"], ["date", "hour", "destination", "gender", "person_count", "percent"])}</section>
<section><h2>시간대별 연령</h2>{_table(aggregates["age.csv"], ["date", "hour", "destination", "age_group", "person_count", "percent"])}</section>
<section><h2>시간대별 성별 × 연령</h2>{_table(aggregates["gender_age.csv"], ["date", "hour", "destination", "gender", "age_group", "person_count", "percent"])}</section>
<section><h2>내보내기 파일</h2><p>Excel에서 바로 열 수 있도록 UTF-8 BOM CSV로 저장했습니다.</p><ul><li><a href="../spatial/destination_exits/overview.csv">시설별 요약</a></li><li><a href="../spatial/destination_exits/area_age_new_model.csv">영역별 연령대 — 새 모델</a></li><li><a href="../spatial/destination_exits/area_gender_age_new_model.csv">영역별 성별×연령대 — 새 모델</a></li><li><a href="../spatial/destination_exits/gender.csv">시간대별 성별</a></li><li><a href="../spatial/destination_exits/age.csv">시간대별 연령</a></li><li><a href="../spatial/destination_exits/gender_age.csv">시간대별 성별×연령</a></li><li><a href="../spatial/destination_exits/persons.csv">판정 대상자 상세</a></li></ul></section>
</main></body></html>'''
    report = output_root / "report/destination_exit_report.html"
    report.write_text(content, encoding="utf-8")
    standalone = report.with_name("destination_exit_report_export.html")
    export_standalone_html(report, standalone)
    return report, standalone


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate destination-direction final disappearance report")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    config_path = args.config or args.output / "resolved_config.yaml"
    with config_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    report, standalone = generate(args.output, cfg)
    print(report)
    print(standalone)


if __name__ == "__main__":
    main()
