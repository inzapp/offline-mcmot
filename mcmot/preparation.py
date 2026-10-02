"""Input overview, calibration and streaming conversion of raw CSV to BEV CSV."""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .geometry import homography_from_points, transform_point
from .inventory import build_manifest

BEV_FIELDS = ["timestamp", "frame_index", "x", "y"]


def section_root(cfg):
    return Path(cfg["data_path"]) if cfg.get("data_path") else Path(cfg["data_root"]) / cfg["site"]


def image_candidates(root, name):
    return sorted(p for p in root.glob(f"{name}.*") if p.suffix.lower() in (".jpg", ".png"))


def bev_image(cfg):
    if cfg.get("bev_image"):
        return Path(cfg["bev_image"])
    candidates = image_candidates(section_root(cfg), "bev")
    if len(candidates) != 1:
        raise ValueError(f"BEV 이미지가 정확히 1개 필요합니다: {candidates}")
    return candidates[0]


def roi_path(cfg, camera):
    return Path(cfg["roi_file"]) if cfg.get("roi_file") else section_root(cfg) / camera / f"{camera}_roi.json"


def camera_payload(path, camera=None):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if camera is not None and camera in data:
        data = data[camera]
        if "rois" not in data:
            data = {"rois": [data]}
    return data


def calibration_path(cfg, camera):
    selected = cfg.get("calibration", {}).get("files", {}).get(str(camera))
    if selected:
        path = Path(selected).expanduser()
        return path if path.is_absolute() else Path(cfg["config_path"]).parent / path
    return section_root(cfg) / camera / "calibration" / "homography.json"


def camera_matrix(cfg, camera):
    path = calibration_path(cfg, camera)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if data.get("direction") != "image1_to_image2":
            raise ValueError(f"지원하지 않는 행렬 방향: {path}; image1=BEV, image2=camera 필요")
        sizes = np.asarray(data["image_sizes"], dtype=float)
        if sizes.shape != (2, 2) or not np.isfinite(sizes).all() or (sizes <= 0).any():
            raise ValueError(f"잘못된 보정 이미지 크기: {path}")
        # Input/output points are normalized; the saved matrix is in original pixels.
        inverse = np.linalg.inv(np.asarray(data["H"], dtype=float))
        matrix = np.diag([1 / sizes[0, 0], 1 / sizes[0, 1], 1]) @ inverse @ np.diag([*sizes[1], 1])
    else:
        path = roi_path(cfg, camera)
        data = camera_payload(path, camera)
        sources, targets = [], []
        for roi in data.get("rois", []):
            sources.extend(roi.get("image2_vertices_normalized", []))
            targets.extend(roi.get("image1_vertices_normalized", []))
        if len(sources) < 4 or len(sources) != len(targets):
            raise ValueError(f"카메라/BEV 대응점이 최소 4쌍 필요합니다: {path}")
        matrix = homography_from_points(sources, targets)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError(f"유효하지 않은 homography: {path}")
    return matrix


def target_bev(record):
    # New layout always writes beside the MP4. Existing split exports remain readable.
    if record.video_path:
        return Path(record.video_path).with_name(record.stem + "_bev.csv")
    return Path(record.raw_path).with_name(record.stem + "_bev.csv")


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def convert_bev(raw_path, output_path, matrix, force=False):
    """Preserve every row, including points outside ROI and normalized image bounds."""
    raw_path, output_path = Path(raw_path), Path(output_path)
    if raw_path.resolve() == output_path.resolve():
        raise ValueError("raw와 BEV 출력 경로가 동일합니다")
    meta_path = output_path.with_suffix(".csv.meta.json")
    signature = {"raw": file_signature(raw_path), "matrix": np.asarray(matrix).tolist(), "version": 1}
    if output_path.exists() and not force:
        if meta_path.exists():
            previous = json.loads(meta_path.read_text())
            if previous.get("inputs") == signature and previous.get("output") == file_signature(output_path):
                print(f"BEV 재사용: {output_path}", flush=True)
                return output_path
        raise FileExistsError(f"기존 BEV를 덮어쓰려면 --force 필요: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(".csv.partial")
    count = 0
    try:
        with raw_path.open(encoding="utf-8-sig", newline="") as source, partial.open("w", encoding="utf-8", newline="") as dest:
            reader = csv.DictReader(source)
            required = {"timestamp", "frame_index", "cx", "cy", "h"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"raw 필수 컬럼 누락: {required - set(reader.fieldnames or [])}")
            writer = csv.DictWriter(dest, fieldnames=BEV_FIELDS)
            writer.writeheader()
            for number, row in enumerate(reader, 2):
                point = (float(row["cx"]), float(row["cy"]) + float(row["h"]) / 2)
                projected = np.asarray(matrix) @ np.array([*point, 1])
                if not np.isfinite(projected).all() or abs(projected[2]) < 1e-12:
                    raise ValueError(f"투영 불가능한 좌표: {raw_path}:{number}")
                x, y = projected[:2] / projected[2]
                writer.writerow({"timestamp": row["timestamp"], "frame_index": row["frame_index"], "x": x, "y": y})
                count += 1
        partial.replace(output_path)
        meta_path.write_text(json.dumps({"inputs": signature, "output": file_signature(output_path), "rows": count}, indent=2))
    finally:
        if partial.exists():
            partial.unlink()
    print(f"BEV 생성: {output_path} ({count:,}행)", flush=True)
    return output_path


def show_status(cfg, *, color=None):
    """Display file availability and filename matching as an English tree."""
    root = section_root(cfg)
    if color is None:
        color = sys.stdout.isatty() and "NO_COLOR" not in os.environ

    def mark(ok):
        symbol = "✓" if ok else "✗"
        return f"\033[{32 if ok else 31}m{symbol}\033[0m" if color else symbol

    def file_node(path, label, parent=None):
        path = Path(path)
        name = os.path.relpath(path, parent) if parent is not None and path.is_relative_to(root) else str(path)
        exists = path.is_file()
        return (f"{mark(exists)} {name} [{label}]" + ("" if exists else " (missing)"), [])

    def image_nodes(parent, name, label, selected=None):
        images = image_candidates(parent, name)
        if len(images) > 1:
            return [(f"{mark(False)} {label}: expected one JPG/PNG; found {len(images)}",
                     [file_node(p, label, parent) for p in images])], images
        path = selected or (images[0] if images else parent / f"{name}.{{jpg,png}}")
        return [file_node(path, label, parent)], images

    records = build_manifest(cfg) if root.is_dir() else []
    cameras = sorted(({p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()}
                      if root.is_dir() else set()) | {r.camera_id for r in records})
    selected = Path(cfg["bev_image"]) if cfg.get("bev_image") else None
    nodes, bev_images = image_nodes(root, "bev", "BEV image", selected)
    if cfg.get("roi_file"):
        nodes.append(file_node(cfg["roi_file"], "shared ROI", root))
    totals = Counter()
    camera_images = 0
    for camera in cameras:
        folder = root / camera
        rows = [r for r in records if r.camera_id == camera]
        counts = {"mp4": sum(bool(r.video_path) for r in rows),
                  "raw": sum(bool(r.raw_path) for r in rows),
                  "bev": sum(bool(r.bev_path) for r in rows),
                  "vr": sum(bool(r.video_path and r.raw_path) for r in rows),
                  "vb": sum(bool(r.video_path and r.bev_path) for r in rows),
                  "all": sum(r.status == "ok" for r in rows)}
        totals.update(counts)
        children, images = image_nodes(folder, camera, "camera image")
        camera_images += len(images)
        roi = roi_path(cfg, camera)
        children.append(file_node(roi, "ROI", folder))
        calibration = calibration_path(cfg, camera)
        if calibration.is_file():
            children.append(file_node(calibration, "homography", folder))
        else:
            children.append((f"{mark(roi.is_file())} Homography from ROI correspondences"
                             + ("" if roi.is_file() else " (ROI missing)"), []))
        for date in sorted(set(cfg["dates"])):
            recordings = [r for r in rows if r.date == date]
            files = []
            for r in recordings:
                video = Path(r.video_path) if r.video_path else Path(cfg["video_root"]) / camera / date / (r.stem + ".mp4")
                raw = Path(r.raw_path) if r.raw_path else video.with_suffix(".csv")
                bev = Path(r.bev_path) if r.bev_path else video.with_name(r.stem + "_bev.csv")
                for path, label in ((video, "MP4"), (raw, "RAW"), (bev, "BEV CSV")):
                    files.append(file_node(path, label, folder / date))
            if not files:
                files.append((f"{mark(False)} No recordings found", []))
            children.append((date + "/", files))
        label = (f"{camera}/  (images: {len(images)} | MP4: {counts['mp4']} | "
                 f"RAW: {counts['raw']} | BEV: {counts['bev']} | matched: {counts['all']}/{len(rows)})")
        nodes.append((label, children))
    if not cameras:
        nodes.append((f"{mark(False)} No cameras or recordings found", []))

    def print_tree(children, prefix=""):
        for index, (label, nested) in enumerate(children):
            last = index == len(children) - 1
            print(prefix + ("└── " if last else "├── ") + label)
            print_tree(nested, prefix + ("    " if last else "│   "))

    print(f"\nInput status: {cfg['site']}")
    print(f"Dates: {', '.join(cfg['dates'])}")
    if Path(cfg["video_root"]) != root:
        print(f"Video root: {cfg['video_root']}")
    print(f"{mark(root.is_dir())} {root}/")
    print_tree(nodes)
    bev_count = len(bev_images) or int(bool(selected and selected.is_file()))
    print(f"\nFiles: BEV images {bev_count} | Camera images {camera_images} | "
          f"MP4 {totals['mp4']} | RAW {totals['raw']} | BEV CSV {totals['bev']}")
    complete = bool(records) and totals['all'] == len(records)
    print(f"{mark(complete)} Matching: MP4 ↔ RAW {totals['vr']} | MP4 ↔ BEV {totals['vb']} | "
          f"Complete {totals['all']}/{len(records)}")
    missing = Counter(r.status.removeprefix("missing_") for r in records if r.status != "ok")
    if missing:
        print("Missing files: " + "; ".join(f"{key}: {count} recording(s)" for key, count in sorted(missing.items())))
    print("File availability and filename matching only. Check CSV contents with validate --deep.\n", flush=True)
    return records


def validate_inputs(cfg, deep=False):
    errors, warnings = [], []
    try:
        if cfg.get('data_path') and len(image_candidates(section_root(cfg), 'bev')) != 1:
            raise ValueError('data_path/bev.jpg 또는 bev.png가 정확히 1개 필요합니다')
        image = bev_image(cfg)
        if cv2.imread(str(image)) is None:
            errors.append(f"BEV 이미지 읽기 실패: {image}")
    except Exception as exc:
        errors.append(str(exc))
    records = build_manifest(cfg, deep=deep)
    if not records:
        errors.append("대상 녹화 데이터가 없습니다")
    for camera in sorted({r.camera_id for r in records}):
        try:
            camera_matrix(cfg, camera)
            rois = camera_payload(roi_path(cfg, camera), camera).get("rois", [])
            if not rois:
                raise ValueError("유효 ROI 없음")
            for roi in rois:
                for key in ("image1_vertices_normalized", "image2_vertices_normalized"):
                    points = np.asarray(roi.get(key, []), dtype=float)
                    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
                        raise ValueError(f"잘못된 ROI: {key}")
            if cfg.get("data_path") and len(image_candidates(section_root(cfg) / camera, camera)) != 1:
                raise ValueError("카메라 샘플 jpg/png가 정확히 1개 필요합니다")
        except Exception as exc:
            errors.append(f"카메라 {camera}: {exc}")
    for r in records:
        if r.status != "ok":
            errors.append(f"{r.camera_id}/{r.stem}: {r.status}")
        if r.raw_path and r.bev_path:
            try:
                with Path(r.raw_path).open(encoding="utf-8-sig", newline="") as raw, Path(r.bev_path).open(encoding="utf-8-sig", newline="") as bev:
                    rr, br = csv.DictReader(raw), csv.DictReader(bev)
                    from .raw import csv_columns
                    if not set(csv_columns()).issubset(rr.fieldnames or []) or br.fieldnames != BEV_FIELDS:
                        raise ValueError("raw/BEV 컬럼 형식 불일치")
                    if deep:
                        last_frame, last_time = -1, None
                        start = datetime.strptime(f"{r.date}_{r.nominal_start}", "%Y-%m-%d_%H%M%S")
                        for line, (a, b) in enumerate(itertools.zip_longest(rr, br), 2):
                            if a is None or b is None or (a['timestamp'], a['frame_index']) != (b['timestamp'], b['frame_index']):
                                raise ValueError(f"{line}행 raw/BEV 매칭 실패")
                            frame = int(a['frame_index']); stamp = datetime.fromisoformat(a['timestamp'])
                            if frame < 0 or frame < last_frame or (last_time and stamp < last_time) or stamp < start:
                                raise ValueError(f"{line}행 시간/프레임 순서 오류")
                            if r.video_frames and frame >= r.video_frames:
                                raise ValueError(f"{line}행 영상 프레임 범위 초과")
                            if not all(math.isfinite(float(a[k])) for k in csv_columns() if k not in ('timestamp', 'frame_index')) or not all(math.isfinite(float(b[k])) for k in ('x', 'y')):
                                raise ValueError(f"{line}행 유효하지 않은 좌표")
                            last_frame, last_time = frame, stamp
            except Exception as exc:
                errors.append(f"{r.camera_id}/{r.stem}: {exc}")
        if deep and r.video_duration and abs(r.video_duration - 1790) > cfg.get('recording', {}).get('duration_tolerance_seconds', 3):
            warnings.append(f"{r.camera_id}/{r.stem}: 영상 길이 {r.video_duration:.1f}초 (예상 1790초)")
    for message in warnings:
        print(f"WARNING: {message}")
    for message in errors:
        print(f"ERROR: {message}")
    print(f"검증: {len(records)}개 녹화 / 오류 {len(errors)} / 경고 {len(warnings)}", flush=True)
    return not errors
