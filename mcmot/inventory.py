from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

@dataclass
class Recording:
    site: str
    camera_id: str
    date: str
    stem: str
    mac: str
    nominal_start: str
    raw_path: str = ""
    bev_path: str = ""
    video_path: str = ""
    raw_bytes: int = 0
    bev_bytes: int = 0
    video_bytes: int = 0
    raw_rows: int = -1
    bev_rows: int = -1
    csv_first_timestamp: str = ""
    csv_last_timestamp: str = ""
    video_fps: float = 0.0
    video_frames: int = 0
    video_duration: float = 0.0
    status: str = ""


def parse_stem(stem: str) -> tuple[str, str, str]:
    parts = stem.split("_")
    if len(parts) < 3:
        return "", "", ""
    return parts[0], parts[1], parts[2]


def csv_stats(path: Path) -> tuple[int, str, str]:
    count, first, last = 0, "", ""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            count += 1
            if not first:
                first = row.get("timestamp", "")
            last = row.get("timestamp", "")
    return count, first, last


def video_stats(path: Path) -> tuple[float, int, float]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=avg_frame_rate,nb_frames,duration",
               "-of", "json", str(path)]
    try:
        payload = json.loads(subprocess.check_output(command, text=True))
        stream = payload["streams"][0]
        numerator, denominator = stream.get("avg_frame_rate", "0/1").split("/")
        fps = float(numerator) / max(float(denominator), 1.0)
        frames = int(stream.get("nb_frames") or 0)
        duration = float(stream.get("duration") or (frames / fps if fps else 0))
        return fps, frames, duration
    except Exception:
        return 0.0, 0, 0.0


def safe_glob(path: Path, pattern: str):
    try:
        yield from path.glob(pattern)
    except OSError:
        return


def build_manifest(cfg: dict, deep: bool = False) -> list[Recording]:
    site, dates = cfg["site"], set(cfg["dates"])
    data_site = Path(cfg["data_root"]) / site
    video_root = Path(cfg["video_root"])
    records: dict[tuple[str, str], Recording] = {}
    cameras = sorted(path.name for path in data_site.iterdir() if path.is_dir())
    for camera in cameras:
        for kind in ("raw", "bev"):
            # The original Pohang export separates raw/ and bev/.  Gumi keeps
            # both CSVs beside the video in camera/date directories.
            paths = safe_glob(data_site / camera / kind, "*/*.csv")
            if kind == "raw":
                paths = (path for path in paths
                         if not path.name.endswith(("_260922.csv", "_pose_only.csv")))
            if not (data_site / camera / kind).is_dir():
                pattern = "*/*_bev.csv" if kind == "bev" else "*/*.csv"
                paths = (path for path in safe_glob(data_site / camera, pattern)
                         if kind == "bev" or (not path.name.endswith("_bev.csv")
                                              and not path.name.endswith(("_260922.csv",
                                                                         "_pose_only.csv"))))
            for path in paths:
                date, mac, start = parse_stem(path.stem)
                recording_stem = path.stem[:-4] if kind == "bev" and path.stem.endswith("_bev") else path.stem
                if date not in dates:
                    continue
                key = (camera, recording_stem)
                rec = records.setdefault(key, Recording(site, camera, date, recording_stem, mac, start))
                setattr(rec, f"{kind}_path", str(path.resolve()))
                setattr(rec, f"{kind}_bytes", path.stat().st_size)
        for date in dates:
            for path in safe_glob(video_root / camera / date, "*.mp4"):
                parsed_date, mac, start = parse_stem(path.stem)
                if parsed_date != date:
                    continue
                key = (camera, path.stem)
                rec = records.setdefault(key, Recording(site, camera, date, path.stem, mac, start))
                rec.video_path, rec.video_bytes = str(path.resolve()), path.stat().st_size
    for rec in records.values():
        missing = [kind for kind in ("raw", "bev", "video") if not getattr(rec, f"{kind}_path")]
        rec.status = "ok" if not missing else "missing_" + "_".join(missing)
        if deep:
            if rec.raw_path:
                rec.raw_rows, rec.csv_first_timestamp, rec.csv_last_timestamp = csv_stats(Path(rec.raw_path))
            if rec.bev_path:
                rec.bev_rows, first, last = csv_stats(Path(rec.bev_path))
                rec.csv_first_timestamp = rec.csv_first_timestamp or first
                rec.csv_last_timestamp = rec.csv_last_timestamp or last
            if rec.video_path:
                rec.video_fps, rec.video_frames, rec.video_duration = video_stats(Path(rec.video_path))
                if rec.video_fps <= 0 or rec.video_frames <= 0 or rec.video_duration <= 0:
                    rec.status += "|invalid_video"
            if rec.raw_rows >= 0 and rec.bev_rows >= 0 and rec.raw_rows != rec.bev_rows:
                rec.status += "|row_mismatch"
    return sorted(records.values(), key=lambda r: (r.date, r.nominal_start, r.camera_id, r.stem))


def write_manifest(records: list[Recording], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(records[0]).keys()) if records else list(Recording.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in records)
