from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from .attributes import infer_all
from .config import ensure_output_dirs, load_config, select_output_root
from .global_match import build_global
from .inventory import build_manifest, write_manifest
from .report import generate_report
from .spatial import build_spatial
from .tracker import TRACK_FIELDS, process_recording
from .visualize import generate_video


PERSON_FIELDS = ["local_uid", "date", "recording_id", "camera_id", "local_id", "first_seen",
    "last_seen", "last_observed", "track_span_seconds", "observed_seconds", "predicted_gap_seconds",
    "watch_seconds", "attention_seconds", "detection_count", "start_boundary_distance",
    "end_boundary_distance", "start_bev_x", "start_bev_y", "end_bev_x", "end_bev_y",
    "end_velocity_x", "end_velocity_y", "crop_path", "crop_frame_index", "crop_watch_condition",
    "crop_width", "crop_height", "crop_area", "quality_flags"]
SEGMENT_FIELDS = ["local_uid", "segment_index", "start", "end", "raw_duration_seconds",
                  "credited_watch_seconds", "credited_attention_seconds"]
ISSUE_FIELDS = ["scope", "severity", "code", "detail"]


def maybe_parquet(csv_path: Path) -> None:
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
        pd.read_csv(csv_path).to_parquet(csv_path.with_suffix(".parquet"), index=False)
    except ImportError:
        return


def inventory(cfg: dict, deep: bool) -> list:
    dirs = ensure_output_dirs(cfg)
    records = build_manifest(cfg, deep=deep)
    path = dirs["manifest"] / "recordings.csv"
    write_manifest(records, path)
    print(f"manifest: {path} ({len(records)} recordings)")
    return records


def run_local(cfg: dict, records: list, resume: bool) -> None:
    dirs = ensure_output_dirs(cfg)
    tracks_path, persons_path = dirs["local"] / "tracks.csv", dirs["local"] / "persons.csv"
    segments_path, issues_path = dirs["local"] / "watch_segments.csv", dirs["quality"] / "issues.csv"
    if resume and tracks_path.exists() and persons_path.exists() and segments_path.exists():
        print("local: reused existing outputs")
        return
    with tracks_path.open("w", encoding="utf-8", newline="") as ts, persons_path.open("w", encoding="utf-8", newline="") as ps, segments_path.open("w", encoding="utf-8", newline="") as ss, issues_path.open("w", encoding="utf-8", newline="") as qs:
        tw, pw, sw, qw = (csv.DictWriter(ts, fieldnames=TRACK_FIELDS), csv.DictWriter(ps, fieldnames=PERSON_FIELDS),
                          csv.DictWriter(ss, fieldnames=SEGMENT_FIELDS), csv.DictWriter(qs, fieldnames=ISSUE_FIELDS))
        for writer in (tw, pw, sw, qw): writer.writeheader()
        for index, record in enumerate(records, 1):
            print(f"local [{index}/{len(records)}]: {record.camera_id} {record.stem}", flush=True)
            try:
                process_recording(record, cfg, tw, pw, sw, dirs["attributes/crops"], qw)
            except Exception as exc:
                qw.writerow({"scope": record.stem, "severity": "error", "code": "recording_failed",
                             "detail": f"{type(exc).__name__}: {exc}"})
    for path in (tracks_path, persons_path, segments_path): maybe_parquet(path)


def run_all(cfg: dict, deep_inventory: bool, resume: bool, skip_inference: bool,
            skip_video: bool) -> None:
    dirs = ensure_output_dirs(cfg)
    resolved = {key: value for key, value in cfg.items() if key != "config_path"}
    (dirs["root"] / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8")
    records = inventory(cfg, deep_inventory)
    run_local(cfg, records, resume)
    attrs_path = dirs["attributes/crops"].parent / "inference_results.csv"
    if not (resume and attrs_path.exists()):
        if skip_inference:
            with attrs_path.open("w", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=["local_uid", "crop_path", "gender", "gender_score", "age",
                    "age_index", "age_estimate", "age_score", "age_vector", "reid_embedding_path", "status"]).writeheader()
        else:
            print("attributes/ReID: inference", flush=True)
            infer_all(dirs["local"] / "persons.csv", attrs_path, cfg)
    print("global: association", flush=True)
    build_global(dirs["local"] / "persons.csv", dirs["local"] / "tracks.csv", attrs_path,
                 dirs["global"], cfg)
    print("spatial: route/region/OOI aggregation", flush=True)
    build_spatial(dirs["root"], cfg)
    for path in dirs["global"].glob("*.csv"):
        maybe_parquet(path)
    if not skip_video:
        print("visualization: rendering best matches", flush=True)
        generate_video(dirs["root"], cfg)
    report = generate_report(dirs["root"], cfg)
    print(f"report: {report}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline MCMOT analytics")
    parser.add_argument("command", choices=["inventory", "run", "report", "visualize"])
    parser.add_argument("--config", default="config.pohang_inside.yaml")
    parser.add_argument("--deep", action="store_true", help="count every CSV row and ffprobe every video")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--report-name", default=None,
                        help="report 명령이 생성할 HTML 파일명 (report 디렉토리 기준)")
    parser.add_argument("--standalone-name", default=None,
                        help="report 명령이 생성할 단독 실행 HTML 파일명")
    parser.add_argument("--use-statistics", action="store_true",
                        help="attributes/statistics의 새 성별·연령 집계를 보고서에 반영")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "inventory":
        cfg = select_output_root(cfg, fresh=True)
        print(f"output: {cfg['output_root']}")
        inventory(cfg, args.deep)
    elif args.command == "run":
        cfg = select_output_root(cfg, fresh=not args.resume)
        print(f"output: {cfg['output_root']}")
        run_all(cfg, args.deep, args.resume, args.skip_inference, args.skip_video)
    elif args.command == "report":
        cfg = select_output_root(cfg, fresh=False)
        print(generate_report(Path(cfg["output_root"]), cfg,
                               filename=args.report_name or "index.html",
                               export_filename=args.standalone_name,
                               use_statistics=args.use_statistics))
    else:
        cfg = select_output_root(cfg, fresh=False)
        print(generate_video(Path(cfg["output_root"]), cfg))


if __name__ == "__main__":
    main()
