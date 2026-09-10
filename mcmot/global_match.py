from __future__ import annotations

import csv
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


class UnionFind:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def embedding_distance(a: str, b: str, embeddings: dict[str, np.ndarray]) -> float | None:
    left, right = embeddings.get(a), embeddings.get(b)
    if left is None or right is None:
        return None
    return float(np.linalg.norm(left - right))


def overlap_seconds(a: dict, b: dict) -> float:
    return (min(dt(a["last_observed"]), dt(b["last_observed"])) -
            max(dt(a["first_seen"]), dt(b["first_seen"]))).total_seconds()


def union_duration(intervals: list[tuple[datetime, datetime]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += max(0.0, (end - start).total_seconds())
            start, end = next_start, next_end
    return total + max(0.0, (end - start).total_seconds())


def session_key(row: dict) -> tuple[str, str]:
    """Map irregular restarts to their scheduled 30-minute recording window."""
    when = dt(row["first_seen"])
    minute = 0 if when.minute < 30 else 30
    return row["date"], f"{when.hour:02d}{minute:02d}00"


def temporal_candidate_pairs(rows: list[dict], max_gap: float):
    """Yield cross-camera pairs whose intervals overlap or are within max_gap."""
    by_camera = defaultdict(list)
    for row in rows:
        by_camera[row["camera_id"]].append(row)
    cameras = sorted(by_camera)
    for left_index, left_camera in enumerate(cameras):
        left_rows = by_camera[left_camera]
        for right_camera in cameras[left_index + 1:]:
            right_rows = sorted(by_camera[right_camera], key=lambda row: dt(row["first_seen"]))
            starts = [dt(row["first_seen"]).timestamp() for row in right_rows]
            ends = [dt(row["last_observed"]).timestamp() for row in right_rows]
            for left in left_rows:
                lower = dt(left["first_seen"]).timestamp() - max_gap
                upper = dt(left["last_observed"]).timestamp() + max_gap
                stop = bisect_right(starts, upper)
                # Starts are ordered; filtering end avoids pairs that ended too early.
                for index in range(stop):
                    if ends[index] >= lower:
                        yield left, right_rows[index]


def build_global(persons_path: Path, tracks_path: Path, attrs_path: Path,
                 out_dir: Path, cfg: dict) -> None:
    persons = read_rows(persons_path)
    attrs = {row["local_uid"]: row for row in read_rows(attrs_path)}
    embeddings = {}
    for uid, row in attrs.items():
        path = row.get("reid_embedding_path")
        if path:
            try:
                embeddings[uid] = np.load(path)
            except Exception:
                pass
    gm, rcfg = cfg["global_matching"], cfg["reid"]
    threshold = float(rcfg["distance_threshold"])
    by_record: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in persons:
        by_record[session_key(row)].append(row)
    boundary_thresholds = {}
    for camera in {row["camera_id"] for row in persons}:
        values = [float(row[key]) for row in persons if row["camera_id"] == camera
                  for key in ("start_boundary_distance", "end_boundary_distance")
                  if row.get(key) not in (None, "")]
        boundary_thresholds[camera] = (float(np.quantile(values, float(gm["boundary_quantile"])))
                                       if values else math.inf)
    candidates = []
    topology_counts = defaultdict(int)
    for (date, session), rows in by_record.items():
        for left, right in temporal_candidate_pairs(rows, float(gm["max_handoff_seconds"])):
                reid_d = embedding_distance(left["local_uid"], right["local_uid"], embeddings)
                overlap = overlap_seconds(left, right)
                kind, temporal_gap, bev_distance = "", 0.0, math.inf
                if overlap >= -float(gm["sync_tolerance_seconds"]):
                    ax = (float(left["start_bev_x"]) + float(left["end_bev_x"])) / 2
                    ay = (float(left["start_bev_y"]) + float(left["end_bev_y"])) / 2
                    bx = (float(right["start_bev_x"]) + float(right["end_bev_x"])) / 2
                    by = (float(right["start_bev_y"]) + float(right["end_bev_y"])) / 2
                    bev_distance = math.hypot(ax - bx, ay - by)
                    if bev_distance <= float(gm["simultaneous_bev_distance"]):
                        kind = "simultaneous_overlap"
                else:
                    first, second = (left, right) if dt(left["last_observed"]) < dt(right["first_seen"]) else (right, left)
                    temporal_gap = (dt(second["first_seen"]) - dt(first["last_observed"])).total_seconds()
                    bev_distance = math.hypot(float(first["end_bev_x"]) - float(second["start_bev_x"]),
                                              float(first["end_bev_y"]) - float(second["start_bev_y"]))
                    reachable = bev_distance <= float(gm["max_bev_speed_per_second"]) * max(temporal_gap, 0.1)
                    boundary_ok = (float(first["end_boundary_distance"]) <= boundary_thresholds[first["camera_id"]]
                                   and float(second["start_boundary_distance"]) <= boundary_thresholds[second["camera_id"]])
                    if boundary_ok and 0 <= temporal_gap <= float(gm["max_handoff_seconds"]) and reachable:
                        kind = "handoff"
                if kind:
                    bev_norm = min(1.0, bev_distance / max(float(gm["simultaneous_bev_distance"]), 1e-6))
                    time_norm = min(1.0, max(0.0, temporal_gap) / max(float(gm["max_handoff_seconds"]), 1e-6))
                    weights = gm["weights"]
                    reid_term = (reid_d / threshold) if reid_d is not None else 10.0
                    score = (float(weights["reid"]) * reid_term +
                             float(weights["bev"]) * bev_norm + float(weights["time"]) * time_norm)
                    eligible = reid_d is not None and reid_d < threshold
                    if kind == "handoff" and eligible:
                        topology_counts[(first["camera_id"], second["camera_id"])] += 1
                    candidates.append({"date": date, "session": session, "left_uid": left["local_uid"],
                        "right_uid": right["local_uid"], "match_type": kind,
                        "reid_distance": "" if reid_d is None else reid_d,
                        "reid_threshold": threshold, "bev_distance": bev_distance,
                        "temporal_gap_seconds": temporal_gap, "association_score": score,
                        "gate_status": "eligible" if eligible else ("reid_unavailable" if reid_d is None else "reid_rejected"),
                        "accepted": 0})
    uf = UnionFind([row["local_uid"] for row in persons])
    members = {row["local_uid"]: [row] for row in persons}
    for edge in sorted(candidates, key=lambda x: x["association_score"]):
        if edge["gate_status"] != "eligible":
            continue
        a, b = uf.find(edge["left_uid"]), uf.find(edge["right_uid"])
        if a == b:
            edge["accepted"] = 1
            continue
        combined = members[a] + members[b]
        conflict = False
        for i, left in enumerate(combined):
            for right in combined[i + 1:]:
                if left["camera_id"] == right["camera_id"] and overlap_seconds(left, right) > 0:
                    conflict = True
        if conflict:
            continue
        uf.union(a, b)
        root = uf.find(a)
        members[root] = combined
        edge["accepted"] = 1
    roots = defaultdict(list)
    for row in persons:
        roots[(row["date"], uf.find(row["local_uid"]))].append(row)
    global_ids = {}
    for date in sorted({row["date"] for row in persons}):
        date_groups = [(key, rows) for key, rows in roots.items() if key[0] == date]
        date_groups.sort(key=lambda item: min(dt(x["first_seen"]) for x in item[1]))
        for index, (_, rows) in enumerate(date_groups, 1):
            gid = f"{date}:G{index:06d}"
            for row in rows:
                global_ids[row["local_uid"]] = gid
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "association_events.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = list(candidates[0]) if candidates else ["date", "session", "left_uid", "right_uid",
            "match_type", "reid_distance", "reid_threshold", "bev_distance", "temporal_gap_seconds",
            "association_score", "accepted"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(candidates)
    with (out_dir / "id_mapping.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["local_uid", "global_id", "date", "camera_id", "recording_id", "local_id"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in persons:
            writer.writerow({key: (global_ids[row["local_uid"]] if key == "global_id" else row[key]) for key in fields})
    if tracks_path.exists():
        with tracks_path.open(encoding="utf-8", newline="") as source, (out_dir / "tracks.csv").open(
                "w", encoding="utf-8", newline="") as target:
            reader = csv.DictReader(source)
            fields = ["global_id", "local_uid", *(reader.fieldnames or [])]
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                uid = f"{row['date']}:{row['camera_id']}:{row['recording_id'].split('_')[-1]}:L{row['local_id']}"
                writer.writerow({"global_id": global_ids.get(uid, ""), "local_uid": uid, **row})
    attr_rows = attrs
    segments = read_rows(persons_path.parent / "watch_segments.csv")
    segments_by_uid = defaultdict(list)
    for segment in segments:
        segments_by_uid[segment["local_uid"]].append(segment)
    with (out_dir / "persons.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["global_id", "date", "first_seen", "last_seen", "exposure_seconds", "observed_seconds",
                  "predicted_gap_seconds", "watch_seconds", "attention_seconds", "camera_ids", "local_id_count",
                  "gender", "age", "representative_crop_path", "quality_flags"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for (_, root), rows in sorted(roots.items()):
            gid = global_ids[rows[0]["local_uid"]]
            best = max(rows, key=lambda x: (int(x.get("crop_watch_condition") or 0),
                                             int(x.get("crop_area") or 0),
                                             int(x.get("detection_count") or 0)))
            attr = attr_rows.get(best["local_uid"], {})
            first, last = min(dt(x["first_seen"]) for x in rows), max(dt(x["last_seen"]) for x in rows)
            exposure = union_duration([(dt(x["first_seen"]), dt(x["last_seen"])) for x in rows])
            watch_intervals, attention_intervals = [], []
            for row in rows:
                for segment in segments_by_uid[row["local_uid"]]:
                    start, end = dt(segment["start"]), dt(segment["end"])
                    watch_intervals.append((start + timedelta(seconds=float(cfg["watch"]["watch_threshold_seconds"])), end))
                    attention_intervals.append((start + timedelta(seconds=float(cfg["watch"]["attention_threshold_seconds"])), end))
            watch_available = any(row.get("watch_seconds") not in (None, "") for row in rows)
            writer.writerow({"global_id": gid, "date": rows[0]["date"], "first_seen": first.isoformat(sep=" "),
                "last_seen": last.isoformat(sep=" "), "exposure_seconds": exposure,
                "observed_seconds": sum(float(x["observed_seconds"]) for x in rows),
                "predicted_gap_seconds": sum(float(x["predicted_gap_seconds"]) for x in rows),
                "watch_seconds": (union_duration([(a, b) for a, b in watch_intervals if a < b])
                                  if watch_available else ""),
                "attention_seconds": (union_duration([(a, b) for a, b in attention_intervals if a < b])
                                      if watch_available else ""),
                "camera_ids": ";".join(sorted({x["camera_id"] for x in rows})), "local_id_count": len(rows),
                "gender": attr.get("gender", ""), "age": attr.get("age", ""),
                "representative_crop_path": best.get("crop_path", ""),
                "quality_flags": ";".join(filter(None, (x.get("quality_flags", "") for x in rows)))})
    with (out_dir / "topology.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["from_camera", "to_camera", "observed_transition_count", "status"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for (left, right), count in sorted(topology_counts.items()):
            writer.writerow({"from_camera": left, "to_camera": right, "observed_transition_count": count,
                             "status": "confirmed" if count >= int(gm["min_topology_observations"]) else "candidate"})
