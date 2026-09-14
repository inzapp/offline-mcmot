from __future__ import annotations

import csv
import json
import itertools
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import (bbox_xyxy, distance_to_polygon, gaze_ray,
                       homography_from_points, iou, is_watch, point_in_polygon,
                       segment_intersects_polygon, transform_point)


TRACK_FIELDS = [
    "site", "date", "recording_id", "camera_id", "timestamp", "frame_index",
    "detection_index", "local_id", "confidence", "cx", "cy", "w", "h",
    "bev_x", "bev_y", "inside_camera_roi", "inside_bev_roi", "is_observed",
    "is_predicted", "watch_condition",
    "gaze_valid", "gaze_camera_start_x", "gaze_camera_start_y",
    "gaze_camera_end_x", "gaze_camera_end_y", "gaze_bev_start_x",
    "gaze_bev_start_y", "gaze_bev_end_x", "gaze_bev_end_y", "gaze_target",
]


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")


class BBoxKalman:
    """Constant velocity Kalman filter for normalized cx, cy, w, h."""

    def __init__(self, measurement: np.ndarray, process_noise=1.0, measurement_noise=4.0):
        self.x = np.r_[measurement, np.zeros(4)].astype(float)
        self.P = np.eye(8) * 10.0
        self.q = float(process_noise)
        self.R = np.eye(4) * float(measurement_noise) * 1e-4
        self.H = np.c_[np.eye(4), np.zeros((4, 4))]

    def predict(self, dt: float) -> np.ndarray:
        dt = max(1e-3, float(dt))
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * dt
        G = np.vstack((np.eye(4) * (dt * dt / 2), np.eye(4) * dt))
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ (np.eye(4) * self.q * 1e-3) @ G.T
        self.x[2:4] = np.maximum(self.x[2:4], 1e-4)
        return self.x[:4].copy()

    def distance(self, measurement: np.ndarray) -> float:
        residual = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(residual.T @ np.linalg.pinv(S) @ residual)

    def update(self, measurement: np.ndarray) -> None:
        residual = measurement - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.x += K @ residual
        self.P = (np.eye(8) - K @ self.H) @ self.P


@dataclass
class Track:
    local_id: int
    kf: BBoxKalman
    first_time: datetime
    last_time: datetime
    last_observed: datetime
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    required_hits: int = 2
    observed_seconds: float = 0.0
    predicted_seconds: float = 0.0
    watch_total: float = 0.0
    attention_total: float = 0.0
    watch_start: datetime | None = None
    watch_last: datetime | None = None
    watch_segments: list[dict] = field(default_factory=list)
    detection_count: int = 0
    best_crop_score: tuple = field(default_factory=lambda: (-1, -1.0))
    best_bbox_size: tuple[float, float] | None = None
    best_crop: dict | None = None
    start_bev: tuple[float, float] | None = None
    end_bev: tuple[float, float] | None = None
    bev_history: list[tuple[datetime, float, float]] = field(default_factory=list)

    def apply_watch(self, when: datetime, watching: bool, cfg: dict) -> None:
        """Mirror tw_accumulate_watched_time: threshold time is not backfilled."""
        if watching:
            if self.watch_start is None:
                self.watch_start = self.watch_last = when
                return
            delta = max(0.0, (when - self.watch_last).total_seconds())
            current = (when - self.watch_start).total_seconds()
            self.watch_last = when
            if current >= float(cfg["watch_threshold_seconds"]):
                self.watch_total += delta
            if current >= float(cfg["attention_threshold_seconds"]):
                self.attention_total += delta
        else:
            self.finish_watch()

    def finish_watch(self) -> None:
        if self.watch_start is not None and self.watch_last is not None:
            duration = (self.watch_last - self.watch_start).total_seconds()
            if duration > 0:
                self.watch_segments.append({"start": self.watch_start, "end": self.watch_last,
                                            "raw_duration": duration})
        self.watch_start = self.watch_last = None


def load_roi(path: Path) -> tuple[list, list]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    camera, bev = [], []
    for roi in data.get("rois", []):
        camera.append(roi.get("image2_vertices_normalized", []))
        bev.append(roi.get("image1_vertices_normalized", []))
    return camera, bev


def load_camera_to_bev(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    sources, targets = [], []
    for roi in data.get("rois", []):
        sources.extend(roi.get("image2_vertices_normalized", []))
        targets.extend(roi.get("image1_vertices_normalized", []))
    if len(sources) < 4 or len(sources) != len(targets):
        raise ValueError(f"insufficient camera/BEV calibration points: {path}")
    return homography_from_points(sources, targets)


def add_gaze_fields(row: dict, camera_to_bev: np.ndarray, cfg: dict) -> None:
    ray = gaze_ray(row, float(cfg.get("keypoint_confidence", .3)))
    row.update({"gaze_valid": False, "gaze_target": ""})
    if ray is None:
        return
    camera_start, camera_end = ray
    try:
        bev_start = transform_point(camera_start, camera_to_bev)
        bev_end = transform_point(camera_end, camera_to_bev)
    except ValueError:
        return
    row.update({"gaze_valid": True,
        "gaze_camera_start_x": camera_start[0], "gaze_camera_start_y": camera_start[1],
        "gaze_camera_end_x": camera_end[0], "gaze_camera_end_y": camera_end[1],
        "gaze_bev_start_x": bev_start[0], "gaze_bev_start_y": bev_start[1],
        "gaze_bev_end_x": bev_end[0], "gaze_bev_end_y": bev_end[1]})
    person = (float(row["bev_x"]), float(row["bev_y"]))
    for target in cfg.get("targets", []):
        if (point_in_polygon(person, target["source_polygon"]) and
                segment_intersects_polygon(bev_start, bev_end, target["target_polygon"])):
            row["gaze_target"] = target["name"]
            break


def paired_frames(raw_path: Path, bev_path: Path) -> Iterator[tuple[datetime, int, list[dict]]]:
    """Stream raw/BEV rows in lockstep; duplicate frame rows retain row-order identity."""
    with raw_path.open(encoding="utf-8", newline="") as rs, bev_path.open(encoding="utf-8", newline="") as bs:
        raw_reader, bev_reader = csv.DictReader(rs), csv.DictReader(bs)
        current_key, rows = None, []
        raw_count = bev_count = 0
        for raw, bev in itertools.zip_longest(raw_reader, bev_reader):
            if raw is None or bev is None:
                raise ValueError(f"raw/BEV row count mismatch: {raw_path.name}")
            raw_count += 1
            bev_count += 1
            if raw["frame_index"] != bev["frame_index"] or raw["timestamp"] != bev["timestamp"]:
                raise ValueError(f"raw/BEV alignment mismatch at row {raw_count}: {raw_path.name}")
            raw.update({"bev_x": float(bev["x"]), "bev_y": float(bev["y"])})
            key = (raw["timestamp"], int(raw["frame_index"]))
            if current_key is not None and key != current_key:
                yield parse_time(current_key[0]), current_key[1], rows
                rows = []
            raw["detection_index"] = len(rows)
            rows.append(raw)
            current_key = key
        if current_key is not None:
            yield parse_time(current_key[0]), current_key[1], rows


def bev_frames(bev_path: Path) -> Iterator[tuple[datetime, int, list[dict]]]:
    with bev_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        current_key, rows = None, []
        for row in reader:
            key = (row["timestamp"], int(row["frame_index"]))
            if current_key is not None and key != current_key:
                yield parse_time(current_key[0]), current_key[1], rows
                rows = []
            row.update({"bev_x": float(row["x"]), "bev_y": float(row["y"]),
                        "detection_index": len(rows)})
            rows.append(row)
            current_key = key
        if current_key is not None:
            yield parse_time(current_key[0]), current_key[1], rows


@dataclass
class PointTrack:
    local_id: int
    position: np.ndarray
    velocity: np.ndarray
    first_time: datetime
    last_time: datetime
    last_observed: datetime
    hits: int = 1
    required_hits: int = 2
    predicted_seconds: float = 0.0
    history: list = field(default_factory=list)


def process_bev_only(record, cfg: dict, track_writer, person_writer, issues_writer) -> None:
    """Use otherwise stranded BEV data without inventing bbox/pose/attributes."""
    _, bev_rois = load_roi(Path(cfg["data_root"]) / cfg["site"] / record.camera_id /
                           f"{record.camera_id}_roi.json")
    tc = cfg["tracking"]
    active: list[PointTrack] = []
    completed: list[PointTrack] = []
    next_id, previous_time = 1, None
    max_gap = float(tc["max_missed_seconds"])
    tentative_gap = float(tc.get("tentative_max_missed_seconds", .5))
    max_speed = float(cfg["global_matching"]["max_bev_speed_per_second"])
    for when, frame_index, detections in bev_frames(Path(record.bev_path)):
        delta = .1 if previous_time is None else max(.001, (when - previous_time).total_seconds())
        previous_time = when
        survivors = []
        for track in active:
            allowed_gap = max_gap if track.hits >= track.required_hits else tentative_gap
            if (when - track.last_observed).total_seconds() > allowed_gap:
                completed.append(track)
            else:
                survivors.append(track)
        active = survivors
        valid = [row for row in detections if any(point_in_polygon((row["bev_x"], row["bev_y"]), roi)
                                                   for roi in bev_rois)]
        costs = np.full((len(active), len(valid)), 1e6)
        for ti, track in enumerate(active):
            predicted = track.position + track.velocity * delta
            for di, row in enumerate(valid):
                distance = float(np.linalg.norm(predicted - [row["bev_x"], row["bev_y"]]))
                if distance <= max(.02, max_speed * delta * 2):
                    costs[ti, di] = distance
        matches = []
        if costs.size:
            rr, cc = linear_sum_assignment(costs)
            matches = [(int(a), int(b)) for a, b in zip(rr, cc) if costs[a, b] < 1e5]
        used_t, used_d = {a for a, _ in matches}, {b for _, b in matches}
        for ti, di in matches:
            track, row = active[ti], valid[di]
            new_position = np.array([row["bev_x"], row["bev_y"]], dtype=float)
            obs_delta = max(.001, (when - track.last_observed).total_seconds())
            measured_velocity = (new_position - track.position) / obs_delta
            track.velocity = .7 * track.velocity + .3 * measured_velocity
            track.position = new_position
            track.last_time = track.last_observed = when
            track.hits += 1
            track.history.append((when, *new_position))
            track_writer.writerow({"site": record.site, "date": record.date, "recording_id": record.stem,
                "camera_id": record.camera_id, "timestamp": when.isoformat(sep=" "), "frame_index": frame_index,
                "detection_index": row["detection_index"], "local_id": track.local_id,
                "bev_x": row["bev_x"], "bev_y": row["bev_y"], "inside_bev_roi": 1,
                "is_observed": 1, "is_predicted": 0, "watch_condition": ""})
        for ti, track in enumerate(active):
            if ti not in used_t:
                track.position += track.velocity * delta
                track.last_time = when
                track.predicted_seconds += delta
                if track.hits >= track.required_hits:
                    track_writer.writerow({"site": record.site, "date": record.date, "recording_id": record.stem,
                        "camera_id": record.camera_id, "timestamp": when.isoformat(sep=" "), "frame_index": frame_index,
                        "local_id": track.local_id, "bev_x": "", "bev_y": "", "inside_bev_roi": "",
                        "is_observed": 0, "is_predicted": 1, "watch_condition": ""})
        for di, row in enumerate(valid):
            if di in used_d: continue
            position = np.array([row["bev_x"], row["bev_y"]], dtype=float)
            boundary_distance = min((distance_to_polygon(position, roi) for roi in bev_rois), default=float("inf"))
            required_hits = int(tc["min_confirmed_hits"] if boundary_distance <= float(tc.get("entry_bev_boundary_distance", .02))
                                else tc.get("center_min_confirmed_hits", tc["min_confirmed_hits"]))
            track = PointTrack(next_id, position, np.zeros(2), when, when, when,
                               required_hits=required_hits, history=[(when, *position)])
            next_id += 1; active.append(track)
            track_writer.writerow({"site": record.site, "date": record.date, "recording_id": record.stem,
                "camera_id": record.camera_id, "timestamp": when.isoformat(sep=" "), "frame_index": frame_index,
                "detection_index": row["detection_index"], "local_id": track.local_id,
                "bev_x": row["bev_x"], "bev_y": row["bev_y"], "inside_bev_roi": 1,
                "is_observed": 1, "is_predicted": 0, "watch_condition": ""})
    completed.extend(active)
    for track in completed:
        if track.hits < track.required_hits: continue
        uid = f"{record.date}:{record.camera_id}:{record.nominal_start}:L{track.local_id}"
        first, last = track.history[0], track.history[-1]
        person_writer.writerow({"local_uid": uid, "date": record.date, "recording_id": record.stem,
            "camera_id": record.camera_id, "local_id": track.local_id,
            "first_seen": track.first_time.isoformat(sep=" "), "last_seen": track.last_time.isoformat(sep=" "),
            "last_observed": track.last_observed.isoformat(sep=" "),
            "track_span_seconds": max(0., (track.last_time-track.first_time).total_seconds()),
            "observed_seconds": max(0., (track.last_observed-track.first_time).total_seconds()),
            "predicted_gap_seconds": track.predicted_seconds, "watch_seconds": "", "attention_seconds": "",
            "detection_count": track.hits,
            "start_boundary_distance": min((distance_to_polygon((first[1],first[2]), roi) for roi in bev_rois), default=""),
            "end_boundary_distance": min((distance_to_polygon((last[1],last[2]), roi) for roi in bev_rois), default=""),
            "start_bev_x": first[1], "start_bev_y": first[2], "end_bev_x": last[1], "end_bev_y": last[2],
            "end_velocity_x": track.velocity[0], "end_velocity_y": track.velocity[1],
            "quality_flags": "bev_only;raw_bbox_pose_unavailable;attributes_unavailable"})
    issues_writer.writerow({"scope": record.stem, "severity": "warning", "code": "bev_only_tracking",
                            "detail": "raw CSV unavailable; watch/attention/crop/attributes remain null"})


def predicted_bev(track: Track, when: datetime) -> np.ndarray | None:
    if not track.bev_history:
        return None
    _, x, y = track.bev_history[-1]
    velocity = np.zeros(2)
    if len(track.bev_history) >= 2:
        previous, current = track.bev_history[-2], track.bev_history[-1]
        seconds = (current[0] - previous[0]).total_seconds()
        if seconds > 0:
            velocity = (np.array(current[1:]) - np.array(previous[1:])) / seconds
    gap = max(0.0, (when - track.last_observed).total_seconds())
    return np.array([x, y]) + velocity * gap


def assignment(tracks: list[Track], detections: list[dict], iou_gate: float,
               mahal_gate: float, when: datetime | None = None,
               bev_recovery_gate: float = 0.0,
               bev_recovery_growth: float = 0.0) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    costs = np.full((len(tracks), len(detections)), 1e6, dtype=float)
    for ti, track in enumerate(tracks):
        predicted = bbox_xyxy(*track.kf.x[:4])
        for di, det in enumerate(detections):
            measurement = np.array([float(det[k]) for k in ("cx", "cy", "w", "h")])
            overlap = iou(predicted, bbox_xyxy(*measurement))
            mahal = track.kf.distance(measurement)
            if overlap >= iou_gate or mahal <= mahal_gate:
                costs[ti, di] = (1.0 - overlap) + min(mahal / mahal_gate, 1.0) * 0.25
                continue
            # A weak detector can return a substantially changed bbox after an
            # occlusion. Recover only if the independently mapped BEV position
            # remains physically close to the track prediction.
            if when is not None and bev_recovery_gate > 0 and track.confirmed:
                expected = predicted_bev(track, when)
                if expected is not None and det.get("bev_x") not in (None, ""):
                    distance = float(np.linalg.norm(expected - [float(det["bev_x"]), float(det["bev_y"])]))
                    gap = max(0.0, (when - track.last_observed).total_seconds())
                    gate = bev_recovery_gate + bev_recovery_growth * gap
                    if distance <= gate:
                        costs[ti, di] = 1.25 + distance / max(gate, 1e-9) * 0.5
    rows, cols = linear_sum_assignment(costs)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols) if costs[r, c] < 1e5]
    used_t, used_d = {x for x, _ in matches}, {x for _, x in matches}
    return matches, [x for x in range(len(tracks)) if x not in used_t], [x for x in range(len(detections)) if x not in used_d]


class VideoCropReader:
    def __init__(self, video_path: str):
        self.path = video_path
        self.cap = cv2.VideoCapture(video_path) if video_path else None
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 0.0

    def crop(self, frame_index: int, row: dict) -> np.ndarray | None:
        if not self.cap or not self.cap.isOpened():
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, image = self.cap.read()
        if not ok:
            return None
        height, width = image.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy(float(row["cx"]), float(row["cy"]),
                                    float(row["w"]), float(row["h"]))
        x1, x2 = np.clip(np.array([x1, x2]) * width, 0, width).astype(int)
        y1, y2 = np.clip(np.array([y1, y2]) * height, 0, height).astype(int)
        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2]

    def close(self):
        if self.cap:
            self.cap.release()


def process_recording(record, cfg: dict, track_writer, person_writer, segment_writer,
                      crop_dir: Path, issues_writer) -> None:
    if not record.raw_path and record.bev_path:
        process_bev_only(record, cfg, track_writer, person_writer, issues_writer)
        return
    if not record.raw_path or not record.bev_path:
        issues_writer.writerow({"scope": record.stem, "severity": "warning",
                                "code": "tracking_input_missing", "detail": record.status})
        return
    roi_path = Path(cfg["data_root"]) / cfg["site"] / record.camera_id / f"{record.camera_id}_roi.json"
    camera_rois, bev_rois = load_roi(roi_path)
    camera_to_bev = load_camera_to_bev(roi_path)
    tc, wc, cc = cfg["tracking"], cfg["watch"], cfg["crop"]
    active: list[Track] = []
    completed: list[Track] = []
    next_id = 1
    previous_time = None
    reader = VideoCropReader(record.video_path)

    def max_track_gap(track: Track) -> float:
        return float(tc["max_missed_seconds"] if track.confirmed else tc.get("tentative_max_missed_seconds", .5))

    def finish(track: Track):
        track.finish_watch()
        completed.append(track)

    def write_detection(track: Track, row: dict, when: datetime, frame: int,
                        observed=True, predicted=False):
        output = {key: "" for key in TRACK_FIELDS}
        output.update({"site": record.site, "date": record.date, "recording_id": record.stem,
                       "camera_id": record.camera_id, "timestamp": when.isoformat(sep=" "),
                       "frame_index": frame, "detection_index": row.get("detection_index", ""),
                       "local_id": track.local_id, "confidence": row.get("confidence", ""),
                       "cx": row.get("cx", track.kf.x[0]), "cy": row.get("cy", track.kf.x[1]),
                       "w": row.get("w", track.kf.x[2]), "h": row.get("h", track.kf.x[3]),
                       "bev_x": row.get("bev_x", ""), "bev_y": row.get("bev_y", ""),
                       "inside_camera_roi": row.get("inside_camera_roi", ""),
                       "inside_bev_roi": row.get("inside_bev_roi", ""),
                       "is_observed": int(observed), "is_predicted": int(predicted),
                       "watch_condition": row.get("watch_condition", "")})
        for key in TRACK_FIELDS:
            if key.startswith("gaze_"):
                output[key] = row.get(key, "")
        track_writer.writerow(output)

    for when, frame_index, rows in paired_frames(Path(record.raw_path), Path(record.bev_path)):
        dt = 0.1 if previous_time is None else max(0.001, (when - previous_time).total_seconds())
        previous_time = when
        still_active = []
        for track in active:
            if (when - track.last_observed).total_seconds() > max_track_gap(track):
                finish(track)
            else:
                still_active.append(track)
        active = still_active
        for track in active:
            track.kf.predict(dt)
        valid = []
        for row in rows:
            point = (float(row["cx"]), float(row["cy"]) + float(row["h"]) / 2)
            row["inside_camera_roi"] = any(point_in_polygon(point, roi) for roi in camera_rois)
            row["inside_bev_roi"] = any(point_in_polygon((row["bev_x"], row["bev_y"]), roi) for roi in bev_rois)
            if row["inside_camera_roi"] and row["inside_bev_roi"] and float(row["confidence"]) >= float(tc["low_confidence"]):
                row["watch_condition"] = is_watch(row, wc)
                add_gaze_fields(row, camera_to_bev, cfg.get("spatial", {}).get("gaze", {}))
                valid.append(row)
        high = [r for r in valid if float(r["confidence"]) >= float(tc["high_confidence"])]
        low = [r for r in valid if float(r["confidence"]) < float(tc["high_confidence"])]
        matches, unmatched_t, unmatched_d = assignment(
            active, high, float(tc["iou_gate"]), float(tc["mahalanobis_gate"]), when,
            float(tc.get("bev_recovery_gate", 0)), float(tc.get("bev_recovery_growth_per_second", 0)))
        second_tracks = [active[i] for i in unmatched_t]
        matches2, unmatched_t2, _ = assignment(
            second_tracks, low, float(tc["second_pass_iou_gate"]), float(tc["mahalanobis_gate"]), when,
            float(tc.get("bev_recovery_gate", 0)), float(tc.get("bev_recovery_growth_per_second", 0)))
        combined = [(ti, high[di]) for ti, di in matches]
        combined += [(unmatched_t[ti], low[di]) for ti, di in matches2]
        matched_indices = set()
        for ti, row in combined:
            matched_indices.add(ti)
            track = active[ti]
            previous_observed = track.last_observed
            measurement = np.array([float(row[k]) for k in ("cx", "cy", "w", "h")])
            track.kf.update(measurement)
            delta = max(0.0, (when - previous_observed).total_seconds())
            track.observed_seconds += min(delta, float(tc["max_missed_seconds"]))
            track.last_time = track.last_observed = when
            track.hits += 1
            track.misses = 0
            track.confirmed = track.confirmed or track.hits >= track.required_hits
            track.detection_count += 1
            watching = bool(row["watch_condition"])
            track.apply_watch(when, watching, wc)
            bev = (float(row["bev_x"]), float(row["bev_y"]))
            track.start_bev = track.start_bev or bev
            track.end_bev = bev
            track.bev_history.append((when, *bev))
            box_w, box_h = float(row["w"]), float(row["h"])
            area = box_w * box_h
            crop_score = (int(watching), area)
            growth = float(cc["required_growth_ratio"])
            bigger = (track.best_bbox_size is None or
                      (box_w > track.best_bbox_size[0] * growth and box_h > track.best_bbox_size[1] * growth))
            pose_ok = not (track.best_crop and track.best_crop["watching"] and not watching)
            should_crop = (track.detection_count % int(cc["update_every_detections"]) == 0
                           and bigger and pose_ok)
            if should_crop:
                crop = reader.crop(frame_index, row)
                if crop is not None and crop.shape[1] >= int(cc["min_width_px"]) and crop.shape[0] >= int(cc["min_height_px"]):
                    path = crop_dir / f"{record.date}_{record.camera_id}_{record.nominal_start}_L{track.local_id}.jpg"
                    cv2.imwrite(str(path), crop)
                    track.best_crop_score = crop_score
                    track.best_bbox_size = (box_w, box_h)
                    track.best_crop = {"path": str(path.resolve()), "frame_index": frame_index,
                                       "timestamp": when.isoformat(sep=" "), "watching": int(watching),
                                       "width": crop.shape[1], "height": crop.shape[0]}
            write_detection(track, row, when, frame_index)

        survivors = []
        for ti, track in enumerate(active):
            if ti in matched_indices:
                survivors.append(track)
                continue
            track.misses += 1
            gap = (when - track.last_observed).total_seconds()
            track.apply_watch(when, False, wc)
            if gap <= max_track_gap(track):
                track.predicted_seconds += dt
                track.last_time = when
                if track.confirmed:
                    write_detection(track, {}, when, frame_index, observed=False, predicted=True)
                survivors.append(track)
            else:
                finish(track)
        active = survivors
        for di in unmatched_d:
            row = high[di]
            measurement = np.array([float(row[k]) for k in ("cx", "cy", "w", "h")])
            camera_point = (float(row["cx"]), float(row["cy"]) + float(row["h"]) / 2)
            bev_point = (float(row["bev_x"]), float(row["bev_y"]))
            camera_boundary = min((distance_to_polygon(camera_point, roi) for roi in camera_rois), default=float("inf"))
            bev_boundary = min((distance_to_polygon(bev_point, roi) for roi in bev_rois), default=float("inf"))
            enters_at_boundary = (camera_boundary <= float(tc.get("entry_camera_boundary_distance", .04)) or
                                  bev_boundary <= float(tc.get("entry_bev_boundary_distance", .02)))
            required_hits = int(tc["min_confirmed_hits"] if enters_at_boundary
                                else tc.get("center_min_confirmed_hits", tc["min_confirmed_hits"]))
            track = Track(next_id, BBoxKalman(measurement, tc["process_noise"], tc["measurement_noise"]),
                          when, when, when, required_hits=required_hits)
            next_id += 1
            track.detection_count = 1
            track.apply_watch(when, bool(row["watch_condition"]), wc)
            bev = (float(row["bev_x"]), float(row["bev_y"]))
            track.start_bev = track.end_bev = bev
            track.bev_history.append((when, *bev))
            active.append(track)
            write_detection(track, row, when, frame_index)
    reader.close()
    for track in active:
        finish(track)
    for track in completed:
        if not track.confirmed:
            continue
        uid = f"{record.date}:{record.camera_id}:{record.nominal_start}:L{track.local_id}"
        span = max(0.0, (track.last_time - track.first_time).total_seconds())
        history = track.bev_history
        vx = vy = 0.0
        if len(history) >= 2:
            delta = (history[-1][0] - history[-2][0]).total_seconds()
            if delta > 0:
                vx = (history[-1][1] - history[-2][1]) / delta
                vy = (history[-1][2] - history[-2][2]) / delta
        person_writer.writerow({"local_uid": uid, "date": record.date, "recording_id": record.stem,
            "camera_id": record.camera_id, "local_id": track.local_id,
            "first_seen": track.first_time.isoformat(sep=" "), "last_seen": track.last_time.isoformat(sep=" "),
            "last_observed": track.last_observed.isoformat(sep=" "), "track_span_seconds": span,
            "observed_seconds": track.observed_seconds, "predicted_gap_seconds": track.predicted_seconds,
            "watch_seconds": track.watch_total, "attention_seconds": track.attention_total,
            "detection_count": track.hits,
            "start_boundary_distance": min((distance_to_polygon(track.start_bev, roi) for roi in bev_rois), default=""),
            "end_boundary_distance": min((distance_to_polygon(track.end_bev, roi) for roi in bev_rois), default=""),
            "start_bev_x": track.start_bev[0] if track.start_bev else "",
            "start_bev_y": track.start_bev[1] if track.start_bev else "",
            "end_bev_x": track.end_bev[0] if track.end_bev else "", "end_bev_y": track.end_bev[1] if track.end_bev else "",
            "end_velocity_x": vx, "end_velocity_y": vy,
            "crop_path": track.best_crop["path"] if track.best_crop else "",
            "crop_frame_index": track.best_crop["frame_index"] if track.best_crop else "",
            "crop_watch_condition": track.best_crop["watching"] if track.best_crop else "",
            "crop_width": track.best_crop["width"] if track.best_crop else "",
            "crop_height": track.best_crop["height"] if track.best_crop else "",
            "crop_area": (track.best_crop["width"] * track.best_crop["height"]) if track.best_crop else "",
            "quality_flags": "" if track.best_crop else "no_crop"})
        for index, segment in enumerate(track.watch_segments):
            segment_writer.writerow({"local_uid": uid, "segment_index": index,
                "start": segment["start"].isoformat(sep=" "), "end": segment["end"].isoformat(sep=" "),
                "raw_duration_seconds": segment["raw_duration"],
                "credited_watch_seconds": max(0.0, segment["raw_duration"] - float(wc["watch_threshold_seconds"])),
                "credited_attention_seconds": max(0.0, segment["raw_duration"] - float(wc["attention_threshold_seconds"]))})
