from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .geometry import point_in_polygon


def _read(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def _hour(value: str) -> str:
    return datetime.fromisoformat(value).strftime("%H:00")


def build_spatial(output_root: Path, cfg: dict) -> Path | None:
    """Create route/region/OOI membership and gender-age hourly aggregates."""
    spatial_cfg = cfg.get("spatial") or {}
    tracks_path = output_root / "global/tracks.csv"
    people_path = output_root / "global/persons.csv"
    if not spatial_cfg or not tracks_path.exists() or not people_path.exists():
        return None
    people = {row["global_id"]: row for row in _read(people_path)}
    routes, regions = spatial_cfg.get("routes", []), spatial_cfg.get("regions", [])
    # Keep only compact per-person state; production track files contain millions of rows.
    states = defaultdict(lambda: {"route": [[None, None, None, None] for _ in routes],
                                  "region": [[None, 0] for _ in regions],
                                  "gaze": defaultdict(lambda: [None, 0])})
    for row in _read(tracks_path):
        if (not row.get("global_id") or row.get("is_observed") != "1" or
                row.get("bev_x") in (None, "") or row.get("bev_y") in (None, "")):
            continue
        gid, when = row["global_id"], row["timestamp"]
        point = (float(row["bev_x"]), float(row["bev_y"]))
        state = states[gid]
        for index, route in enumerate(routes):
            item = state["route"][index]
            in_start = point_in_polygon(point, route["start_gate"])
            in_end = point_in_polygon(point, route["end_gate"])
            if in_start:
                item[0] = min(item[0], when) if item[0] else when
                item[1] = max(item[1], when) if item[1] else when
            if in_end:
                item[2] = min(item[2], when) if item[2] else when
                item[3] = max(item[3], when) if item[3] else when
        for index, region in enumerate(regions):
            if point_in_polygon(point, region["polygon"]):
                item = state["region"][index]
                item[0] = item[0] or when
                item[1] += 1
        if row.get("gaze_target"):
            item = state["gaze"][row["gaze_target"]]
            item[0] = item[0] or when
            item[1] += 1
    events = []
    for gid, state in states.items():
        person = people.get(gid, {})
        common = {"global_id": gid, "date": person.get("date", ""),
                  "gender": person.get("gender", ""), "age": person.get("age", "")}
        for route, item in zip(routes, state["route"]):
            forward = item[0] is not None and item[3] is not None and item[0] <= item[3]
            reverse = (route.get("bidirectional") and item[2] is not None and item[1] is not None
                       and item[2] <= item[1])
            if forward or reverse:
                first_seen = min(value for value in (item[0] if forward else None,
                                                      item[2] if reverse else None) if value is not None)
                events.append({**common, "dimension": "route", "name": route["name"],
                               "first_seen": first_seen, "seconds": ""})
        for region, item in zip(regions, state["region"]):
            if item[0] is not None:
                events.append({**common, "dimension": "region", "name": region["name"],
                               "first_seen": item[0], "seconds": f"{item[1] / 10:.1f}"})
        for name, item in state["gaze"].items():
            events.append({**common, "dimension": "gaze", "name": name,
                           "first_seen": item[0], "seconds": f"{item[1] / 10:.1f}"})

    target = output_root / "spatial"
    target.mkdir(parents=True, exist_ok=True)
    fields = ["global_id", "date", "first_seen", "dimension", "name", "seconds", "gender", "age"]
    with (target / "person_events.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(events)
    counts = defaultdict(set)
    for event in events:
        keys = [
            (event["date"], "all", event["dimension"], event["name"], event["gender"], event["age"]),
            (event["date"], _hour(event["first_seen"]), event["dimension"], event["name"], event["gender"], event["age"]),
        ]
        for key in keys:
            counts[key].add(event["global_id"])
    summary_fields = ["date", "hour", "dimension", "name", "gender", "age", "person_count"]
    with (target / "demographics_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields); writer.writeheader()
        for key, ids in sorted(counts.items()):
            writer.writerow(dict(zip(summary_fields[:-1], key), person_count=len(ids)))
    return target / "person_events.csv"


def spatial_map_asset(output_root: Path, cfg: dict) -> tuple[str, list[dict]]:
    image = cv2.imread(str(cfg["bev_image"]))
    spatial_cfg = cfg.get("spatial") or {}
    if image is None or not spatial_cfg:
        return "", []
    height, width = image.shape[:2]
    legend = []
    colors = [(20, 50, 240), (30, 170, 255), (180, 80, 220), (50, 180, 80)]
    for index, route in enumerate(spatial_cfg.get("routes", []), 1):
        color = colors[(index - 1) % len(colors)]
        polygons = []
        for gate in (route["start_gate"], route["end_gate"]):
            points = np.asarray([(int(x * width), int(y * height)) for x, y in gate], np.int32)
            polygons.append(points)
            cv2.polylines(image, [points], True, color, 3, cv2.LINE_AA)
        centers = [tuple(np.mean(points, axis=0).astype(int)) for points in polygons]
        cv2.arrowedLine(image, centers[0], centers[1], color, 4, cv2.LINE_AA, tipLength=.12)
        if route.get("bidirectional"):
            cv2.arrowedLine(image, centers[1], centers[0], color, 4, cv2.LINE_AA, tipLength=.12)
        cv2.circle(image, centers[0], 15, color, -1, cv2.LINE_AA)
        cv2.putText(image, str(index), (centers[0][0] - 6, centers[0][1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
        legend.append({"번호": index, "구분": "동선", "이름": route["name"],
                       "판정": "양방향 gate 통과" if route.get("bidirectional") else "start→end gate 통과"})
    base = len(legend)
    for offset, target in enumerate(spatial_cfg.get("gaze", {}).get("targets", []), 1):
        index, color = base + offset, (190, 60, 190)
        source = np.asarray([(int(x * width), int(y * height)) for x, y in target["source_polygon"]], np.int32)
        goal = np.asarray([(int(x * width), int(y * height)) for x, y in target["target_polygon"]], np.int32)
        cv2.polylines(image, [source], True, color, 3, cv2.LINE_AA)
        cv2.polylines(image, [goal], True, color, 3, cv2.LINE_AA)
        a, b = tuple(np.mean(source, axis=0).astype(int)), tuple(np.mean(goal, axis=0).astype(int))
        cv2.arrowedLine(image, a, b, color, 4, cv2.LINE_AA, tipLength=.18)
        cv2.circle(image, a, 15, color, -1, cv2.LINE_AA)
        cv2.putText(image, str(index), (a[0] - 6, a[1] + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (255, 255, 255), 2, cv2.LINE_AA)
        legend.append({"번호": index, "구분": "시선", "이름": target["name"],
                       "판정": "출발 구역 내 OOI ray가 대상 polygon과 교차"})
    assets = output_root / "report/assets"
    assets.mkdir(parents=True, exist_ok=True)
    path = assets / "spatial_definition.jpg"
    cv2.imwrite(str(path), image)
    return "assets/spatial_definition.jpg", legend
