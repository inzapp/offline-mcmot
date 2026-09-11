from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


COLORS = [(38, 170, 255), (80, 210, 120), (230, 120, 70), (190, 90, 220)]


def _rows(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def _save(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 91] if path.suffix == ".jpg" else [])
    return f"assets/{path.name}"


def _label(image: np.ndarray, text: str, at: tuple[int, int], scale=.65) -> None:
    cv2.putText(image, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 20, 28), 4, cv2.LINE_AA)
    cv2.putText(image, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 250, 252), 1, cv2.LINE_AA)


def _roi_specs(cfg: dict) -> dict[str, dict]:
    root = Path(cfg["data_root"]) / cfg["site"]
    specs = {}
    for path in sorted(root.glob("*/*_roi.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("rois"):
            specs[path.parent.name] = payload["rois"][0]
    return specs


def _camera_roi_images(cfg: dict, assets: Path, specs: dict[str, dict]) -> list[dict]:
    root = Path(cfg["data_root"]) / cfg["site"]
    result = []
    for index, (camera, spec) in enumerate(sorted(specs.items())):
        source = cv2.imread(str(root / camera / f"{camera}.png"))
        if source is None:
            continue
        overlay = source.copy()
        points = np.asarray(spec["image2_vertices_px"], np.int32)
        color = COLORS[index % len(COLORS)]
        cv2.fillPoly(overlay, [points], color)
        image = cv2.addWeighted(overlay, .24, source, .76, 0)
        cv2.polylines(image, [points], True, color, 5, cv2.LINE_AA)
        # The outer band is where track starts/ends are considered for handoff topology.
        cv2.polylines(image, [points], True, (255, 255, 255), 12, cv2.LINE_AA)
        cv2.polylines(image, [points], True, color, 5, cv2.LINE_AA)
        for number, point in enumerate(points):
            cv2.circle(image, tuple(point), 7, (255, 255, 255), -1)
            _label(image, str(number + 1), (int(point[0]) + 8, int(point[1]) - 8), .48)
        _label(image, f"CAM {camera} / tracking ROI", (24, 42), .9)
        result.append({"camera": camera, "src": _save(assets / f"camera_{camera}_roi.jpg", image),
                       "vertices": len(points)})
    return result


def _bev_overview(cfg: dict, assets: Path, specs: dict[str, dict], topology: list[dict]) -> tuple[str, str]:
    bev = cv2.imread(str(cfg["bev_image"]))
    if bev is None:
        return "", ""
    coverage = bev.copy()
    centers = {}
    for index, (camera, spec) in enumerate(sorted(specs.items())):
        points = np.asarray(spec["image1_vertices_px"], np.int32)
        color = COLORS[index % len(COLORS)]
        layer = coverage.copy()
        cv2.fillPoly(layer, [points], color)
        coverage = cv2.addWeighted(layer, .20, coverage, .80, 0)
        cv2.polylines(coverage, [points], True, color, 3, cv2.LINE_AA)
        clipped = np.clip(points, [0, 0], [bev.shape[1] - 1, bev.shape[0] - 1])
        center = tuple(np.mean(clipped, axis=0).astype(int))
        centers[camera] = center
        _label(coverage, f"CAM {camera}", center, .62)
    coverage_src = _save(assets / "bev_camera_coverage.jpg", coverage)

    graph = coverage.copy()
    for row in topology:
        left, right = centers.get(row.get("from_camera")), centers.get(row.get("to_camera"))
        if left is None or right is None:
            continue
        cv2.arrowedLine(graph, left, right, (20, 35, 235), 4, cv2.LINE_AA, tipLength=.12)
        mid = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
        _label(graph, f"{row.get('observed_transition_count', '0')} transitions", mid, .5)
    return coverage_src, _save(assets / "bev_topology.jpg", graph)


def _track_visuals(output_root: Path, cfg: dict, assets: Path, mappings: list[dict]) -> tuple[str, list[dict], dict]:
    bev = cv2.imread(str(cfg["bev_image"]))
    if bev is None:
        return "", [], {}
    local_to_global = {row["local_uid"]: row["global_id"] for row in mappings}
    members = defaultdict(list)
    for uid, gid in local_to_global.items():
        members[gid].append(uid)
    selected_gids = [gid for gid, uids in sorted(members.items(), key=lambda item: -len(item[1])) if len(uids) > 1][:6]
    wanted = {uid for gid in selected_gids for uid in members[gid]}
    heat = np.zeros(bev.shape[:2], np.float32)
    paths = defaultdict(list)
    cameras = defaultdict(set)
    endpoints = {}
    sample_index = 0
    for row in _rows(output_root / "local/tracks.csv") or []:
        if row.get("is_observed") != "1":
            continue
        try:
            x, y = float(row["bev_x"]), float(row["bev_y"])
        except (ValueError, TypeError):
            continue
        px = min(bev.shape[1] - 1, max(0, int(x * bev.shape[1])))
        py = min(bev.shape[0] - 1, max(0, int(y * bev.shape[0])))
        sample_index += 1
        if sample_index % 3 == 0:
            heat[py, px] += 1
        uid = f"{row['date']}:{row['camera_id']}:{row['recording_id'].rsplit('_', 1)[-1]}:L{row['local_id']}"
        try:
            camera_point = (float(row["cx"]), float(row["cy"]))
        except (ValueError, TypeError):
            camera_point = None
        if camera_point is not None:
            if uid not in endpoints:
                endpoints[uid] = [camera_point, camera_point]
            else:
                endpoints[uid][1] = camera_point
        if uid in wanted and len(paths[uid]) < 400:
            paths[uid].append((px, py))
            cameras[local_to_global[uid]].add(row["camera_id"])
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=14)
    if heat.max() > 0:
        scaled = np.uint8(np.clip(heat / np.percentile(heat[heat > 0], 99) * 255, 0, 255))
        colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        alpha = np.minimum(.78, scaled.astype(np.float32) / 255 * .78)[..., None]
        heatmap = (bev * (1 - alpha) + colored * alpha).astype(np.uint8)
    else:
        heatmap = bev.copy()
    _label(heatmap, "Observed pedestrian density (actual BEV points)", (18, 34), .68)
    heat_src = _save(assets / "bev_heatmap.jpg", heatmap)

    gallery = []
    for index, gid in enumerate(selected_gids):
        image = bev.copy()
        all_points = []
        for uid in members[gid]:
            points = paths.get(uid, [])
            if len(points) < 2:
                continue
            all_points.extend(points)
            cv2.polylines(image, [np.asarray(points, np.int32)], False, COLORS[index % len(COLORS)], 4, cv2.LINE_AA)
            cv2.circle(image, points[0], 8, (70, 220, 100), -1)
            cv2.circle(image, points[-1], 8, (40, 80, 240), -1)
        if not all_points:
            continue
        _label(image, f"{gid} / {len(members[gid])} local IDs / {len(cameras[gid])} cameras", (18, 34), .65)
        gallery.append({"global_id": gid, "local_ids": len(members[gid]),
                        "cameras": ", ".join(sorted(cameras[gid])),
                        "src": _save(assets / f"trajectory_{index + 1}.jpg", image)})
    return heat_src, gallery, endpoints


def _boundary_images(cfg: dict, assets: Path, specs: dict[str, dict], persons: list[dict],
                     endpoints: dict) -> list[dict]:
    root = Path(cfg["data_root"]) / cfg["site"]
    quantile = float(cfg["global_matching"]["boundary_quantile"])
    result = []
    for index, (camera, spec) in enumerate(sorted(specs.items())):
        camera_people = [row for row in persons if row.get("camera_id") == camera]
        values = [float(row[key]) for row in camera_people
                  for key in ("start_boundary_distance", "end_boundary_distance")
                  if row.get(key) not in (None, "")]
        if not values:
            continue
        threshold = float(np.quantile(values, quantile))
        source = cv2.imread(str(root / camera / f"{camera}.png"))
        if source is None:
            continue
        height, width = source.shape[:2]
        points = np.asarray(spec["image2_vertices_px"], np.int32)
        image = source.copy()
        shade = source.copy()
        cv2.polylines(shade, [points], True, (0, 210, 255), 38, cv2.LINE_AA)
        image = cv2.addWeighted(shade, .27, image, .73, 0)
        cv2.polylines(image, [points], True, (0, 210, 255), 4, cv2.LINE_AA)
        starts, ends = [], []
        for row in camera_people:
            pair = endpoints.get(row["local_uid"])
            if not pair:
                continue
            if row.get("start_boundary_distance") and float(row["start_boundary_distance"]) <= threshold:
                starts.append((int(pair[0][0] * width), int(pair[0][1] * height)))
            if row.get("end_boundary_distance") and float(row["end_boundary_distance"]) <= threshold:
                ends.append((int(pair[1][0] * width), int(pair[1][1] * height)))
        density = np.zeros((height, width), np.float32)
        for x, y in starts + ends:
            if 0 <= x < width and 0 <= y < height:
                density[y, x] += 1
        if density.max() > 0:
            density = cv2.GaussianBlur(density, (0, 0), sigmaX=18)
            density = np.uint8(np.clip(density / max(np.percentile(density[density > 0], 99), 1e-9) * 255, 0, 255))
            color_heat = cv2.applyColorMap(density, cv2.COLORMAP_TURBO)
            alpha = np.minimum(.58, density.astype(np.float32) / 255 * .58)[..., None]
            image = (image * (1 - alpha) + color_heat * alpha).astype(np.uint8)
        # Plot a stable sample so dense regions remain readable; the heat layer contains all candidates.
        stride_start = max(1, len(starts) // 220)
        stride_end = max(1, len(ends) // 220)
        for point in starts[::stride_start]:
            cv2.circle(image, point, 6, (255, 190, 20), -1, cv2.LINE_AA)
        for point in ends[::stride_end]:
            cv2.drawMarker(image, point, (30, 40, 240), cv2.MARKER_TILTED_CROSS, 12, 3, cv2.LINE_AA)
        cv2.rectangle(image, (14, 12), (630, 105), (16, 28, 40), -1)
        _label(image, f"CAM {camera} / boundary-distance bottom {quantile * 100:.0f}%", (26, 43), .75)
        _label(image, f"ENTRY {len(starts):,}   EXIT {len(ends):,}   threshold {threshold:.4f} (BEV norm)", (26, 76), .58)
        cv2.circle(image, (31, 94), 5, (255, 190, 20), -1)
        cv2.drawMarker(image, (150, 94), (30, 40, 240), cv2.MARKER_TILTED_CROSS, 10, 2)
        _label(image, "entry", (42, 100), .43)
        _label(image, "disappearance", (162, 100), .43)
        result.append({"camera": camera, "threshold": f"{threshold:.4f}", "entries": len(starts),
                       "exits": len(ends),
                       "note": "원본 카메라 좌표 없음(BEV-only)" if not starts and not ends else "실제 camera track 좌표",
                       "src": _save(assets / f"camera_{camera}_boundary_15pct.jpg", image)})
    return result


def _reid_gallery(output_root: Path, assets: Path, associations: list[dict], persons: list[dict],
                  mappings: list[dict], excluded_global_ids: set[str], preferred_pairs: list[str]) -> list[dict]:
    person_by_uid = {row["local_uid"]: row for row in persons}
    gid_by_uid = {row["local_uid"]: row["global_id"] for row in mappings}
    accepted = [row for row in associations if row.get("accepted") == "1" and row.get("reid_distance")]
    accepted.sort(key=lambda row: float(row["reid_distance"]))
    event_by_pair = {f'{row["left_uid"]}|{row["right_uid"]}': row for row in accepted}
    ordered_events = [event_by_pair[pair] for pair in preferred_pairs if pair in event_by_pair]
    pinned = set(preferred_pairs)
    ordered_events.extend(row for row in accepted
                          if f'{row["left_uid"]}|{row["right_uid"]}' not in pinned)
    chosen, used_global_ids = [], set()
    for row in ordered_events:
        global_id = gid_by_uid.get(row["left_uid"], "-")
        left, right = person_by_uid.get(row["left_uid"]), person_by_uid.get(row["right_uid"])
        if not left or not right or left.get("camera_id") == right.get("camera_id"):
            continue
        if global_id in excluded_global_ids or global_id in used_global_ids:
            continue
        left_img, right_img = cv2.imread(left.get("crop_path", "")), cv2.imread(right.get("crop_path", ""))
        if left_img is None or right_img is None:
            continue
        panel = np.full((360, 720, 3), 245, np.uint8)
        for offset, crop in ((20, left_img), (380, right_img)):
            scale = min(320 / crop.shape[1], 250 / crop.shape[0])
            resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
            x, y = offset + (320 - resized.shape[1]) // 2, 50 + (250 - resized.shape[0]) // 2
            panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        _label(panel, f"CAM {left['camera_id']}", (20, 34), .7)
        _label(panel, f"CAM {right['camera_id']}", (380, 34), .7)
        distance, threshold = float(row["reid_distance"]), float(row["reid_threshold"])
        _label(panel, f"L2 {distance:.3f} < {threshold:.3f} / ACCEPT", (155, 335), .72)
        chosen.append({"global_id": global_id, "left_camera": left["camera_id"],
                       "right_camera": right["camera_id"], "distance": f"{distance:.4f}",
                       "threshold": f"{threshold:.4f}", "match_type": row["match_type"],
                       "bev_distance": f"{float(row['bev_distance']):.4f}",
                       "time_gap": f"{float(row['temporal_gap_seconds']):.2f}",
                       "src": _save(assets / f"reid_match_{len(chosen) + 1}.jpg", panel)})
        used_global_ids.add(global_id)
        if len(chosen) >= 12:
            break
    return chosen


def _global_id_explainer(cfg: dict, assets: Path, associations: list[dict], persons: list[dict],
                         mappings: list[dict]) -> dict:
    person_by_uid = {row["local_uid"]: row for row in persons}
    gid_by_uid = {row["local_uid"]: row["global_id"] for row in mappings}
    candidates = [row for row in associations if row.get("accepted") == "1" and row.get("reid_distance")]
    candidates.sort(key=lambda row: float(row.get("association_score") or 999))
    event = next((row for row in candidates
                  if person_by_uid.get(row["left_uid"], {}).get("camera_id") !=
                     person_by_uid.get(row["right_uid"], {}).get("camera_id")), None)
    if event is None:
        return {}
    left, right = person_by_uid[event["left_uid"]], person_by_uid[event["right_uid"]]
    left_crop, right_crop = cv2.imread(left.get("crop_path", "")), cv2.imread(right.get("crop_path", ""))
    bev = cv2.imread(str(cfg["bev_image"]))
    if left_crop is None or right_crop is None or bev is None:
        return {}
    width, height = 1600, 650
    canvas = np.full((height, width, 3), (244, 247, 249), np.uint8)
    panel_x = [25, 415, 805, 1195]
    panel_w = 360
    titles = ["1. LOCAL TRACKS", "2. SPACE + TIME", "3. ReID APPEARANCE", "4. GLOBAL ID"]
    for x, title in zip(panel_x, titles):
        cv2.rectangle(canvas, (x, 80), (x + panel_w, 610), (255, 255, 255), -1)
        cv2.rectangle(canvas, (x, 80), (x + panel_w, 610), (210, 220, 226), 2)
        _label(canvas, title, (x + 18, 120), .72)
    _label(canvas, "HOW TWO LOCAL IDs BECOME ONE GLOBAL ID (ACTUAL ACCEPTED CASE)", (30, 47), .92)
    for x in (390, 780, 1170):
        cv2.arrowedLine(canvas, (x - 25, 345), (x + 18, 345), (8, 126, 164), 7, cv2.LINE_AA, tipLength=.3)

    def place_crop(crop, x, y):
        fitted = np.full((190, 145, 3), 238, np.uint8)
        scale = min(145 / crop.shape[1], 190 / crop.shape[0])
        resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
        ox, oy = (145 - resized.shape[1]) // 2, (190 - resized.shape[0]) // 2
        fitted[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
        canvas[y:y + 190, x:x + 145] = fitted
        cv2.rectangle(canvas, (x, y), (x + 145, y + 190), (185, 198, 207), 2)

    place_crop(left_crop, panel_x[0] + 20, 165)
    place_crop(right_crop, panel_x[0] + 195, 165)
    _label(canvas, f"CAM {left['camera_id']}", (panel_x[0] + 28, 385), .58)
    _label(canvas, f"CAM {right['camera_id']}", (panel_x[0] + 203, 385), .58)
    _label(canvas, f"Local L{left['local_id']}", (panel_x[0] + 28, 420), .52)
    _label(canvas, f"Local L{right['local_id']}", (panel_x[0] + 203, 420), .52)
    _label(canvas, "Different camera IDs", (panel_x[0] + 62, 500), .6)

    map_view = cv2.resize(bev, (320, 280))
    mx, my = panel_x[1] + 20, 155
    canvas[my:my + 280, mx:mx + 320] = map_view
    p1 = (mx + int(float(left["end_bev_x"]) * 320), my + int(float(left["end_bev_y"]) * 280))
    p2 = (mx + int(float(right["start_bev_x"]) * 320), my + int(float(right["start_bev_y"]) * 280))
    cv2.line(canvas, p1, p2, (20, 120, 240), 4, cv2.LINE_AA)
    cv2.circle(canvas, p1, 10, (255, 170, 20), -1)
    cv2.circle(canvas, p2, 10, (40, 210, 100), -1)
    _label(canvas, f"BEV distance  {float(event['bev_distance']):.4f}", (panel_x[1] + 28, 480), .55)
    _label(canvas, f"Time gap     {float(event['temporal_gap_seconds']):.2f} sec", (panel_x[1] + 28, 515), .55)
    _label(canvas, f"Gate          {event['match_type']}", (panel_x[1] + 28, 550), .50)

    place_crop(left_crop, panel_x[2] + 20, 165)
    place_crop(right_crop, panel_x[2] + 195, 165)
    distance, threshold = float(event["reid_distance"]), float(event["reid_threshold"])
    ratio = min(1.0, distance / max(threshold, 1e-9))
    cv2.rectangle(canvas, (panel_x[2] + 28, 445), (panel_x[2] + 332, 475), (225, 233, 237), -1)
    cv2.rectangle(canvas, (panel_x[2] + 28, 445), (panel_x[2] + 28 + int(304 * ratio), 475), (45, 175, 110), -1)
    _label(canvas, f"L2 {distance:.3f}  <  threshold {threshold:.3f}", (panel_x[2] + 28, 515), .52)
    _label(canvas, "Appearance: PASS", (panel_x[2] + 65, 565), .68)

    gx = panel_x[3]
    cv2.circle(canvas, (gx + 180, 285), 105, (42, 171, 108), -1, cv2.LINE_AA)
    _label(canvas, "SAME", (gx + 115, 270), .95)
    _label(canvas, "PERSON", (gx + 100, 315), .95)
    gid = gid_by_uid.get(event["left_uid"], "-")
    short_gid = gid.rsplit(":", 1)[-1]
    _label(canvas, short_gid, (gx + 92, 445), .9)
    _label(canvas, "Space + Time + ReID", (gx + 62, 505), .58)
    _label(canvas, "ALL CONDITIONS PASS", (gx + 57, 550), .62)
    return {"src": _save(assets / "global_id_matching_explainer.jpg", canvas), "global_id": gid,
            "left_camera": left["camera_id"], "right_camera": right["camera_id"],
            "distance": f"{distance:.4f}", "threshold": f"{threshold:.4f}",
            "bev_distance": f"{float(event['bev_distance']):.4f}",
            "time_gap": f"{float(event['temporal_gap_seconds']):.2f}", "match_type": event["match_type"]}


def generate_report_assets(output_root: Path, cfg: dict, topology: list[dict], associations: list[dict],
                           persons: list[dict], mappings: list[dict]) -> dict:
    assets = output_root / "report/assets"
    assets.mkdir(parents=True, exist_ok=True)
    specs = _roi_specs(cfg)
    cameras = _camera_roi_images(cfg, assets, specs)
    coverage, topology_src = _bev_overview(cfg, assets, specs, topology)
    heatmap, trajectories, endpoints = _track_visuals(output_root, cfg, assets, mappings)
    boundaries = _boundary_images(cfg, assets, specs, persons, endpoints)
    excluded = set(cfg.get("visualization", {}).get("excluded_reid_report_global_ids", []))
    preferred = list(cfg.get("visualization", {}).get("reid_report_pairs", []))
    matches = _reid_gallery(output_root, assets, associations, persons, mappings, excluded, preferred)
    explainer = _global_id_explainer(cfg, assets, associations, persons, mappings)
    return {"camera_rois": cameras, "coverage": coverage, "topology": topology_src,
            "heatmap": heatmap, "trajectories": trajectories, "boundary_positions": boundaries,
            "reid_matches": matches, "global_id_explainer": explainer}
