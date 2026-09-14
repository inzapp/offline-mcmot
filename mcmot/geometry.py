from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def bbox_xyxy(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    left, top = np.maximum(a[:2], b[:2])
    right, bottom = np.minimum(a[2:], b[2:])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def point_in_polygon(point: tuple[float, float], vertices: Iterable[Iterable[float]]) -> bool:
    x, y = point
    poly = list(vertices)
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        crosses = (yi > y) != (yj > y)
        if crosses and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def distance_to_polygon(point: tuple[float, float], vertices: Iterable[Iterable[float]]) -> float:
    p = np.asarray(point, dtype=float)
    poly = np.asarray(list(vertices), dtype=float)
    if len(poly) < 2:
        return math.inf
    best = math.inf
    for a, b in zip(poly, np.vstack((poly[1:], poly[:1]))):
        delta = b - a
        ratio = np.clip(np.dot(p - a, delta) / max(np.dot(delta, delta), 1e-12), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (a + ratio * delta))))
    return best


def signed_angle(ax: float, ay: float, bx: float, by: float,
                 cx: float, cy: float) -> float:
    v1 = np.array([ax - bx, ay - by], dtype=float)
    v2 = np.array([cx - bx, cy - by], dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    v1 /= n1
    v2 /= n2
    return math.degrees(math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2))


def is_watch(row: dict, cfg: dict) -> bool:
    names = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
    threshold = float(cfg["keypoint_confidence"])
    if any(float(row.get(f"{name}_conf", 0.0)) < threshold for name in names):
        return False
    angle = (float(cfg["small_angle_deg"]) if float(row["w"]) < float(cfg["small_bbox_width"])
             else float(cfg["normal_angle_deg"]))
    eye_dx = abs(float(row["right_eye_x"]) - float(row["left_eye_x"]))
    side_l = abs(float(row["left_ear_x"]) - float(row["left_eye_x"]))
    side_r = abs(float(row["right_ear_x"]) - float(row["right_eye_x"]))
    ratio_threshold = 1.4 - angle * (0.1 / 15.0)
    if eye_dx / (max(side_l, side_r) + 1e-7) < ratio_threshold:
        return False
    target = "left_ear" if float(row["left_ear_y"]) < float(row["right_ear_y"]) else "right_ear"
    y_angle = signed_angle(float(row["nose_x"]), float(row[f"{target}_y"]),
                           float(row[f"{target}_x"]), float(row[f"{target}_y"]),
                           float(row["nose_x"]), float(row["nose_y"]))
    return abs(y_angle) <= angle * 0.5


def gaze_ray(row: dict, confidence_threshold: float = 0.3) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the normalized image-space gaze ray used by adddi_light OOI.

    The preferred direction is ear-midpoint -> nose.  When nose/eye confidence
    is insufficient, the perpendicular to the ear line is used, matching
    ``tw_get_gaze_ray`` in adddi_light/app_main/src/tracker_wrapper.cpp.
    """
    def point(name: str) -> np.ndarray:
        return np.array([float(row[f"{name}_x"]), float(row[f"{name}_y"])], dtype=float)

    def valid(name: str) -> bool:
        return float(row.get(f"{name}_conf") or 0) >= confidence_threshold

    if not (valid("left_ear") or valid("right_ear")):
        return None
    left_ear, right_ear = point("left_ear"), point("right_ear")
    ear_mid = (left_ear + right_ear) * .5
    ear_delta = right_ear - left_ear
    if np.linalg.norm(ear_delta) < 1e-9:
        return None
    if valid("nose") and (valid("left_eye") or valid("right_eye")):
        origin = point("nose")
        direction = origin - ear_mid
    else:
        origin = ear_mid
        direction = np.array([ear_delta[1], -ear_delta[0]], dtype=float)
    if np.linalg.norm(direction) < 1e-9:
        return None
    bounds = []
    for axis in range(2):
        if direction[axis] > 1e-9:
            bounds.append((1.0 - origin[axis]) / direction[axis])
        elif direction[axis] < -1e-9:
            bounds.append((0.0 - origin[axis]) / direction[axis])
    positive = [value for value in bounds if value > 0]
    if not positive:
        return None
    end = origin + direction * min(positive)
    return np.clip(origin, 0, 1), np.clip(end, 0, 1)


def homography_from_points(source: Iterable[Iterable[float]],
                           target: Iterable[Iterable[float]]) -> np.ndarray:
    import cv2
    matrix, _ = cv2.findHomography(np.asarray(list(source), dtype=np.float64),
                                   np.asarray(list(target), dtype=np.float64), 0)
    if matrix is None:
        raise ValueError("homography could not be estimated")
    return matrix


def transform_point(point: Iterable[float], matrix: np.ndarray) -> tuple[float, float]:
    value = np.asarray([*point, 1.0], dtype=float)
    mapped = matrix @ value
    if abs(mapped[2]) < 1e-12:
        raise ValueError("point projects to infinity")
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def segment_intersects_polygon(start: Iterable[float], end: Iterable[float],
                               vertices: Iterable[Iterable[float]]) -> bool:
    """OOI-style forward-ray intersection with a closed polygon."""
    p, q = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    direction = q - p
    polygon = np.asarray(list(vertices), dtype=float)
    if len(polygon) < 3 or np.linalg.norm(direction) < 1e-12:
        return False
    if point_in_polygon(tuple(p), polygon) or point_in_polygon(tuple(q), polygon):
        return True
    cross = lambda a, b: float(a[0] * b[1] - a[1] * b[0])
    for a, b in zip(polygon, np.vstack((polygon[1:], polygon[:1]))):
        edge = b - a
        denom = cross(direction, edge)
        if abs(denom) < 1e-12:
            continue
        delta = a - p
        t, u = cross(delta, edge) / denom, cross(delta, direction) / denom
        if 0 <= t <= 1 and 0 <= u <= 1:
            return True
    return False
