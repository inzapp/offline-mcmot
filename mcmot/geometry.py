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
