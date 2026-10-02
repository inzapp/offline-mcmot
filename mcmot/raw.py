"""Adapted from ultralytics/video_pose_to_csv.py: detection + pose, pose-first NMS."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

from .preparation import file_signature

KEYPOINT_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
                  "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                  "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
DEFAULTS = dict(det_model="yolov8m.pt", pose_model="yolov8m-pose.pt", det_imgsz=896,
                pose_imgsz=1152, det_conf=.05, pose_conf=.02, merge_iou=.45,
                model_iou=.7, max_det=10000, device=None, timestamp_fps=None, person_class=0)


def csv_columns():
    return ["timestamp", "frame_index", "confidence", "cx", "cy", "w", "h"] + [
        f"{name}_{field}" for name in KEYPOINT_NAMES for field in ("conf", "x", "y")]


def parse_start_time(path):
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})_.+_(\d{6})", Path(path).stem)
    if not match:
        raise ValueError(f"파일명은 YYYY-MM-DD_DEVICE_HHMMSS 형식이어야 합니다: {path}")
    return datetime.strptime("_".join(match.groups()), "%Y-%m-%d_%H%M%S")


def merged_detections(pose, detection, threshold):
    from .geometry import bbox_xyxy, iou
    # Pose always has priority over detector confidence. Within each source use confidence.
    candidates = sorted(pose, key=lambda r: -r['confidence']) + sorted(detection, key=lambda r: -r['confidence'])
    kept = []
    for row in candidates:
        box = bbox_xyxy(*(row[k] for k in ('cx', 'cy', 'w', 'h')))
        if all(iou(box, other) <= threshold for _, other in kept):
            kept.append((row, box))
    return [row for row, _ in kept]


def result_rows(result, conf, is_pose):
    if result.boxes is None:
        return []
    boxes, scores = result.boxes.xywhn.cpu().tolist(), result.boxes.conf.cpu().tolist()
    # Ultralytics may omit keypoint confidence when no pose boxes were detected.
    if not boxes:
        return []
    points = confidences = None
    if is_pose:
        if result.keypoints is None or result.keypoints.conf is None:
            raise ValueError("17개 keypoint confidence를 출력하는 pose 모델이 필요합니다")
        points, confidences = result.keypoints.xyn.cpu().tolist(), result.keypoints.conf.cpu().tolist()
    rows = []
    for index, (box, score) in enumerate(zip(boxes, scores)):
        if score <= conf:
            continue
        row = dict(zip(('cx', 'cy', 'w', 'h'), box), confidence=score)
        if is_pose and len(points[index]) != 17:
            raise ValueError("COCO 17 keypoint pose 모델이 필요합니다")
        for k, name in enumerate(KEYPOINT_NAMES):
            row.update({f'{name}_conf': confidences[index][k] if is_pose else 0,
                        f'{name}_x': points[index][k][0] if is_pose else 0,
                        f'{name}_y': points[index][k][1] if is_pose else 0})
        rows.append(row)
    return rows


class Extractor:
    def __init__(self, options):
        self.options = {**DEFAULTS, **options}
        for key in ('det_conf', 'pose_conf', 'merge_iou', 'model_iou'):
            if not 0 <= self.options[key] <= 1:
                raise ValueError(f'{key} must be in [0, 1]')
        for key in ('det_imgsz', 'pose_imgsz', 'max_det'):
            if self.options[key] <= 0:
                raise ValueError(f'{key} must be positive')
        if self.options['timestamp_fps'] is not None and self.options['timestamp_fps'] <= 0:
            raise ValueError('timestamp_fps must be positive')
        self.models = None

    def convert(self, video, output=None, force=False):
        import cv2
        video = Path(video); output = Path(output) if output else video.with_suffix('.csv')
        meta = output.with_suffix('.csv.meta.json')
        signature = {'video': file_signature(video), 'options': self.options, 'version': 1}
        for key in ('det_model', 'pose_model'):
            model = Path(self.options[key])
            if model.is_file():
                signature[key] = file_signature(model)
        if output.exists() and not force:
            if meta.exists():
                saved = json.loads(meta.read_text())
                if saved.get('inputs') == signature and saved.get('output') == file_signature(output):
                    print(f'RAW 재사용: {output}', flush=True)
                    return output
            raise FileExistsError(f'기존 raw를 덮어쓰려면 --force 필요: {output}')
        capture = cv2.VideoCapture(str(video))
        try:
            if not capture.isOpened():
                raise RuntimeError(f'영상 열기 실패: {video}')
            fps = self.options['timestamp_fps'] or capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError(f'FPS 확인 실패: {video}; --timestamp-fps 필요')
            start = parse_start_time(video)
            if self.models is None:
                from ultralytics import YOLO
                # The shared MiVOLO environment retains YOLO 8.1 checkpoints. Limit the
                # legacy pickle-loading compatibility override to loading these models.
                import torch
                from functools import partial
                from unittest.mock import patch
                with patch('torch.load', partial(torch.load, weights_only=False)):
                    self.models = (YOLO(self.options['det_model']), YOLO(self.options['pose_model']))
                for key in ('det_model', 'pose_model'):
                    if Path(self.options[key]).is_file():
                        signature[key] = file_signature(self.options[key])
            common = {'verbose': False, 'max_det': self.options['max_det'], 'iou': self.options['model_iou']}
            if self.options['device'] is not None:
                common['device'] = self.options['device']
            output.parent.mkdir(parents=True, exist_ok=True)
            partial = output.with_suffix('.csv.partial')
            index, count = 0, 0
            with partial.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=csv_columns()); writer.writeheader()
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    det = self.models[0].predict(frame, classes=[self.options['person_class']], conf=self.options['det_conf'], imgsz=self.options['det_imgsz'], **common)[0]
                    pose = self.models[1].predict(frame, conf=self.options['pose_conf'], imgsz=self.options['pose_imgsz'], **common)[0]
                    rows = merged_detections(result_rows(pose, self.options['pose_conf'], True),
                                             result_rows(det, self.options['det_conf'], False), self.options['merge_iou'])
                    stamp = (start + timedelta(seconds=index / fps)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    for row in rows:
                        writer.writerow({'timestamp': stamp, 'frame_index': index, **{k: f'{v:.6f}' for k, v in row.items()}})
                    count += len(rows); index += 1
                    if index % 100 == 0:
                        print(f'RAW {video.name}: {index:,} frames / {count:,} rows', flush=True)
            expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not index or (expected > 0 and index != expected):
                raise RuntimeError(f'영상 디코딩 미완료: {index}/{expected} frames: {video}')
            partial.replace(output)
            meta.write_text(json.dumps({'inputs': signature, 'output': file_signature(output), 'frames': index, 'rows': count}, indent=2))
            print(f'RAW 생성: {output} ({index:,} frames / {count:,} rows)', flush=True)
            return output
        finally:
            capture.release()


def add_arguments(parser):
    for key, value in DEFAULTS.items():
        kind = int if key in ('det_imgsz', 'pose_imgsz', 'max_det', 'person_class') else float if key in ('det_conf', 'pose_conf', 'merge_iou', 'model_iou', 'timestamp_fps') else str
        parser.add_argument('--' + key.replace('_', '-'), type=kind, default=None,
                            help=f'기본값: {value}; CLI > YAML extraction > 기본값')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--force', action='store_true'); add_arguments(parser)
    args = parser.parse_args()
    options = {k: getattr(args, k) for k in DEFAULTS if getattr(args, k) is not None}
    Extractor(options).convert(args.video, args.output, args.force)


if __name__ == '__main__':
    main()
