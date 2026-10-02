"""MiVOLO V2 스냅샷 추론: 얼굴 검출 → 얼굴/몸 전처리 → 나이·성별 → 저장.

전체 CCTV 화면이 아니라 대상 사람 한 명의 crop을 입력한다.
얼굴이 없거나 여러 개면 몸만 사용하며, 추정 결과는 실제 나이/성별 정답이 아니다.
명령행 사용법은 루트의 mivolo_snapshot.py 주석과 --help를 참고한다.
"""
import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time

# 설치 위치를 기준으로 모델과 기본 출력 경로를 찾으므로 폴더째 옮겨도 동작한다.
ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def annotate_snapshot(bgr, row):
    import cv2
    import numpy as np
    annotated = bgr.copy()
    if row['face_bbox']:
        x1,y1,x2,y2 = row['face_bbox']
        cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
    lines = [f'Age: ~{row["age"]}', f'Gender: {row["gender"]}',
             f'Male: {row["male_score"]:.4f}', f'Female: {row["female_score"]:.4f}',
             row['input_mode']]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, .48, 1
    width = max(annotated.shape[1], max(cv2.getTextSize(s,font,scale,thickness)[0][0] for s in lines)+16)
    header = len(lines)*23+12
    canvas = np.zeros((annotated.shape[0]+header,width,3),dtype=np.uint8)
    left = (width-annotated.shape[1])//2
    canvas[header:,left:left+annotated.shape[1]] = annotated
    for i,line in enumerate(lines):
        cv2.putText(canvas,line,(8,23+i*23),font,scale,(255,255,255),thickness,cv2.LINE_AA)
    return canvas


def select_face(candidates, width, height, minimum_size=12):
    """Clip detections and reject ambiguous faces rather than choosing another person."""
    # 이미지 경계로 좌표를 자른 뒤 최소 크기를 검사한다. 얼굴 여러 개는 연결하지 않는다.
    valid = []
    for bbox, score in candidates:
        if not all(math.isfinite(float(v)) for v in [*bbox, score]):
            continue
        x1, y1 = max(0, math.floor(bbox[0])), max(0, math.floor(bbox[1]))
        x2, y2 = min(width, math.ceil(bbox[2])), min(height, math.ceil(bbox[3]))
        if min(x2-x1, y2-y1) >= minimum_size:
            valid.append(([x1, y1, x2, y2], float(score)))
    if not valid:
        return None, None, 'no_usable_face'
    if len(valid) > 1:
        return None, None, 'multiple_faces'
    return *valid[0], 'detected'


class SnapshotEstimator:
    def __init__(self, device='auto', threads=4, face_conf=0.4, min_face_size=12,
                 body_only=False, detector_size=640, model_dir=None, detector_path=None, repository=None):
        # 모델은 로컬 weights에서만 읽고, 라이브러리 캐시는 프로젝트 내부에 둔다.
        os.environ.setdefault('HF_HOME', str(ROOT / 'work/mivolo/hf-cache'))
        os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'work/mivolo/yolo'))
        os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/mivolo/mpl'))
        os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / 'work/mivolo/cache'))
        os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
        import sys
        if repository:
            sys.path.insert(0, str(repository))
        import torch
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        self.torch = torch
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA 사용 불가: NVIDIA 드라이버와 GPU용 PyTorch를 확인하거나 --device cpu를 사용하세요.')
        self.device = torch.device(device)
        # CPU는 float32, GPU는 메모리/속도를 위해 float16으로 추론한다.
        self.dtype = torch.float32 if self.device.type == 'cpu' else torch.float16
        torch.set_num_threads(threads)
        self.face_conf, self.min_face_size = face_conf, min_face_size
        self.body_only, self.detector_size = body_only, detector_size
        model_dir = Path(model_dir) if model_dir else ROOT / 'weights/mivolo_v2/age_gender'
        self.model = AutoModelForImageClassification.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True,
            torch_dtype=self.dtype).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True)
        self.detector = None
        if not body_only:
            from ultralytics import YOLO
            # PyTorch 2.6+의 기본 weights_only=True는 기존 YOLO 8.1 체크포인트와
            # 호환되지 않는다. 배포한 공식 파일의 SHA-256을 확인한 뒤 이 로딩에만
            # weights_only=False를 적용한다 (다른 torch.load 호출에는 영향 없음).
            import hashlib
            from functools import partial
            from unittest.mock import patch
            detector_path = Path(detector_path) if detector_path else ROOT / 'weights/mivolo_v2/yolov8x_person_face.pt'
            expected = '2620f45609a65f909eb876bd7401308b5a8f3843ad5a03cb7416066a3e492989'
            if hashlib.sha256(detector_path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('얼굴 검출기 체크섬 불일치: 배포한 공식 모델 파일을 확인하세요.')
            with patch('torch.load', partial(torch.load, weights_only=False)):
                self.detector = YOLO(str(detector_path))
            self.face_ids = [int(k) for k,v in self.detector.names.items() if v.lower() == 'face']
            if not self.face_ids:
                raise RuntimeError(f'검출기에 face 클래스가 없습니다: {self.detector.names}')

    def sync(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)

    def predict(self, bgr):
        """Accept an H×W×3 uint8 BGR snapshot containing one target person."""
        # OpenCV 형식(uint8 BGR)의 한 사람 이미지를 받는다. 모델 객체는 재사용한다.
        h, w = bgr.shape[:2]
        self.sync()
        start = time.perf_counter()
        bbox, face_score, reason = None, None, 'disabled'
        if self.detector is not None:
            detections = self.detector.predict(bgr, device=str(self.device),
                classes=self.face_ids, conf=self.face_conf, imgsz=self.detector_size,
                half=self.device.type == 'cuda', verbose=False)[0]
            candidates = [(b.xyxy[0].cpu().tolist(), float(b.conf[0])) for b in detections.boxes]
            bbox, face_score, reason = select_face(candidates, w, h, self.min_face_size)
        self.sync()
        detected = time.perf_counter()
        # 얼굴 누락을 임의의 머리 crop으로 대체하지 않는다. 공식 전처리기가 None을 처리한다.
        face = None if bbox is None else bgr[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
        faces = self.processor(images=[face])['pixel_values'].to(device=self.device, dtype=self.dtype)
        bodies = self.processor(images=[bgr])['pixel_values'].to(device=self.device, dtype=self.dtype)
        self.sync()
        prepared = time.perf_counter()
        # 얼굴과 전체 몸을 두 입력으로 전달한다. 학습/gradient 계산은 하지 않는다.
        with self.torch.inference_mode():
            out = self.model(faces_input=faces, body_input=bodies)
        self.sync()
        end = time.perf_counter()
        age = float(out.age_output.item())
        score = float(out.gender_probs.item())
        if not math.isfinite(age) or not math.isfinite(score):
            raise RuntimeError('모델 출력에 유효하지 않은 값이 있습니다.')
        idx = int(out.gender_class_idx.item())
        labels = self.model.config.gender_id2label
        gender = labels[idx] if idx in labels else labels[str(idx)]
        probabilities = out.raw_gender_output.float().softmax(dim=-1)[0].cpu().tolist()
        gender_scores = {str(labels[i] if i in labels else labels[str(i)]).lower():float(v)
                         for i,v in enumerate(probabilities)}
        return {'age': round(age, 2), 'gender': gender, 'gender_score': score,
                'male_score': gender_scores['male'], 'female_score': gender_scores['female'],
                'input_mode': 'body_only' if face is None else 'face_body',
                'face_status': reason, 'face_bbox': bbox, 'face_detection_score': face_score,
                'image_width': w, 'image_height': h,
                'detection_ms': round((detected-start)*1000,2),
                'preprocess_ms': round((prepared-detected)*1000,2),
                'inference_ms': round((end-prepared)*1000,2),
                'pipeline_ms': round((end-start)*1000,2)}, face
