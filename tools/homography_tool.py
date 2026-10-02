#!/usr/bin/env python3
"""두 이미지의 대응점으로 호모그래피를 계산하고 양방향 투영과 ROI를 확인한다."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def project(point, matrix):
    # [x,y,1]에 H를 곱하고 마지막 성분으로 나눠 원본 픽셀 좌표로 되돌린다.
    result = matrix @ np.array([*point, 1.0], dtype=float)
    if not np.isfinite(result).all() or abs(result[2]) < 1e-10:
        return None
    result = result[:2] / result[2]
    return tuple(result) if np.isfinite(result).all() else None


def estimate(source, target, threshold=5.0):
    a, b = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2 or len(a) < 4:
        raise ValueError("대응점을 최소 4쌍 선택하세요.")
    for points in (a, b):
        if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != len(points):
            raise ValueError("중복되거나 유효하지 않은 점이 있습니다.")
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError("점들을 한 직선 위가 아닌 이미지 전역에 분산해서 선택하세요.")
    # 잘못 찍은 점은 RANSAC으로 제외한다. threshold는 대상 원본 이미지의 픽셀 단위다.
    matrix, mask = cv2.findHomography(a, b, cv2.RANSAC, threshold)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("호모그래피 계산 실패: 대응점 배치를 확인하세요.")
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 4:
        raise ValueError("유효한 대응점이 4쌍 미만입니다.")
    mapped = [project(point, matrix) for point in a]
    if any(point is None for point in mapped):
        raise ValueError("일부 대응점이 무한대로 투영됩니다. 점 배치를 확인하세요.")
    errors = np.linalg.norm(np.array(mapped) - b, axis=1)
    return matrix, inliers, errors


# Fixed BGR palette: cyan, orange, magenta, green, blue.
ROI_COLORS = [(255, 220, 0), (0, 150, 255), (220, 50, 220), (60, 210, 60), (255, 90, 60)]

# Bright BGR colors, shared by each pin's source and projected point.
PIN_COLORS = [
    (0, 255, 255), (255, 255, 0), (255, 80, 255), (80, 255, 80),
    (0, 160, 255), (255, 130, 70), (100, 100, 255), (255, 100, 180),
    (150, 255, 210), (255, 210, 150), (180, 150, 255), (80, 210, 255),
]


class HomographyTool:
    def __init__(self, paths, output, max_width=1000, max_height=700, threshold=5.0):
        self.paths = [Path(p).resolve() for p in paths]
        self.images = [read_image(p) for p in self.paths]
        # 축소 화면은 표시용이다. 클릭 좌표는 원본 좌표로 환산해서 저장한다.
        self.views, self.scales, self.offsets = [], [], []
        for image in self.images:
            h, w = image.shape[:2]
            scale = min(max_width / w, max_height / h, 1.0)
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            self.views.append(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
            self.scales.append(np.array([size[0] / w, size[1] / h]))
            self.offsets.append(np.maximum(1, np.ceil(np.array(size) * .05)).astype(int))
        self.names = ['Image 1 - source', 'Image 2 - target']
        self.output = Path(output)
        self.threshold = threshold
        self.points = [[], []]
        self.pending = None
        self.matrix = self.inverse = self.inliers = self.errors = None
        self.cursor = [None, None]
        self.pins = []
        self.rois = [None] * 5
        self.roi_index = None
        self.roi_mode = False
        self.roi_side = None
        self.roi_draft = []

    @classmethod
    def load(cls, path, output=None, max_width=1000, max_height=700, image_paths=None):
        path = Path(path).resolve()
        if path.is_dir():
            path = path / 'homography.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or data.get('version') != 1:
            raise ValueError('지원하지 않는 저장 파일 형식입니다.')
        if data.get('coordinate_system') != 'original_image_pixels' or data.get('direction') != 'image1_to_image2':
            raise ValueError('저장 파일의 좌표계 또는 변환 방향이 맞지 않습니다.')
        paths = image_paths or data.get('images')
        if not isinstance(paths, (list, tuple)) or len(paths) != 2 or not all(isinstance(p, str) for p in paths):
            raise ValueError('저장 파일에 두 이미지 경로가 필요합니다.')
        paths = [Path(p) if Path(p).is_absolute() else path.parent / p for p in paths]
        threshold = float(data.get('ransac_threshold_px', 5.0))
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError('유효하지 않은 RANSAC 임계값입니다.')
        tool = cls(paths, output or Path(__file__).resolve().parents[1] / 'homography_results',
                   max_width, max_height, threshold)
        if data.get('image_sizes') != [[im.shape[1], im.shape[0]] for im in tool.images]:
            raise ValueError('이미지 크기가 저장 당시와 다릅니다. 같은 해상도의 원본을 사용하세요.')
        a = np.asarray(data.get('points1'), dtype=float)
        b = np.asarray(data.get('points2'), dtype=float)
        if a.ndim != 2 or a.shape[1:] != (2,) or len(a) < 4 or a.shape != b.shape or not np.isfinite([a, b]).all():
            raise ValueError('저장된 대응점 형식이 올바르지 않습니다.')
        matrix = np.asarray(data.get('H'), dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
            raise ValueError('저장된 호모그래피 행렬이 올바르지 않습니다.')
        mapped = [project(point, matrix) for point in a]
        if any(point is None for point in mapped):
            raise ValueError('저장된 대응점이 무한대로 투영됩니다.')
        inliers = np.asarray(data.get('inliers'), dtype=bool)
        if inliers.shape != (len(a),) or inliers.sum() < 4:
            raise ValueError('저장된 유효점 정보가 올바르지 않습니다.')
        pins = np.asarray(data.get('projection_pairs', []), dtype=float)
        if pins.size and (pins.ndim != 3 or pins.shape[1:] != (2, 2) or not np.isfinite(pins).all()):
            raise ValueError('저장된 확인점 형식이 올바르지 않습니다.')
        tool.points = [a.tolist(), b.tolist()]
        tool.matrix, tool.inverse = matrix, np.linalg.inv(matrix)
        tool.inliers = inliers
        tool.errors = np.linalg.norm(np.array(mapped) - b, axis=1)
        tool.pins = pins.tolist() if pins.size else []
        saved_rois = data.get('rois')
        if saved_rois is None:
            saved_rois = [dict(data['roi'], id=1)] if data.get('roi') is not None else []
        if not isinstance(saved_rois, list) or len(saved_rois) > 5:
            raise ValueError('ROI는 최대 5개까지 지원합니다.')
        for roi in saved_rois:
            if not isinstance(roi, dict):
                raise ValueError('저장된 ROI 형식이 잘못되었습니다.')
            rid = roi.get('id')
            if type(rid) is not int or not 1 <= rid <= 5 or tool.rois[rid-1] is not None:
                raise ValueError('ROI 번호는 중복 없는 1~5여야 합니다.')
            tool.validate_roi(roi.get('source_image'), roi.get('vertices_px'))
            tool.rois[rid-1] = dict(source_image=roi['source_image'], vertices_px=roi['vertices_px'])
        print(f'불러오기 완료: {path} ({len(a)}쌍). E: 수정, S: 새 버전 저장')
        return tool

    def roi_pair(self, side, points):
        matrix = self.matrix if side == 0 else self.inverse
        points = np.asarray(points, dtype=float)
        denominators = np.c_[points, np.ones(len(points))] @ matrix[2]
        if np.any(np.abs(denominators) < 1e-10) or not (np.all(denominators > 0) or np.all(denominators < 0)):
            raise ValueError('ROI가 투영 무한선을 가로지릅니다. 영역을 줄여 주세요.')
        mapped = [project(point, matrix) for point in points]
        if any(point is None for point in mapped):
            raise ValueError('ROI 투영 좌표를 계산할 수 없습니다.')
        pair = [None, None]
        pair[side], pair[1-side] = points.tolist(), np.asarray(mapped).tolist()
        return pair

    def validate_roi(self, side, points):
        if self.matrix is None:
            raise ValueError('호모그래피 계산 후 ROI를 지정하세요.')
        if side not in (0, 1):
            raise ValueError('ROI 기준 이미지가 잘못되었습니다.')
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 4 or not np.isfinite(points).all():
            raise ValueError('ROI 꼭짓점을 순서대로 최소 4개 선택하세요.')
        h, w = self.images[side].shape[:2]
        normalized = points / [w, h]
        if np.any(normalized < -.01 - 1e-10) or np.any(normalized > 1.01 + 1e-10):
            raise ValueError('ROI 꼭짓점은 정규화 좌표 -0.01~1.01 범위여야 합니다.')
        if len(np.unique(points, axis=0)) != len(points):
            raise ValueError('ROI에 중복 꼭짓점이 있습니다.')
        # Nonadjacent edges must not intersect, including collinear overlaps.
        def cross(a, b, c):
            ab, ac = b-a, c-a
            return ab[0]*ac[1]-ab[1]*ac[0]
        def on(a, b, c):
            return abs(cross(a,b,c)) < 1e-8 and np.all(c >= np.minimum(a,b)-1e-8) and np.all(c <= np.maximum(a,b)+1e-8)
        n = len(points)
        for i in range(n):
            a, b = points[i], points[(i+1)%n]
            for j in range(i+1,n):
                if j == i+1 or (i == 0 and j == n-1):
                    continue
                c, d = points[j], points[(j+1)%n]
                if (cross(a,b,c)*cross(a,b,d) < 0 and cross(c,d,a)*cross(c,d,b) < 0) or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b):
                    raise ValueError('ROI 선분이 교차합니다. 둘레 순서대로 선택하세요.')
        if abs(cv2.contourArea(points.astype(np.float32))) < 1e-6:
            raise ValueError('ROI 면적이 0입니다.')
        self.roi_pair(side, points)

    def start_roi(self, index=None):
        if self.matrix is None:
            raise ValueError('호모그래피 계산 후 P를 누르세요.')
        if index is None:
            index = next((i for i, roi in enumerate(self.rois) if roi is None), None)
            if index is None:
                raise ValueError('ROI는 최대 5개입니다. 숫자 1~5로 수정할 ROI를 선택하세요.')
        self.roi_index = index
        roi = self.rois[index]
        self.roi_mode = True
        self.roi_side = roi['source_image'] if roi else None
        self.roi_draft = [list(p) for p in roi['vertices_px']] if roi else []
        self.cursor = [None, None]
        print(f'ROI {index+1}: 4점 이상 클릭 → Enter 확정. Z 취소, C 비우기, D ROI 삭제, P 편집 취소')

    def finish_roi(self):
        self.validate_roi(self.roi_side, self.roi_draft)
        self.rois[self.roi_index] = dict(source_image=self.roi_side, vertices_px=[list(p) for p in self.roi_draft])
        self.roi_mode = False
        print(f'ROI {self.roi_index+1} 정규화 좌표:', json.dumps(self.roi_payload(self.roi_index)['vertices_normalized']))

    def select_roi(self, side, point):
        hits = []
        for index, roi in enumerate(self.rois):
            if roi is None:
                continue
            polygon = np.asarray(self.roi_pair(roi['source_image'], roi['vertices_px'])[side], dtype=np.float32)
            if cv2.pointPolygonTest(polygon, tuple(map(float, point)), False) >= 0:
                hits.append(index)
        if not hits:
            self.roi_index = None
            print('ROI 선택 해제')
            return
        # Repeated right clicks cycle through overlapping polygons.
        self.roi_index = hits[(hits.index(self.roi_index)+1) % len(hits)] if self.roi_index in hits else hits[0]
        print(f'ROI {self.roi_index+1} 선택됨. D: 삭제, 숫자 {self.roi_index+1}: 수정')

    def delete_roi(self):
        if self.roi_index is None:
            print('먼저 ROI 내부를 우클릭해서 선택하세요.')
            return
        index = self.roi_index
        self.rois[index] = None
        self.roi_mode = False
        self.roi_draft = []
        self.roi_side = None
        self.roi_index = None
        print(f'ROI {index+1} 삭제됨. S를 눌러 변경 사항을 저장하세요.')

    def roi_payload(self, index):
        roi = self.rois[index]
        if roi is None:
            return None
        side = roi['source_image']
        self.validate_roi(side, roi['vertices_px'])
        pair = self.roi_pair(side, roi['vertices_px'])
        h, w = self.images[side].shape[:2]
        return dict(**roi, id=index+1, color_bgr=list(ROI_COLORS[index]),
                    normalization='x / width, y / height', normalized_range=[-.01, 1.01],
                    vertices_normalized=(np.asarray(pair[side]) / [w,h]).tolist(),
                    image1_vertices_px=pair[0], image2_vertices_px=pair[1],
                    image1_vertices_normalized=(np.asarray(pair[0]) / [self.images[0].shape[1], self.images[0].shape[0]]).tolist(),
                    image2_vertices_normalized=(np.asarray(pair[1]) / [self.images[1].shape[1], self.images[1].shape[0]]).tolist(),
                    image1_size=[self.images[0].shape[1], self.images[0].shape[0]],
                    image2_size=[self.images[1].shape[1], self.images[1].shape[0]],
                    normalized_range_applies_to='vertices_normalized (source image only)')

    def draw_rois(self, image, side, scale, offset=(0, 0)):
        for index, roi in enumerate(self.rois):
            draft = self.roi_mode and index == self.roi_index
            if draft:
                roi = dict(source_image=self.roi_side, vertices_px=self.roi_draft)
            image = self.draw_roi(image, side, scale, roi, index, draft, offset)
        return image

    def draw_roi(self, image, side, scale, roi, index, draft=False, offset=(0, 0)):
        color = ROI_COLORS[index]
        if roi is None or not roi['vertices_px'] or self.matrix is None:
            return image
        points = roi['vertices_px']
        if side != roi['source_image']:
            try:
                points = self.roi_pair(roi['source_image'], points)[side]
            except ValueError:
                return image
        # Bound raster coordinates to avoid integer overflow near the horizon.
        poly = np.rint(np.clip(np.asarray(points)*scale + offset, -1000000, 1000000)).astype(np.int32)
        if not draft and len(poly) >= 4:
            layer = image.copy()
            cv2.fillPoly(layer, [poly], color)
            image = cv2.addWeighted(layer, .25, image, .75, 0)
        cv2.polylines(image, [poly], not draft, color,
                      4 if index == self.roi_index else 2, cv2.LINE_AA)
        for i, (x,y) in enumerate(poly):
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(image, (int(x),int(y)), 4, color, -1)
                cv2.putText(image, f'ROI{index+1}:{i+1}', (int(x)+6,int(y)+16), cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
        return image

    def mouse(self, side, event, x, y, flags, param):
        h, w = self.views[side].shape[:2]
        ox, oy = self.offsets[side]
        if not (0 <= x < w + 2*ox and 0 <= y < h + 2*oy):
            return  # Excludes the help footer.
        local = np.array([x, y]) - [ox, oy]
        inside = 0 <= local[0] < w and 0 <= local[1] < h
        point = tuple(local / self.scales[side])
        if self.roi_mode:
            original_h, original_w = self.images[side].shape[:2]
            normalized = np.asarray(point) / [original_w, original_h]
            normalized = np.where(normalized < 0, -.01,
                                  np.where(normalized >= 1, 1.01, normalized))
            point = tuple(normalized * [original_w, original_h])
            if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
                self.cursor[side] = point
                self.cursor[1 - side] = project(point, self.matrix if side == 0 else self.inverse)
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.roi_side is None:
                    self.roi_side = side
                if side == self.roi_side:
                    self.roi_draft.append(point)
                else:
                    print('ROI 점은 처음 선택한 이미지에서 계속 찍으세요. C로 초기화하면 변경할 수 있습니다.')
            return
        if self.matrix is not None and event == cv2.EVENT_RBUTTONDOWN:
            self.select_roi(side, point)
            return
        if not inside:
            self.cursor = [None, None]
            return
        if self.matrix is None:
            if event == cv2.EVENT_RBUTTONDOWN and self.points[side]:
                distances = np.linalg.norm(np.asarray(self.points[side]) * self.scales[side] - local, axis=1)
                index = int(np.argmin(distances))
                if distances[index] <= 15:
                    for points in self.points:
                        points.pop(index)
                    print(f'대응점 {index + 1}쌍 삭제됨')
                return
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            if side == 0:
                self.pending = point
            elif self.pending is not None:
                self.points[0].append(self.pending)
                self.points[1].append(point)
                self.pending = None
                print(f"대응점 {len(self.points[0])}쌍 선택됨")
        elif event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self.cursor[side] = point
            self.cursor[1 - side] = project(point, self.matrix if side == 0 else self.inverse)
            if event == cv2.EVENT_LBUTTONDOWN and all(p is not None for p in self.cursor):
                self.pins.append(tuple(self.cursor))
                print(f"확인점 {len(self.pins)}: {self.cursor[0]} -> {self.cursor[1]}")

    def calculate(self):
        # H는 이미지1 → 이미지2, inverse는 이미지2 → 이미지1 변환이다.
        if self.pending is not None:
            raise ValueError("Image 2의 대응점을 선택하거나 Z로 미완성 점을 취소하세요.")
        matrix, inliers, errors = estimate(*self.points, self.threshold)
        inverse = np.linalg.inv(matrix)
        self.matrix, self.inverse = matrix, inverse
        self.inliers, self.errors = inliers, errors
        print('H (원본 이미지 1 좌표 -> 원본 이미지 2 좌표):\n', matrix)
        print(f'RANSAC 유효점: {inliers.sum()}/{len(inliers)}, '
              f'유효점 평균 오차: {errors[inliers].mean():.2f} px')

    def edit(self):
        self.roi_mode = False
        self.rois = [None] * 5
        self.roi_index = None
        self.roi_draft = []
        self.roi_side = None
        self.matrix = self.inverse = self.inliers = self.errors = None
        self.cursor = [None, None]
        self.pins.clear()

    def mark(self, image, side, point, label, color, outline=False):
        if point is None:
            return
        x, y = np.array(point) * self.scales[side] + self.offsets[side]
        h, w = image.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        p = (round(x), round(y))
        if outline:
            cv2.drawMarker(image, p, (0, 0, 0), cv2.MARKER_CROSS, 16, 4)
            cv2.putText(image, label, (p[0] + 7, max(15, p[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.drawMarker(image, p, color, cv2.MARKER_CROSS, 16, 2)
        cv2.putText(image, label, (p[0] + 7, max(15, p[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)

    def render(self, side):
        ox, oy = map(int, self.offsets[side])
        image = cv2.copyMakeBorder(self.views[side], oy, oy, ox, ox, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        image = self.draw_rois(image, side, self.scales[side], self.offsets[side])
        for i, point in enumerate(self.points[side]):
            color = (0, 220, 0) if self.inliers is None or self.inliers[i] else (0, 140, 255)
            self.mark(image, side, point, str(i + 1), color)
        if side == 0:
            self.mark(image, side, self.pending, '?', (0, 255, 255))
        for i, pair in enumerate(self.pins):
            self.mark(image, side, pair[side], f'P{i + 1}',
                      PIN_COLORS[i % len(PIN_COLORS)], outline=True)
        self.mark(image, side, self.cursor[side], '+', (0, 0, 255))
        if self.matrix is None:
            next_side = 1 if self.pending is not None else 0
            status = f'SELECT: {len(self.points[0])} pairs | Click Image {next_side + 1}'
            help_text = 'Enter/H: calculate | Right click: delete pair | Z: undo | Q: quit'
        else:
            status = f'PROJECT | inliers {self.inliers.sum()}/{len(self.inliers)} | move mouse / click to pin'
            if self.roi_index is not None and self.rois[self.roi_index] is not None:
                status += f' | Selected ROI {self.roi_index+1}'
            help_text = 'Right click: select ROI | D: delete | P: new | 1-5: edit | S: save'
        if self.roi_mode:
            status = f'ROI {self.roi_index+1}: {len(self.roi_draft)} vertices | Image {self.roi_side+1 if self.roi_side is not None else 1} or first click'
            help_text = 'Enter: finish | Z: undo | C: clear | D: delete ROI | P: cancel'
        coordinate = ''
        point = self.cursor[side]
        if self.matrix is not None and any(p is not None for p in self.cursor):
            if point is None:
                coordinate = 'Projection undefined (at infinity)'
            else:
                h, w = self.images[side].shape[:2]
                suffix = '' if 0 <= point[0] < w and 0 <= point[1] < h else ' [outside image]'
                coordinate = f'Original px: ({point[0]:.1f}, {point[1]:.1f}) | norm: ({point[0]/w:.4f}, {point[1]/h:.4f}){suffix}'
        footer = np.full((85, image.shape[1], 3), 28, np.uint8)
        for i, line in enumerate((status, help_text, coordinate)):
            cv2.putText(footer, line, (8, 21 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                        min(.48, image.shape[1] / 1500), (240, 240, 240), 1, cv2.LINE_AA)
        return np.vstack((image, footer))

    def save(self):
        # 매번 새 시간 폴더에 행렬/대응점/ROI와 정·역방향 합성 이미지를 보존한다.
        if self.matrix is None:
            raise ValueError('먼저 Enter 또는 H로 호모그래피를 계산하세요.')
        if self.roi_mode:
            raise ValueError('ROI를 Enter로 확정하거나 P로 취소한 뒤 저장하세요.')
        roi_data = [self.roi_payload(i) for i, roi in enumerate(self.rois) if roi is not None]
        if getattr(self, 'require_roi', False) and not roi_data:
            raise ValueError('분석용 ROI가 없습니다. P → 한 이미지에서 ROI 4점 이상 클릭 → Enter → S 순서로 저장하세요. 대응점과 ROI는 별도입니다.')
        pair_name = self.paths[0].name + '__' + self.paths[1].name
        folder = self.output / pair_name / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        folder.mkdir(parents=True, exist_ok=False)
        h, w = self.images[1].shape[:2]
        warped = cv2.warpPerspective(self.images[0], self.matrix, (w, h))
        overlay = cv2.addWeighted(warped, .5, self.images[1], .5, 0)
        source_h, source_w = self.images[0].shape[:2]
        warped_reverse = cv2.warpPerspective(self.images[1], self.inverse, (source_w, source_h))
        overlay_reverse = cv2.addWeighted(warped_reverse, .5, self.images[0], .5, 0)
        for name, image in [('warped.png', warped), ('overlay.png', overlay),
                            ('warped_reverse.png', warped_reverse),
                            ('overlay_reverse.png', overlay_reverse)]:
            ok, encoded = cv2.imencode('.png', image)
            if not ok:
                raise OSError(f'이미지 인코딩 실패: {name}')
            encoded.tofile(str(folder / name))
        if roi_data:
            (folder / 'roi.json').write_text(json.dumps(dict(version=2, rois=roi_data), indent=2, ensure_ascii=False), encoding='utf-8')
            for side in (0, 1):
                roi_image = self.draw_rois(self.images[side].copy(), side, np.ones(2))
                ok, encoded = cv2.imencode('.png', roi_image)
                if not ok:
                    raise OSError('ROI 이미지 인코딩 실패')
                encoded.tofile(str(folder / f'roi_image{side+1}.png'))
        payload = dict(rois=roi_data, version=1, coordinate_system='original_image_pixels',
                       direction='image1_to_image2', images=[str(p) for p in self.paths],
                       image_sizes=[[im.shape[1], im.shape[0]] for im in self.images],
                       points1=self.points[0], points2=self.points[1],
                       H=self.matrix.tolist(), H_inverse=self.inverse.tolist(),
                       ransac_threshold_px=self.threshold, inliers=self.inliers.tolist(),
                       reprojection_errors_px=self.errors.tolist(), projection_pairs=self.pins)
        (folder / 'homography.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        np.save(folder / 'H.npy', self.matrix)
        print(f'저장 완료: {folder.resolve()}')
        return folder

    def run(self):
        print('Image 1 클릭 → Image 2 대응점 클릭, 최소 4쌍 → Enter/H')
        print('계산 후 양쪽 이미지에서 마우스 이동: 실시간 투영, 클릭: 확인점 고정')
        print('S 저장 / E 수정 / Z 취소 / R 전체 초기화 / C 확인점 삭제 / Q 또는 Esc 종료')
        print('확정 ROI 내부 우클릭: 선택 → D 삭제 → S 저장. 겹친 ROI는 반복 우클릭으로 순환 선택.')
        print('P: 새 ROI (최대 5개), 숫자 1~5: ROI 수정 → 4점 이상 → Enter → S 저장. 편집 중 D: 삭제. E/R은 모든 ROI 초기화.')
        print('수정 모드에서 점 근처 우클릭: 해당 대응점 쌍 삭제 (15px 이내)')
        try:
            for side, name in enumerate(self.names):
                cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback(name, lambda e, x, y, f, p, side=side: self.mouse(side, e, x, y, f, p))
            while True:
                for side, name in enumerate(self.names):
                    cv2.imshow(name, self.render(side))
                key = cv2.waitKey(20) & 0xFF
                if key in (27, ord('q')) or any(cv2.getWindowProperty(n, cv2.WND_PROP_VISIBLE) < 1 for n in self.names):
                    break
                try:
                    if key == ord('p'):
                        if self.roi_mode:
                            self.roi_mode = False
                        else:
                            self.start_roi()
                    elif key in tuple(map(ord, '12345')) and not self.roi_mode:
                        self.start_roi(key - ord('1'))
                    elif key == ord('d'):
                        self.delete_roi()
                    elif self.roi_mode and key in (10,13):
                        self.finish_roi()
                    elif self.roi_mode and key == ord('z'):
                        if self.roi_draft:
                            self.roi_draft.pop()
                    elif self.roi_mode and key == ord('c'):
                        self.roi_draft = []
                        self.roi_side = None
                    elif key in (10, 13, ord('h')) and self.matrix is None:
                        self.calculate()
                    elif key == ord('s'):
                        self.save()
                    elif key == ord('e'):
                        self.edit()
                    elif key == ord('r'):
                        self.edit()
                        self.points = [[], []]
                        self.pending = None
                    elif key == ord('c'):
                        self.pins.clear()
                    elif key == ord('z'):
                        if self.matrix is not None:
                            if self.pins:
                                self.pins.pop()
                        elif self.pending is not None:
                            self.pending = None
                        elif self.points[0]:
                            self.points[0].pop()
                            self.points[1].pop()
                except (ValueError, OSError, cv2.error, np.linalg.LinAlgError) as exc:
                    print(f'오류: {exc}')
        finally:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image1', nargs='?', help='원본 이미지 경로')
    parser.add_argument('image2', nargs='?', help='대응 대상 이미지 경로')
    parser.add_argument('--load', metavar='JSON_OR_FOLDER', help='homography.json 또는 해당 파일이 있는 폴더')
    parser.add_argument('--output', default=str(Path(__file__).resolve().parents[1] / 'homography_results'))
    parser.add_argument('--require-roi', action='store_true', help='분석용 ROI가 지정된 경우에만 저장')
    parser.add_argument('--max-width', type=int, default=1000)
    parser.add_argument('--max-height', type=int, default=700)
    parser.add_argument('--ransac-threshold', type=float, default=5.0, help='대상 원본 이미지 기준 픽셀 오차')
    args = parser.parse_args()
    if min(args.max_width, args.max_height) < 1 or not np.isfinite(args.ransac_threshold) or args.ransac_threshold <= 0:
        parser.error('표시 크기와 RANSAC 임계값은 양수여야 합니다.')
    if bool(args.image1) != bool(args.image2):
        parser.error('이미지 경로는 두 개를 함께 입력하세요.')
    if not args.load and not args.image1:
        parser.error('두 이미지 경로 또는 --load homography.json을 지정하세요.')
    try:
        if args.load:
            override = [args.image1, args.image2] if args.image1 else None
            tool = HomographyTool.load(args.load, args.output, args.max_width, args.max_height, override)
        else:
            tool = HomographyTool([args.image1, args.image2], args.output, args.max_width,
                                  args.max_height, args.ransac_threshold)
        tool.require_roi = args.require_roi
        tool.run()
    except (OSError, ValueError, TypeError, cv2.error, np.linalg.LinAlgError) as exc:
        parser.exit(1, f'오류: {exc}\n')


if __name__ == '__main__':
    main()
