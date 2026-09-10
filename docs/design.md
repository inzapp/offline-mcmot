# 분석 설계와 판정 기준

## 처리 단위

ID namespace는 날짜와 녹화 파일마다 새로 시작한다. 따라서 정규 녹화 사이의
10초 term이나 비정상 전원 공백을 건너 local/global ID를 연결하지 않는다.
`2026-09-04`, `2026-09-05` 이외의 파일은 manifest와 분석에서 제외한다.

CSV 한 행의 identity는 `camera_id + recording_id + frame_index + detection_index`다.
raw와 BEV CSV는 timestamp와 frame index가 같고 같은 프레임 내 행 순서가 같을
때만 결합한다. 어긋나면 해당 녹화를 실패로 표시하며 임의 정렬·보간하지 않는다.

## Local MOT

상태 벡터 `[cx, cy, w, h, vx, vy, vw, vh]`의 constant-velocity Kalman filter를
사용한다. 1차 association은 confidence가 높은 bbox, 2차 association은 낮은
confidence bbox를 대상으로 한다. IoU와 Mahalanobis gating 후 Hungarian
assignment를 수행한다. 짧은 miss는 예측 상태로 ID를 유지하지만 실제 관측으로
간주하지 않는다.

- `track_span_seconds`: 허용된 짧은 예측 구간을 포함한 노출 시간
- `observed_seconds`: 실제 detection 기반 시간
- `predicted_gap_seconds`: Kalman-only 시간
- `is_predicted`: frame 결과에서 관측과 예측을 구분

Raw bbox가 없는 카메라는 BEV point tracker만 수행한다. 이 경우 시청, 주목,
crop, 성별, 연령, ReID는 null이고 품질 플래그가 붙는다.

## 시청과 주목

`adddi_light/app_main/src/tracker_wrapper.cpp`의 `tw_is_watch_condition`과
`tw_accumulate_watched_time`을 이식했다. nose/eyes/ears confidence, eye-to-ear
비율과 수직 각도를 사용한다. 작은 bbox에는 더 좁은 55도 조건을 적용한다.

요청에 따라 시청 threshold는 1초, 주목 threshold는 3초다. C++와 동일하게
threshold 이전 시간을 소급 합산하지 않는다. 예컨대 4초 연속 응시는 샘플
간격에 따른 경계 오차를 제외하면 시청 약 3초, 주목 약 1초로 적립된다.

## Crop과 속성

실제 관측 bbox만 crop한다. 최소 크기는 2×2 pixel이다. 일정 detection 주기마다
정면 응시 여부를 먼저, 다음으로 crop 면적을 비교해 대표 이미지를 갱신한다.
Global ID에서는 연결된 local crop 중 같은 기준으로 최종 대표 crop을 고른다.

- gender: `mGtEx-sba-001.onnx`, 64×64 RGB NCHW, `/255`, score 0.42
- age: `5age_cls_128x64_simclf.onnx`, 128×64 RGB NCHW, `/255`, raw argmax
- ReID: `reid_284/best_30000_iter_rank1_0.7643.h5`, 128×64 RGB NHWC, `/255`

연령의 5번째 class는 `unknown`으로 그대로 둔다. 원본 C++의 random fallback은
요청한 “임의 생성 금지” 원칙 때문에 사용하지 않는다.

## Global association과 topology

카메라별 track 시작/종료점에서 BEV ROI 경계까지의 거리를 구한다. 각 카메라의
거리 분포 하위 quantile을 exit/entry 범위로 사용한다. cross-camera 후보는
다음 hard gate를 모두 통과해야 한다.

1. 날짜와 녹화 슬롯이 같아 10초 term을 건너지 않음
2. 동시 관측이면 BEV 거리가 허용 범위 이내
3. handoff이면 exit/entry 경계 범위, 시간 창, 속도 기반 도달 가능성 충족
4. ReID L2 distance < 0.9358519

통과 후보는 ReID, BEV 거리, 시간의 가중 score로 정렬해 병합한다. 동일 카메라에
동시에 존재하는 두 local track이 하나의 global ID가 되는 병합은 금지한다.
Global 시간 지표는 카메라별 합이 아닌 시간 interval union으로 계산한다.

보고서에는 승인과 거절을 포함한 association 감사 로그, 데이터 누락, topology,
설정값을 보존한다. 시각화 영상은 score가 가장 좋은 승인 사례를 선택해 crop,
BEV 연결, embedding 거리와 threshold를 동시에 보여준다.
