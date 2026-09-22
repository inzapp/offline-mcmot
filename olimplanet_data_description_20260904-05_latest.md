# 올림플래닛 제공용 Pohang raw 데이터 설명서

- 패키지: `olimplanet_raw_json_20260904-05_latest`
- 기준 데이터: 2026-09-04 ~ 2026-09-05, KST
- 형식: UTF-8 JSON

## 1. 산출물 구성

- raw 패키지: `olimplanet_raw_json_20260904-05_latest`
- JSON은 사이트별 `combined/all_cameras_raw.json`과 `by_camera/camera_*_raw.json`으로 구성됩니다.
- 성별·연령은 track별 개선 모델 결과를 연결하며, 연결 결과가 없는 track은 `null`로 표시합니다.

## 2. 적용한 원천 및 최신 버전

| 사이트 | 데이터 기간 | 카메라 | 추적 ID | 1초 path point | 성별·연령 매칭 |
|---|---|---|---:|---:|---:|
| `pohang_inside` | 2026-09-04, 2026-09-05 | 3417, 3419, 3420, 3421 | 17,887 | 436,679 | 17,869/17,887 |
| `pohang_outside` | 2026-09-04, 2026-09-05 | 3414, 3415, 3416 | 15,125 | 733,954 | 15,123/15,125 |

각 사이트에서 번호 기준으로 가장 최신인 처리 결과를 사용했으며, 두 사이트 모두 동일한 기간의 데이터를 포함합니다.

샘플과 동일한 `tracks`·`measure`·`path` 구조를 유지하고, 두 날짜를 한 파일에 담기 위해 JSON 머리말의 `data_dates`를 배열로 표기했습니다.

## 3. JSON 구조

각 `tracks` 항목은 한 카메라 녹화 안에서 관측된 한 사람의 추적 구간입니다. `person_id`는 서로 다른 녹화 구간에서도 구분할 수 있도록 구성한 익명 추적 ID입니다.

| 필드 | 설명 |
|---|---|
| `camera_number` | 카메라 번호 |
| `person_id` | 날짜·카메라·녹화 시작 시각·순번으로 구성한 익명 추적 ID |
| `measure.exposed_time` | 화면에 머문 추적 시간(초, 예측 구간 포함) |
| `measure.watched_time` | 시청 판정 누적 시간(초) |
| `measure.attention_time` | 주목 판정 누적 시간(초) |
| `measure.gender` | 개선 모델 성별 결과(`male`/`female`), 결과가 없으면 `null` |
| `measure.age` | 개선 모델 추정 연령(세), 결과가 없으면 `null` |
| `measure.appear_timing` | 최초 등장 시각(KST) |
| `path` | 관측 row를 1초 단위로 줄인 시간순 좌표 |

`path.p`는 카메라 화면 정규화 bbox 중심 하단 좌표입니다. `z`와 `b`는 원천 값이 없어 각각 `0`, `-1` sentinel을 사용했습니다. `g`와 `e`는 저장된 카메라 gaze ray에서 변환한 값이며, 유효하지 않으면 `-1`/`[-1,-1]`입니다.

## 4. 성별·연령 데이터 전달 원칙

각 track의 `measure.gender`와 `measure.age`는 개선 모델의 track별 결과를 `person_id`에 연결한 값입니다. 결과 파일에서 해당 track을 찾을 수 없거나 값이 비어 있는 경우에는 해당 필드를 `null`로 표시합니다.

## 5. 검증 방법

- `manifest.json`에 패키지 파일별 byte 수와 SHA-256을 기록했습니다.
- JSON track 수·path point 수·카메라별 집계는 manifest의 counts와 일치해야 합니다.
- 성별·연령 매칭 건수는 manifest의 attribute matching 집계와 일치해야 합니다.

세부 필드 정의는 패키지의 `schema.json`, 파일 목록과 검증 집계는 `manifest.json`을 참고하십시오.
