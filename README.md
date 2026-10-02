# Offline MCMOT analytics

통합 실행은 `pipeline.sh`를 사용합니다. 필요한 가상환경과 패키지는 자동 생성·설치됩니다.
새 데이터 구조·수동 보정·MiVOLO·단계별 실행 방법은 [통합 실행 안내](docs/pipeline.md)를 참고하세요.
새 설정의 출발점은 [config.pipeline.example.yaml](config.pipeline.example.yaml)입니다.

카메라별 pose detection CSV와 BEV 좌표, 원본 영상을 결합해 local MOT,
cross-camera global ID, 시청/주목 시간, 성별/연령, HTML 보고서와 검증 영상을
생성하는 재현 가능한 오프라인 파이프라인입니다.

## 기본 사용법

저장소 루트에서 예제 설정을 복사합니다.

```bash
cp config.pipeline.example.yaml config.section.yaml
```

`config.section.yaml`의 `data_path`, `dates`, `output_root`, MiVOLO·ReID 모델 경로를
실제 데이터에 맞게 수정합니다. `spatial` 구역은 아래 GUI에서 설정합니다. BEV 이미지와 카메라별 샘플 이미지를
준비하고, 기존 ROI/호모그래피를 복사하거나 보정 GUI에서 대응점을 지정합니다.

```bash
# 이미지·영상·Raw CSV·BEV CSV 수와 매칭 현황
./pipeline.sh status --config config.section.yaml

# 기존 보정이 없을 때: 카메라 ↔ BEV 대응점 4쌍 이상 지정
./pipeline.sh calibrate --config config.section.yaml

# BEV 위에서 동선·공간·시선 구역을 그리고 YAML에 저장
./pipeline.sh spatial --config config.section.yaml

# Raw 추론 → BEV 변환 → 검증 → tracking/ReID/MiVOLO → HTML → PPT
./pipeline.sh all --config config.section.yaml --deep
```

PPT 생성은 YAML의 `pptx.template`, `pptx.prompt`를 지정하고 Codex CLI 로그인을
준비해야 합니다. 설정하지 않으면 분석과 HTML 보고서까지 생성합니다.
처음 설치에는 Python 3.10과 인터넷이 필요하며, 영상 검증에는 `ffprobe`가 필요합니다.
영상 마운트와 모델 파일 경로도 접근 가능해야 합니다.

Raw·BEV 생성 결과는 영상 옆에, 분석 결과는 YAML의 `output_root`에 저장합니다.
기존 분석 출력이 있으면 번호가 붙은 새 출력 경로를 사용하며,
입력·설정이 같은 분석 결과를 재사용하려면 `--resume`을 추가합니다.
Raw·BEV는 완료 메타데이터가 일치하면 자동 재사용합니다.
기본 실행은 결과 시각화 영상 생성을 생략하며, 필요하면 `--with-video`를 추가합니다.

기존 TensorFlow ReID와 MiVOLO의 의존성 호환성 때문에 환경은 두 개입니다.
`.venv`는 분석·ReID, `.venv_ultralytics`는 YOLO Raw·MiVOLO를 함께 실행합니다.
수동으로 환경을 활성화할 필요는 없습니다.

데이터 폴더 구조, YAML 항목, 단계별 명령과 재실행 규칙은
[통합 실행 안내](docs/pipeline.md)에 정리했습니다.

## 주요 산출물

- `manifest/recordings.csv`: raw/bev/mp4 정합성 및 누락 정보
- `local/tracks.csv`: frame별 local ID와 관측/예측 여부
- `local/persons.csv`: local ID별 노출·시청·주목·대표 crop
- `global/id_mapping.csv`: local ID와 global ID 관계
- `global/tracks.csv`: frame별 local/global ID를 함께 담은 join-ready 결과
- `global/persons.csv`: 카메라 중복 제거 결과
- `spatial/person_events.csv`: global ID별 동선·공간·OOI 시선 판정 결과
- `spatial/demographics_summary.csv`: 날짜·1시간·동선/공간/시선·성별·연령별 인원
- `global/association_events.csv`: 위치·시간·ReID 판단 근거
- `attributes/inference_results.csv`: 성별/연령 모델 원시 출력
- `quality/issues.csv`: 처리 중 발견된 결손과 예외
- `report/index.html`: 상세 고객/검증 보고서
- `report/index_export.html`: 이미지 asset이 내장되어 단독 전송 가능한 보고서
- `visualization/best_matches.mp4`: 우수 cross-camera 매칭 사례 영상

GAMFF 동선 gate와 실외 우드부스 OOI polygon은 사이트별 YAML의 `spatial`에
BEV 정규화 좌표로 정의되어 있습니다. OOI 시선 ray가 추가된 결과를 만들려면
기존 `--resume` 결과가 아니라 새 `run`이 필요합니다.

기존 보고서를 단일 파일로 다시 export할 수도 있습니다.

```bash
.venv/bin/python -m mcmot.export_html \
  output/report/index.html output/report/index_export.html
```

성별·연령은 처음부터 MiVOLO 결과를 보고서에 반영합니다.
`attributes/results.csv`에 원시 추론 결과를 저장하고,
`attributes/statistics/`에는 crop 기준 처리 통계를 저장합니다.
Global ID 기준 방문객 집계는 `global/persons.csv`를 사용합니다.

포항 실내 설정은 [config.pohang_inside.yaml](config.pohang_inside.yaml), 포항 실외
설정은 [config.pohang_outside.yaml](config.pohang_outside.yaml)에 기록되어 있습니다.
새 `run` 또는 `inventory` 실행 시 설정의 output 경로가 이미 존재하면 `output2`,
`output3`처럼 번호가 붙은 새 경로를 자동으로 사용합니다. `--resume`, `report`,
`visualize`는 가장 최근 번호의 output을 선택합니다.
판정 순서와 누락 처리의 상세 내용은 [docs/design.md](docs/design.md)를 참고하세요.
