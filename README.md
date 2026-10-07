# Offline MCMOT analytics

통합 실행은 `pipeline.sh`를 사용합니다. 필요한 가상환경과 패키지는 자동 생성·설치됩니다.
새 데이터 구조·수동 보정·MiVOLO·단계별 실행 방법은 [통합 실행 안내](docs/pipeline.md)를 참고하세요.
새 설정의 출발점은 [config.pipeline.example.yaml](config.pipeline.example.yaml)입니다.

카메라별 pose detection CSV와 BEV 좌표, 원본 영상을 결합해 local MOT,
cross-camera global ID, 시청/주목 시간, 성별/연령, HTML 보고서와 검증 영상을
생성하는 재현 가능한 오프라인 파이프라인입니다.

## 입력 데이터 구조

권장 구조는 아래와 같습니다. 예제 설정의 `data_path: data/section_example`은
사이트 데이터 폴더를 가리키며, 상대경로는 YAML 파일 위치를 기준으로 해석합니다.
카메라 ID와 날짜는 실제 데이터에 맞게 바꾸고, 분석할 날짜를 YAML의 `dates`에 지정하세요.

```text
data/
└── section_example/                         # YAML의 data_path
    ├── bev.png                              # 필수: BEV 이미지 (JPG도 가능); 직접 준비, CLI 생성 명령 없음
    ├── roi.json                             # 선택: 통합 ROI; roi_file 지정 시 pipeline calibrate로 저장
    ├── 3417/                                # 카메라 ID
    │   ├── 3417.jpg                         # 필수: 샘플 이미지 (PNG도 가능); 직접 준비, CLI 생성 명령 없음
    │   ├── 3417_roi.json                    # 통합 ROI 미사용 시 필요; pipeline calibrate로 생성
    │   ├── calibration/
    │   │   └── homography.json              # 보정 행렬; pipeline calibrate로 생성
    │   └── 2026-09-03/                      # 녹화 날짜 (YYYY-MM-DD)
    │       ├── 2026-09-03_DEVICE_100000.mp4   # 원본 영상; 직접 준비, CLI 생성 명령 없음
    │       ├── 2026-09-03_DEVICE_100000.csv   # detection/pose CSV; pipeline raw 또는 all로 생성
    │       └── 2026-09-03_DEVICE_100000_bev.csv # BEV 좌표 CSV; pipeline bev 또는 all로 생성
    └── 3419/                                # 다른 카메라도 같은 구조
        ├── 3419.jpg
        ├── 3419_roi.json
        ├── calibration/
        │   └── homography.json
        └── 2026-09-03/
            ├── 2026-09-03_DEVICE_100000.mp4
            ├── 2026-09-03_DEVICE_100000.csv
            └── 2026-09-03_DEVICE_100000_bev.csv
```

트리 주석의 `pipeline <명령>`은 `python -m mcmot.pipeline <명령> --config config.section.yaml`을
뜻하며, 가상환경을 자동으로 선택하는 `./pipeline.sh <명령> --config config.section.yaml`로
실행할 수 있습니다. `mcmot.cli`에는 입력 이미지·영상·ROI·Raw/BEV CSV를 생성하는 명령이
없습니다. `python -m mcmot.cli run --config config.section.yaml`은 준비된 입력을 사용해
분석 결과를 생성하고, `inventory`는 `output_root/manifest/recordings.csv`를 생성합니다.
`validate --deep`은 입력을 검증하며, `report`와 `visualize`는 기존 분석 결과로
HTML 보고서와 검증 영상을 생성합니다.

처음에는 BEV 이미지, 카메라별 샘플 이미지, 원본 영상을 준비합니다.
기존 ROI/보정 파일이 없으면 `calibrate`로 ROI와 보정 행렬을 생성합니다.
Raw·BEV CSV는 `all` 실행 시 자동 생성되므로 미리 준비할 필요가 없으며,
기존 CSV가 있다면 아래 형식에 맞춰 재사용할 수 있습니다.

- 영상 파일명은 `YYYY-MM-DD_DEVICE_HHMMSS.mp4`입니다. `DEVICE`는 장치 식별자,
  `HHMMSS`는 녹화 시작 시각이며, 같은 녹화의 Raw CSV는 영상과 같은 이름,
  BEV CSV는 이름 끝에 `_bev`를 붙입니다.
- Raw CSV는 검출된 사람당 한 행이며 `timestamp,frame_index,confidence,cx,cy,w,h`와
  COCO 17개 keypoint 각각의 `<이름>_conf`, `<이름>_x`, `<이름>_y` 컬럼을 포함합니다.
  `cx,cy,w,h`와 keypoint 좌표는 카메라 이미지 기준 정규화 좌표입니다.
  `timestamp`는 `YYYY-MM-DD HH:MM:SS.sss`, `frame_index`는 0부터 시작합니다.
- BEV CSV의 헤더는 `timestamp,frame_index,x,y`입니다. `x,y`는 bbox 하단 중앙을
  BEV로 투영한 정규화 좌표이며, Raw CSV와 행 수·행 순서·시각·프레임이 일치해야 합니다.
- 통합 ROI는 `{"3417":{"rois":[...]},"3419":{"rois":[...]}}` 형식입니다.
  카메라별 ROI는 `{"rois":[...]}` 형식이며, 각 ROI에
  `image1_vertices_normalized`(BEV)와 `image2_vertices_normalized`(카메라)를 지정합니다.
  별도 보정 행렬이 없으면 두 이미지의 대응점이 최소 4쌍 필요합니다.

기존 `data_path/<카메라 ID>/raw/<날짜>/*.csv` 및
`data_path/<카메라 ID>/bev/<날짜>/*.csv` 분리 구조도 지원합니다.
영상을 별도로 보관하면 YAML의 `video_root` 아래에 `<카메라 ID>/<날짜>/*.mp4`를
배치합니다. `video_root`를 생략하면 `data_path`를 사용합니다.
모델 가중치는 YAML의 MiVOLO·ReID 경로에 별도로 준비하며, 분석 결과는 `output_root`에 저장됩니다.
입력 파일 매칭은 `status`, CSV 내용까지 검증하려면 `validate --deep`을 사용하세요.
보정과 전체 실행 절차는 [통합 실행 안내](docs/pipeline.md)를 참고하세요.

## 기본 사용법

저장소 루트에서 필요한 두 가상환경을 설치합니다. MiVOLO를 기본으로 포함하며 설정 파일이나 데이터 없이 실행할 수 있습니다.

```bash
./pipeline.sh setup
```

이후 예제 설정을 복사합니다.

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
