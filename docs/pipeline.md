# 통합 실행

저장소 루트에서 `pipeline.sh`를 실행합니다. 환경을 수동으로 활성화하지 않아도 됩니다.
첫 실행은 `.venv`를 생성하고 입력 현황을 표시하는 데 필요한 패키지를 설치합니다.
추론/분석 패키지는 해당 단계에서 설치합니다. 최초 설치에는 인터넷과 Python 3.10이 필요합니다.
다른 Python이 기본값이면 `PIPELINE_PYTHON=/경로/python3.10 ./pipeline.sh ...`로 지정하세요.

## 설정과 데이터

`config.pipeline.example.yaml`을 복사하여 `data_path`, `dates`, `output_root`,
MiVOLO/ReID 모델 경로를 실제 값으로 지정합니다. `spatial` 분석 구역은 GUI에서 지정할 수 있습니다.
기존 사이트 설정도 읽을 수 있고, 기존 사이트 설정의 성별/연령 backend는 MiVOLO로 변경했습니다.
모델 가중치는 YAML에 명시된 경로에서 읽으며 모델 파일 자체는 복사하지 않습니다.
MiVOLO Python 소스와 Homography GUI는 이 저장소로 통합했습니다.

제공된 YAML의 외부 데이터·모델·PPT 경로는 회사 PC 로컬 `/mnt/nvme1n1/` 기준입니다.
MiVOLO 가중치는 `/mnt/nvme1n1/Olim/Scripts/mivolvo/weights/mivolo_v2/`,
ReID는 `/mnt/nvme1n1/offline-mcmot/models/reid/`, 샘플 PPT는
`/mnt/nvme1n1/offline-mcmot/sample_output_pptx/`를 사용합니다.
`data/`, `vendor/mivolo`, `prompts/`, 출력 폴더 등 상대경로는 YAML 파일 위치를
기준으로 해석하므로 저장소를 회사 PC의 다른 경로에 복사해도 유지할 수 있습니다.
YOLO 모델은 `yolov8m.pt`, `yolov8m-pose.pt` 이름으로 지정할 수 있으며,
실행 경로에 파일이 없으면 Ultralytics가 최초 실행 시 다운로드합니다.

```text
data_path/
  bev.png                         # bev.jpg 또는 bev.png 중 하나
  roi.json                        # 선택: 카메라 ID별 ROI를 합친 JSON
  3417/
    3417.jpg                      # 카메라 샘플 JPG/PNG 하나
    3417_roi.json                 # 통합 roi.json이 없을 때 카메라별 ROI
    calibration/homography.json   # GUI 보정 후 자동 등록
    2026-09-03/
      2026-09-03_DEVICE_100000.mp4
      2026-09-03_DEVICE_100000.csv
      2026-09-03_DEVICE_100000_bev.csv
```

기존 `camera/raw/date/*.csv`, `camera/bev/date/*.csv` 구조도 지원합니다.
BEV 옆 통합 `roi.json`은 `{"3417":{"rois":[...]},"3419":...}` 형식이며,
각 ROI에는 `image1_vertices_normalized`(BEV), `image2_vertices_normalized`(카메라)가 필요합니다.
기존의 카메라별 단일 ROI 객체 형식도 읽습니다.
동선/공간/시선 구역은 YAML의 `spatial.routes`, `spatial.regions`, `spatial.gaze.targets`에 확정합니다.
`spatial` GUI에서 BEV에 직접 그릴 수 있습니다. GUI 보정과 구역 작업은 데스크톱/X11 연결이 필요합니다.

## 명령

```bash
# 모든 실행은 먼저 BEV·카메라 이미지 경로와 파일 매칭 표를 표시합니다.
./pipeline.sh status --config config.section.yaml

# 카메라별 대응점 4쌍 이상과 유효 ROI를 지정하고 S로 저장합니다.
./pipeline.sh calibrate --config config.section.yaml
./pipeline.sh calibrate --config config.section.yaml --camera 3417

# BEV 위에서 동선·공간·시선 분석 구역 설정
./pipeline.sh spatial --config config.section.yaml

# 수동 준비가 끝나면 raw → BEV → 검증 → 분석 → (설정된 경우) PPT
./pipeline.sh all --config config.section.yaml --deep

# 필요한 단계만 실행
./pipeline.sh raw --config config.section.yaml
./pipeline.sh bev --config config.section.yaml
./pipeline.sh validate --config config.section.yaml --deep
./pipeline.sh validate --config config.section.yaml --deep --check-models
./pipeline.sh run --config config.section.yaml --deep
./pipeline.sh pptx --config config.section.yaml
```

보정 GUI에서는 대응점 지정 후 Enter/H로 행렬을 계산하고, **P → 한 이미지에서
ROI 꼭짓점 4개 이상 클릭 → Enter로 확정 → S로 저장 → Q로 종료**합니다.
대응점과 ROI는 별도이며 `calibrate`는 ROI가 없는 저장을 차단합니다.
GUI 종료 후 호모그래피와 ROI를 실행 경로에 자동 등록하므로 수동 복사는 필요하지 않습니다.
등록되지 않은 이전 저장본이 있으면 같은 이미지 쌍의 마지막 저장본을 불러와 작업을
이어갈 수 있습니다. `--camera`가 없으면 카메라별로 순차 진행합니다.

### 동선·공간·시선 GUI (`spatial`)

기존 YAML의 구역을 불러옵니다. 유형을 선택하고 **추가 → 이름 입력 → 그릴 구역 선택 →
그리기 → BEV 이미지에서 꼭짓점 3개 이상 클릭 → 확정** 순서로 지정합니다.
동선은 시작·종료 구역과 양방향 여부, 공간은 하나의 구역, 시선은 출발·대상 구역을 지정합니다.
시선의 출발 구역은 사람이 위치하는 곳이며 대상 구역은 바라보는 물체가 있는 곳입니다.
선택한 구역은 강조 표시되고 동선·시선의 방향은 화살표로 표시됩니다.

목록에서 기존 항목을 선택해 이름을 수정하거나 폴리곤을 다시 그릴 수 있습니다.
`점 취소` 또는 마우스 오른쪽 버튼으로 마지막 점을 취소하고, `그리기 취소` 또는 Esc로
그리기를 취소합니다. 취소하면 기존 폴리곤이 유지됩니다. `삭제`는 선택한 항목을 지웁니다.
목록의 ↑/↓는 검사 순서를 변경하며, 시선 대상이 겹칠 때 필요한 순서를 유지할 수 있습니다.

**저장 후 종료**하면 좌표가 BEV의 0~1 정규화 좌표로 같은 YAML의 `spatial`에 저장됩니다.
미완성 구역, 중복 이름, 면적이 없는 구역, 선이 교차하는 구역은 저장을 차단합니다.
다른 YAML 설정값과 상대경로, 시선 판정 파라미터는 유지합니다. YAML은 다시 직렬화하므로
기존 주석·서식은 유지되지 않으며, 최초 저장 전 원본을 `<config>.spatial.bak`에 보관합니다.
GUI를 열어 둔 동안 외부에서 YAML을 변경하면 덮어쓰기를 차단합니다.

저장 후 `all`을 실행하면 구역이 tracking·시선 판정·통계·보고서에 반영됩니다.
`all`이 GUI를 자동으로 열지는 않습니다. 구역을 변경한 후에는 `--resume` 없이 새 분석을 실행하세요.
구역 편집은 BEV 이미지와 YAML만 있으면 가능하며, 카메라 보정 ROI와 별도로 저장됩니다.
GUI는 Python 표준 라이브러리 `tkinter`를 사용합니다. Tk 지원이 없는 Python에서는
시스템의 Tk 패키지와 Tk 지원 Python을 준비해야 합니다.

`status`는 영문 트리로 BEV 이미지, 카메라별 이미지·ROI·보정 정보와 날짜별
MP4·Raw·BEV 파일을 표시합니다. 존재하는 파일은 초록색 `✓`, 누락은 빨간색 `✗`로
표시합니다. JPG/PNG 이미지가 중복되거나 선택된 날짜에 녹화가 없는 경우도 `✗`로
표시합니다. 저장된 homography가 없으면 ROI 대응점을 보정 소스로 표시합니다.
색상은 터미널 출력에서만 적용하며, 리다이렉션·로그 출력은 색상 코드 없이 기호를
유지합니다. `NO_COLOR=1 ./pipeline.sh status --config config.section.yaml`로 색상을 끌 수 있습니다.

출력 하단의 MP4/RAW/BEV는 선택된 날짜의 파일 수이며, MP4↔RAW/MP4↔BEV/Complete는
카메라와 녹화 파일명이 같은 파일들의 개수입니다. 없는 영상, CSV만 있는 녹화도 표에 반영합니다.
파일명 정합성과 CSV 내부 정합성은 구분합니다. `validate --deep`는 ffprobe로 영상 메타데이터를
확인하고 raw/BEV의 행 수, 행별 시각/프레임, 좌표 유효성을 검사합니다.
ffprobe는 시스템에 설치되어 있어야 합니다. 1,790초와 다른 영상 길이는 경고로 표시합니다.
입력 오류와 local tracking 오류는 분석을 중단합니다.

## 추론 및 가상환경

기존 TensorFlow 2.12는 NumPy `<1.24`를 요구하고 참조 MiVOLO 환경은 NumPy `1.26.4`를 사용합니다.
기존 H5 ReID 모델의 실행 환경을 유지하기 위해 두 환경을 사용합니다.

- `.venv`: NumPy 1.23.5 / TensorFlow 2.12 / 보정·BEV·분석·ReID·PPT 보조 코드
- `.venv_ultralytics`: raw detection/pose와 MiVOLO를 함께 실행

raw만 사용하면 pip 업그레이드 후 `pip install ultralytics`로 설치합니다.
MiVOLO를 사용하면 같은 환경에 참조 패키지 버전과 MiVOLO 소스를 설치하며,
공유 Ultralytics 버전은 참조 환경의 8.1.0으로 고정합니다. 별도 requirements_ultralytics.txt는 없습니다.
torch/torchvision은 참조 서버의 2.7.1/0.22.1 CUDA 12.8 조합이 기본값입니다.
CPU 환경은 YAML attributes의 `torch_index_url: https://download.pytorch.org/whl/cpu`, `device: cpu`로 지정할 수 있습니다.
설치 후 `pip check`를 수행하고 설치된 버전을 각 환경의 `installed-packages.txt`에 저장합니다.

raw의 기본값은 detection `yolov8m.pt`, 896, conf >0.05 / pose `yolov8m-pose.pt`, 1152, conf >0.02입니다.
사람 클래스만 추출하며 pose를 우선하는 IoU 0.45 NMS로 병합합니다.
Detection만 있는 행은 keypoint 좌표와 confidence를 0으로 저장합니다.
매 프레임 처리하며 파일명 시작 시각과 실제 영상 FPS로 timestamp를 계산합니다.
다음 영상은 파일명 시각에서 시작하므로 10초 녹화 공백이 유지됩니다.

```bash
./pipeline.sh raw --config config.section.yaml \
  --det-model yolov8m.pt --det-imgsz 896 --det-conf 0.05 \
  --pose-model yolov8m-pose.pt --pose-imgsz 1152 --pose-conf 0.02 \
  --merge-iou 0.45 --model-iou 0.7 --max-det 10000 --device 0
```

CLI 인자가 YAML extraction 설정보다 우선합니다. FPS가 잘못 기록된 영상은 `--timestamp-fps`로 지정합니다.
직접 실행은 `tools/extract_video_raw.py --help`와 `tools/project_raw_to_bev.py --help`를 참고하세요.

## BEV 형식과 재실행

BEV CSV는 기존 데이터와 동일한 `timestamp,frame_index,x,y` 컬럼입니다.
`x,y`는 bbox 하단 중앙을 카메라 → BEV로 투영한 정규화 좌표입니다.
ROI 밖이나 0~1 범위 밖 좌표도 보존하며 raw 행을 삭제하거나 정렬하지 않습니다.
보정 행렬은 image1=BEV / image2=카메라의 GUI 저장 행렬을 역변환하고 정규화합니다.
별도 행렬이 없으면 기존 ROI의 양쪽 대응점으로 행렬을 계산합니다.

기존 외부 생성 CSV는 기본적으로 유지합니다. 변경된 모델이나 보정으로 재생성하려면
해당 단계에 `--force`를 지정합니다. raw/BEV 생성은 `.partial`로 기록한 후 완료 파일로 교체합니다.
생성 파일 옆 `.csv.meta.json`에 입력 파일·파라미터·완료 상태를 기록하며,
메타데이터가 일치하면 재실행 시 재사용합니다. 실패한 raw 영상은 처음부터 재처리합니다.
검출이 없는 마지막 프레임도 완료 여부에 반영됩니다.

분석 결과는 기존 정책대로 `output`, `output2`, ...에 저장합니다.
`--resume`은 마지막 분석 출력과 입력·설정이 일치할 때만 허용합니다.
입력이나 설정이 바뀌면 `--resume`을 빼고 새 분석 결과를 생성하세요.
`--with-video`로 결과 시각화 영상도 생성할 수 있습니다. 기본값은 `--skip-video` 동작입니다.
원본 영상은 기본 실행에서도 crop/ReID에 사용됩니다.

## MiVOLO 결과와 PPT

MiVOLO는 local tracking에서 생성된 대표 crop에 대해 모델을 한 번 로드해 추론합니다.
연속형 나이는 `age_estimate`, 연령 구간은 `age`로 분리하며 실패는 unknown으로 반영합니다.
MiVOLO 실패 여부와 ReID embedding 저장은 분리합니다.

- `attributes/results.csv`: local_uid를 포함한 MiVOLO 원시 추론 결과
- `attributes/inference_results.csv`: MiVOLO 결과와 ReID 경로를 결합한 공통 결과
- `attributes/statistics/`: crop 기준 성별·연령·처리시간·오류 통계
- `global/persons.csv`, `spatial/demographics_summary.csv`: Global ID 대표 crop 결과를 사용한 집계

HTML은 처음부터 새 모델 결과를 사용하며 `--use-statistics`로 다시 수정할 필요가 없습니다.
crop 처리 통계와 중복 제거한 방문객 집계의 분모를 구분합니다.

PPT를 포함하려면 다음 설정을 추가합니다. Codex CLI 설치와 로그인이 사전에 필요합니다.

```yaml
pptx:
  template: /실제/샘플.pptx
  prompt: prompts/report_pptx.md
  # output 생략 시 이번 분석 output/report/<site>.pptx
  timeout_seconds: 1800
```

준비된 프롬프트와 분석 자료 경로를 `codex exec`에 전달하며 템플릿 복사본으로 작업합니다.
작업 로그는 `report/pptx_job/`에 저장하고 실제 PPTX ZIP 형식을 확인한 뒤 출력합니다.
콘솔에는 PPT 생성 시작·출력 경로, Codex 작업 메시지·명령 실행 상황과 경과 시간을 표시합니다.
새 메시지가 없는 동안에도 30초마다 `Still generating...`과 마지막 작업을 출력합니다.
완료 시 PPTX 파일 검증과 최종 저장 경로를 표시하며, 전체 이벤트와 오류 내용은
`events.jsonl`, `stderr.log`에 보존합니다.
템플릿이 지정되지 않은 all 실행은 분석까지 완료하고 PPT 설정 대기를 표시합니다.
동일한 시각적 형식의 재현은 실제 템플릿으로 검증해야 합니다.
