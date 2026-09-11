# Offline MCMOT analytics

카메라별 pose detection CSV와 BEV 좌표, 원본 영상을 결합해 local MOT,
cross-camera global ID, 시청/주목 시간, 성별/연령, HTML 보고서와 검증 영상을
생성하는 재현 가능한 오프라인 파이프라인입니다.

## 실행 환경

이 장비에서는 TensorFlow, ONNX Runtime, OpenCV가 함께 설치된 `tf` pyenv를
사용합니다.

```bash
PYENV_VERSION=tf python -m mcmot.cli inventory --config config.pohang_inside.yaml
PYENV_VERSION=tf python -m mcmot.cli run --config config.pohang_inside.yaml
```

원본 영상 마운트가 접근 가능한지 먼저 확인해야 합니다. 영상이 없거나 마운트가
끊기면 tracking은 가능한 입력으로 계속되지만 crop, ReID, 성별/연령과 매칭
사례 영상은 생성할 수 없습니다.

빠른 구조 검증은 영상 추론 없이 실행할 수 있습니다.

```bash
PYENV_VERSION=tf python -m unittest discover -s tests -v
PYENV_VERSION=tf python -m mcmot.cli inventory --config config.pohang_inside.yaml
```

전체 실행 결과는 기본적으로 `output/` 아래에 저장됩니다. 중간 단계 CSV가
이미 있으면 `--resume`으로 재사용할 수 있습니다. Parquet은 `pyarrow`가 설치된
경우 CSV와 함께 생성되고, 없으면 CSV만 생성됩니다.

## 주요 산출물

- `manifest/recordings.csv`: raw/bev/mp4 정합성 및 누락 정보
- `local/tracks.csv`: frame별 local ID와 관측/예측 여부
- `local/persons.csv`: local ID별 노출·시청·주목·대표 crop
- `global/id_mapping.csv`: local ID와 global ID 관계
- `global/tracks.csv`: frame별 local/global ID를 함께 담은 join-ready 결과
- `global/persons.csv`: 카메라 중복 제거 결과
- `global/association_events.csv`: 위치·시간·ReID 판단 근거
- `attributes/inference_results.csv`: 성별/연령 모델 원시 출력
- `quality/issues.csv`: 처리 중 발견된 결손과 예외
- `report/index.html`: 상세 고객/검증 보고서
- `report/index_export.html`: 이미지 asset이 내장되어 단독 전송 가능한 보고서
- `visualization/best_matches.mp4`: 우수 cross-camera 매칭 사례 영상

기존 보고서를 단일 파일로 다시 export할 수도 있습니다.

```bash
PYENV_VERSION=tf python -m mcmot.export_html \
  output/report/index.html output/report/index_export.html
```

포항 실내 설정은 [config.pohang_inside.yaml](config.pohang_inside.yaml), 포항 실외
설정은 [config.pohang_outside.yaml](config.pohang_outside.yaml)에 기록되어 있습니다.
새 `run` 또는 `inventory` 실행 시 설정의 output 경로가 이미 존재하면 `output2`,
`output3`처럼 번호가 붙은 새 경로를 자동으로 사용합니다. `--resume`, `report`,
`visualize`는 가장 최근 번호의 output을 선택합니다.
판정 순서와 누락 처리의 상세 내용은 [docs/design.md](docs/design.md)를 참고하세요.
