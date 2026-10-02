import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mcmot.mivolo_backend import SnapshotEstimator
options = json.loads(Path(sys.argv[1]).read_text())
allowed = ('device', 'threads', 'face_conf', 'min_face_size', 'body_only', 'detector_size', 'model_dir', 'repository')
selected = {k: options[k] for k in allowed if k in options}
if options.get('detector'):
    selected['detector_path'] = options['detector']
estimator = SnapshotEstimator(**selected)
result, _ = estimator.predict(np.zeros((256, 128, 3), dtype=np.uint8))
print(f'MiVOLO 모델 검사 완료: {result["input_mode"]} / {estimator.device}')
