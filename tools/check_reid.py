import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mcmot.attributes import ReIDModel
model = ReIDModel(json.loads(Path(sys.argv[1]).read_text()))
values = model.embed([np.zeros((model.rows, model.cols, 3), dtype=np.uint8)])
if not np.isfinite(values).all():
    raise RuntimeError('ReID embedding이 유효하지 않습니다')
print(f'ReID 모델 검사 완료: {values.shape}', flush=True)
