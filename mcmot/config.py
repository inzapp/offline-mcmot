from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    base = path.parent
    for key in ("data_root", "output_root", "bev_image"):
        value = Path(cfg[key])
        if not value.is_absolute():
            cfg[key] = str((base / value).resolve())
    cfg["video_root"] = str(Path(cfg["video_root"]).expanduser())
    cfg["config_path"] = str(path)
    return cfg


def ensure_output_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg["output_root"])
    names = ["manifest", "local", "global", "attributes/crops", "quality",
             "report/assets", "visualization"]
    result = {"root": root}
    for name in names:
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        result[name] = target
    return result
