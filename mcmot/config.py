from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    base = path.parent
    for key in ("data_root", "data_path", "video_root", "output_root", "bev_image", "roi_file"):
        if key not in cfg:
            continue
        value = Path(cfg[key]).expanduser()
        if not value.is_absolute():
            cfg[key] = str((base / value).resolve())
    if cfg.get("data_path"):
        cfg.setdefault("site", Path(cfg["data_path"]).name)
        cfg.setdefault("video_root", cfg["data_path"])
        cfg.setdefault("data_root", str(Path(cfg["data_path"]).parent))
        root = Path(cfg["data_path"])
        images = [p for p in root.glob("bev.*") if p.suffix.lower() in (".png", ".jpg")]
        if "bev_image" not in cfg and len(images) == 1:
            cfg["bev_image"] = str(images[0])
        if "roi_file" not in cfg and (root / "roi.json").is_file():
            cfg["roi_file"] = str(root / "roi.json")
    for section, keys in (("attributes", ("model_dir", "detector", "repository", "python", "gender_model", "age_model")),
                          ("reid", ("repository", "config", "model")),
                          ("extraction", ("det_model", "pose_model")),
                          ("pptx", ("template", "prompt", "output"))):
        for key in keys:
            if cfg.get(section, {}).get(key):
                value = Path(cfg[section][key]).expanduser()
                if section == "extraction" and not value.is_absolute() and len(value.parts) == 1:
                    continue  # Keep model names stable before and after automatic download.
                cfg[section][key] = str(value if value.is_absolute() else (base / value).resolve())
    cfg["config_path"] = str(path)
    return cfg


def ensure_output_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg["output_root"])
    names = ["manifest", "local", "global", "spatial", "attributes/crops", "quality",
             "report/assets", "visualization"]
    result = {"root": root}
    for name in names:
        target = root / name
        target.mkdir(parents=True, exist_ok=True)
        result[name] = target
    return result


def select_output_root(cfg: dict[str, Any], fresh: bool) -> dict[str, Any]:
    """Choose a non-overwriting output for a new run, or the latest for reuse."""
    selected = dict(cfg)
    base = Path(cfg["output_root"])
    candidates: list[tuple[int, Path]] = []
    if base.exists():
        candidates.append((1, base))
    for path in base.parent.glob(f"{base.name}[0-9]*"):
        suffix = path.name[len(base.name):]
        if suffix.isdigit() and int(suffix) >= 2:
            candidates.append((int(suffix), path))
    if fresh:
        number = max((number for number, _ in candidates), default=0) + 1
        selected["output_root"] = str(base if number == 1 else base.with_name(f"{base.name}{number}"))
    elif candidates:
        selected["output_root"] = str(max(candidates, key=lambda item: item[0])[1])
    return selected
