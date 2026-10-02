#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcmot.config import load_config
from mcmot.inventory import build_manifest
from mcmot.raw import Extractor, DEFAULTS, add_arguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--force', action='store_true')
    add_arguments(parser); args = parser.parse_args()
    cfg = load_config(args.config)
    options = {**cfg.get('extraction', {}), **{k: getattr(args, k) for k in DEFAULTS if getattr(args, k) is not None}}
    extractor = Extractor(options)
    records = build_manifest(cfg)
    videos = [r for r in records if r.video_path]
    if not videos:
        raise ValueError('추출할 MP4가 없습니다')
    for record in videos:
        if record.raw_path and not args.force and not Path(record.raw_path).with_suffix('.csv.meta.json').exists():
            print(f'기존 RAW 유지: {record.raw_path} (모델 설정 변경 시 --force 사용)', flush=True)
            continue
        extractor.convert(record.video_path, record.raw_path or None, args.force)


if __name__ == '__main__':
    main()
