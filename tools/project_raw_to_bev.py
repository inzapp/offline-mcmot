#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcmot.config import load_config
from mcmot.preparation import camera_matrix, convert_bev


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='raw CSV → timestamp,frame_index,x,y 정규화 BEV CSV')
    parser.add_argument('raw', type=Path); parser.add_argument('--config', required=True)
    parser.add_argument('--camera', required=True); parser.add_argument('--output', type=Path)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    convert_bev(args.raw, args.output or args.raw.with_name(args.raw.stem + '_bev.csv'),
                camera_matrix(load_config(args.config), args.camera), args.force)
