#!/usr/bin/env python3
"""Standalone entry point; see --help for detection/pose defaults."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcmot.raw import main

if __name__ == '__main__':
    main()
