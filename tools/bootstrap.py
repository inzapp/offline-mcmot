#!/usr/bin/env python3
"""Create project environments without shell activation or changing system packages."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def environment(name):
    root = ROOT / name
    python = root / 'bin/python'
    if not python.exists():
        candidate = os.environ.get('PIPELINE_PYTHON', sys.executable)
        version = json.loads(subprocess.check_output([candidate, '-c', 'import json,sys; print(json.dumps(list(sys.version_info[:2])))'], text=True))
        if version not in ([3, 10], [3, 11]):
            raise RuntimeError('이 파이프라인은 Python 3.10/3.11이 필요합니다. PIPELINE_PYTHON을 지정하세요.')
        print(f'환경 생성: {root}', flush=True)
        subprocess.run([candidate, '-m', 'venv', str(root)], check=True)
    return python


def install(python, name, arguments, fingerprint):
    marker = python.parent.parent / f'.pipeline-{name}'
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()
    if marker.exists() and marker.read_text() == digest:
        return
    subprocess.run([str(python), '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', *arguments], check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
    marker.write_text(digest)
    subprocess.run([str(python), '-m', 'pip', 'freeze'], check=True,
                   stdout=(python.parent.parent / 'installed-packages.txt').open('w'))


def light_environment():
    python = environment('.venv')
    install(python, 'light', ['numpy==1.23.5', 'PyYAML>=6', 'opencv-python==4.11.0.86'], 'light-v1')
    return python


def analytics_environment():
    python = light_environment()
    files = [ROOT / 'requirements.txt', ROOT / 'constraints-analytics.txt']
    install(python, 'analytics', ['-r', str(files[0]), '-c', str(files[1])], ''.join(p.read_text() for p in files))
    return python


def ultralytics_environment(attributes=None):
    attributes = attributes or {}
    python = environment('.venv_ultralytics')
    mivolo = attributes.get('backend') == 'mivolo'
    if mivolo:
        repo = Path(attributes.get('repository', ROOT / 'vendor/mivolo'))
        if not repo.is_dir():
            raise FileNotFoundError(f'MiVOLO repository 없음: {repo}')
        req = ROOT / 'requirements-mivolo.txt'
        index = attributes.get('torch_index_url', 'https://download.pytorch.org/whl/cu128')
        torch = attributes.get('torch_version', '2.7.1')
        vision = attributes.get('torchvision_version', '0.22.1')
        install(python, 'torch', [f'torch=={torch}', f'torchvision=={vision}', '--index-url', index], f'{torch}|{vision}|{index}')
        install(python, 'mivolo', ['-r', str(req)], req.read_text() + str(repo))
        install(python, 'mivolo-vendor', ['--no-deps', '--no-build-isolation', str(repo)], str(repo) + (repo / 'setup.py').read_text())
    else:
        # Do not upgrade an already shared MiVOLO environment to a different YOLO version.
        shared = python.parent.parent / '.pipeline-mivolo'
        if not shared.exists():
            install(python, 'ultralytics', ['ultralytics'], 'ultralytics-v1')
    return python


if __name__ == '__main__':
    if not sys.argv[1:] or sys.argv[1:] in (['--help'], ['-h']):
        print('Usage: ./pipeline.sh {status,setup,calibrate,spatial,raw,bev,validate,run,pptx,all} --config FILE\n'
              'Options: --camera ID --force --resume --deep --check-models --skip-inference --with-video --skip-video\n'
              'Raw options: --det-model --pose-model --det-imgsz --pose-imgsz --det-conf --pose-conf --merge-iou --device')
        raise SystemExit(0)
    python = light_environment()
    os.chdir(ROOT)
    os.execv(str(python), [str(python), '-u', '-m', 'mcmot.pipeline', *sys.argv[1:]])
