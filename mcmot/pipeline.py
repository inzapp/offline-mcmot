"""Repository-level preparation and analysis commands used by pipeline.sh."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .config import load_config, select_output_root
from .inventory import build_manifest
from .preparation import (bev_image, calibration_path, camera_matrix, convert_bev, file_signature,
                          image_candidates, roi_path, section_root, show_status, target_bev, validate_inputs)

ROOT = Path(__file__).resolve().parents[1]


def run_process(arguments):
    print('실행: ' + ' '.join(map(str, arguments)), flush=True)
    subprocess.run(list(map(str, arguments)), check=True, cwd=ROOT)


def calibrate(cfg, camera=None):
    if not os.environ.get('DISPLAY'):
        raise RuntimeError('Homography GUI는 데스크톱 또는 X11 DISPLAY 연결이 필요합니다')
    root = section_root(cfg)
    cameras = [camera] if camera else sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    for camera in cameras:
        images = image_candidates(root / camera, camera)
        if len(images) != 1:
            raise ValueError(f'{camera}: 카메라 jpg/png 이미지가 정확히 1개 필요합니다')
        output = root / camera / 'calibration'
        bev = bev_image(cfg)
        arguments = [sys.executable, ROOT / 'tools/homography_tool.py', bev, images[0], '--output', output, '--require-roi']
        current = calibration_path(cfg, camera)
        if current.exists():
            arguments += ['--load', current]
        else:
            # Recover explicitly saved GUI work that was not registered (e.g. ROI missing).
            drafts = list((output / (bev.name + '__' + images[0].name)).glob('*/homography.json'))
            if drafts:
                draft = max(drafts, key=lambda p: p.stat().st_mtime_ns)
                arguments += ['--load', draft]
                print(f'이전 저장 대응점 불러오기: {draft}', flush=True)
        before = set(output.glob('*/*/homography.json'))
        run_process(arguments)
        created = set(output.glob('*/*/homography.json')) - before
        if not created:
            raise RuntimeError(f'{camera}: 저장된 새 보정 결과 없음; GUI에서 S로 저장해야 합니다')
        selected = max(created, key=lambda p: p.stat().st_mtime_ns)
        # Stable path points at the explicitly saved revision, never an arbitrary latest file.
        selected_roi = selected.with_name('roi.json')
        if not selected_roi.exists():
            raise RuntimeError(f'{camera}: ROI 없음. calibrate를 다시 실행하고 P → ROI 4점 이상 → Enter → S → Q 순서로 저장하세요.')
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_text(selected.read_text(), encoding='utf-8')
        destination = roi_path(cfg, camera)
        if cfg.get('roi_file'):
            data = json.loads(destination.read_text()) if destination.exists() else {}
            data[camera] = json.loads(selected_roi.read_text())
            destination.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            destination.write_text(selected_roi.read_text(), encoding='utf-8')
        print(f'보정 등록: {current}; ROI: {destination}', flush=True)


def prepare_bev(cfg, force):
    matrices = {}
    records = build_manifest(cfg)
    raw_records = [r for r in records if r.raw_path]
    if not raw_records:
        raise ValueError('변환할 raw CSV가 없습니다')
    for record in raw_records:
        if record.bev_path and not force and not Path(record.bev_path).with_suffix('.csv.meta.json').exists():
            print(f'기존 BEV 유지: {record.bev_path} (재생성은 --force)', flush=True)
            continue
        if record.camera_id not in matrices:
            matrices[record.camera_id] = camera_matrix(cfg, record.camera_id)
        convert_bev(record.raw_path, record.bev_path or target_bev(record), matrices[record.camera_id], force)


def check_models(cfg):
    import tempfile
    from .attributes import AttributeModels
    # Release TensorFlow's validation resources before the PyTorch worker starts.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=ROOT / 'work', delete=True) as stream:
        json.dump(cfg['reid'], stream); stream.flush()
        run_process([sys.executable, ROOT / 'tools/check_reid.py', stream.name])
    if cfg.get('attributes', {}).get('backend') == 'mivolo':
        options = cfg['attributes']
        python = options.get('python') or str(ROOT / '.venv_ultralytics/bin/python')
        # Exercise the same loader and model input/output as real inference, once.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=ROOT / 'work', delete=True) as stream:
            json.dump(options, stream); stream.flush()
            run_process([python, ROOT / 'tools/check_mivolo.py', stream.name])
    else:
        AttributeModels(cfg['attributes'])


def analysis(cfg, args):
    from .cli import run_all
    selected = select_output_root(cfg, fresh=not args.resume)
    root = Path(selected['output_root'])
    signatures = []
    for record in build_manifest(cfg):
        signatures.extend(file_signature(p) for p in (record.raw_path, record.bev_path, record.video_path) if p)
    for camera in sorted({r.camera_id for r in build_manifest(cfg)}):
        for p in (roi_path(cfg, camera), calibration_path(cfg, camera)):
            if p.exists():
                signatures.append(file_signature(p))
    for section, keys in (('attributes', ('model_dir', 'detector', 'gender_model', 'age_model')),
                          ('reid', ('model', 'config'))):
        for key in keys:
            value = cfg.get(section, {}).get(key)
            if value:
                path = Path(value)
                if path.is_file():
                    signatures.append(file_signature(path))
                elif path.is_dir():
                    signatures.extend(file_signature(p) for p in sorted(path.rglob('*')) if p.is_file())
    if cfg.get('bev_image') and Path(cfg['bev_image']).is_file():
        signatures.append(file_signature(cfg['bev_image']))
    fingerprint_cfg = {k: v for k, v in cfg.items() if k not in ('config_path', 'output_root', 'pptx')}
    signature = {'config': fingerprint_cfg, 'inputs': signatures, 'skip_inference': args.skip_inference}
    marker = root / 'pipeline_inputs.json'
    if args.resume and root.exists():
        if not marker.exists() or json.loads(marker.read_text()) != signature:
            raise RuntimeError('--resume 입력/설정이 기존 실행과 다릅니다. --resume 없이 새 output으로 실행하세요')
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(signature, ensure_ascii=False, indent=2))
    print(f'분석 출력: {root}', flush=True)
    run_all(selected, args.deep, args.resume, args.skip_inference, not args.with_video)
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['status', 'setup', 'calibrate', 'spatial', 'raw', 'bev', 'validate', 'run', 'pptx', 'all'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--camera'); parser.add_argument('--force', action='store_true')
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--deep', action='store_true')
    parser.add_argument('--skip-inference', action='store_true')
    video_options = parser.add_mutually_exclusive_group()
    video_options.add_argument('--with-video', action='store_true')
    video_options.add_argument('--skip-video', action='store_true', help='기본 동작: 결과 시각화 영상 생략')
    parser.add_argument('--check-models', action='store_true')
    from .raw import DEFAULTS, add_arguments
    add_arguments(parser); args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cfg.setdefault('pipeline', {}).setdefault('strict', True)
    if cfg.get('attributes', {}).get('backend') == 'mivolo':
        cfg['attributes'].setdefault('repository', str(ROOT / 'vendor/mivolo'))
    # Always display current input counts before processing or installing heavy dependencies.
    show_status(cfg)
    if args.command == 'status':
        return
    sys.path.insert(0, str(ROOT / 'tools'))
    from bootstrap import analytics_environment, ultralytics_environment
    if args.command in ('setup', 'run', 'all', 'pptx') or args.check_models:
        analytics_environment()
    if args.command in ('setup', 'raw', 'all') or (args.check_models and cfg.get('attributes', {}).get('backend') == 'mivolo') or (args.command == 'run' and cfg.get('attributes', {}).get('backend') == 'mivolo' and not args.skip_inference):
        ultralytics_environment(cfg.get('attributes', {}))
    if args.command == 'setup':
        print('환경 준비 완료: .venv + .venv_ultralytics (raw/MiVOLO 공유)')
        return
    if args.command == 'calibrate':
        calibrate(cfg, args.camera)
    if args.command == 'spatial':
        from .spatial_editor import launch
        launch(cfg['config_path'], bev_image(cfg))
    if args.command in ('raw', 'all'):
        arguments = [ROOT / '.venv_ultralytics/bin/python', ROOT / 'tools/raw_batch.py', '--config', cfg['config_path']]
        if args.force:
            arguments.append('--force')
        for key in DEFAULTS:
            value = getattr(args, key)
            if value is not None:
                arguments += ['--' + key.replace('_', '-'), str(value)]
        run_process(arguments)
        show_status(cfg)
    if args.command in ('bev', 'all'):
        prepare_bev(cfg, args.force)
        show_status(cfg)
    if args.command in ('validate', 'run', 'all'):
        if not validate_inputs(cfg, args.deep):
            raise RuntimeError('입력 검증 실패: 위 ERROR를 해결한 뒤 재실행하세요')
        if args.check_models or (args.command in ('run', 'all') and not args.skip_inference):
            (ROOT / 'work').mkdir(exist_ok=True)
            check_models(cfg)
    if args.command in ('run', 'all'):
        cfg = analysis(cfg, args)
    if args.command == 'pptx' or (args.command == 'all' and cfg.get('pptx', {}).get('template')):
        from .pptx_agent import generate
        generate(cfg if args.command == 'all' else select_output_root(cfg, fresh=False))
    elif args.command == 'all':
        print('PPT 생성 대기: YAML pptx.template과 pptx.prompt를 지정하면 all 실행에 포함됩니다')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
