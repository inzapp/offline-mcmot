#!/usr/bin/env python3
"""Verify real E2E artifacts, including frame completion and demographic propagation."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcmot.config import load_config, select_output_root
from mcmot.inventory import build_manifest


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def verify(config, output=None, duration_seconds=None, fps=None):
    cfg = load_config(config)
    root = Path(output) if output else Path(select_output_root(cfg, fresh=False)['output_root'])
    recordings = build_manifest(cfg, deep=True)
    assert len(recordings) == 2, f'Expected 2 recordings; found {len(recordings)}'
    raw_count, frame_count = 0, 0
    for recording in recordings:
        assert recording.status == 'ok', recording.status
        raw, bev = rows(recording.raw_path), rows(recording.bev_path)
        assert len(raw) == len(bev) and raw, 'CSV row count mismatch or empty inference'
        assert [(r['timestamp'], r['frame_index']) for r in raw] == [(r['timestamp'], r['frame_index']) for r in bev]
        meta = json.loads(Path(recording.raw_path).with_suffix('.csv.meta.json').read_text())
        assert meta['frames'] == recording.video_frames, 'Incomplete inference'
        if duration_seconds is not None and fps is not None:
            assert meta['frames'] == round(duration_seconds * fps), 'Unexpected clip frame count'
        raw_count += len(raw); frame_count += meta['frames']
    local = rows(root / 'local/persons.csv')
    global_people = rows(root / 'global/persons.csv')
    mapping = rows(root / 'global/id_mapping.csv')
    attrs = rows(root / 'attributes/inference_results.csv')
    predictions = rows(root / 'attributes/results.csv')
    assert local and global_people and predictions, 'Missing actual tracking/attribute outputs'
    assert {r['local_uid'] for r in attrs} == {r['local_uid'] for r in local}
    assert {r['local_uid'] for r in mapping} == {r['local_uid'] for r in local}
    by_uid = {r['local_uid']: r for r in attrs}
    for attr in attrs:
        embedding = np.load(attr['reid_embedding_path'], allow_pickle=False)
        assert embedding.size and np.isfinite(embedding).all(), 'Invalid ReID embedding'
    for prediction in predictions:
        if prediction['status'] == 'ok':
            normalized = by_uid[prediction['local_uid']]
            assert normalized['gender'] == prediction['gender']
            assert float(normalized['age_estimate']) == float(prediction['age'])
    by_crop = {r['crop_path']: r for r in attrs}
    for person in global_people:
        attr = by_crop.get(person['representative_crop_path'])
        if attr:
            assert (person['gender'], person['age']) == (attr['gender'], attr['age']), 'Global demographics differ from model output'
    issues = rows(root / 'quality/issues.csv')
    assert not [r for r in issues if r['severity'] == 'error'], 'Tracking errors'
    assert not [r for r in predictions if r['status'] != 'ok' and r.get('file')], 'MiVOLO errors on actual crops'
    report = root / 'report/index.html'
    assert report.is_file() and (root / 'report/index_export.html').is_file()
    assert (root / 'spatial/demographics_summary.csv').is_file()
    associations = rows(root / 'global/association_events.csv')
    spatial_events = rows(root / 'spatial/person_events.csv')
    people_by_id = {r['global_id']: r for r in global_people}
    spatial_groups = {}
    for event in spatial_events:
        person = people_by_id[event['global_id']]
        assert (event['gender'], event['age']) == (person['gender'], person['age']), 'Spatial demographics differ'
        for hour in ('all', event['first_seen'][11:13] + ':00'):
            key = (event['date'], hour, event['dimension'], event['name'], event['gender'], event['age'])
            spatial_groups.setdefault(key, set()).add(event['global_id'])
    spatial_summary = rows(root / 'spatial/demographics_summary.csv')
    actual_groups = {tuple(r[k] for k in ('date', 'hour', 'dimension', 'name', 'gender', 'age')):
                     int(r['person_count']) for r in spatial_summary}
    assert actual_groups == {k: len(v) for k, v in spatial_groups.items()}, 'Spatial aggregation mismatch'
    summary = {'recordings': len(recordings), 'frames': frame_count, 'raw_rows': raw_count,
               'bev_rows': raw_count, 'local_ids': len(local), 'global_ids': len(global_people),
               'mivolo_status': dict(Counter(r['status'] for r in predictions)),
               'mivolo_input_modes': dict(Counter(r.get('input_mode', '') for r in predictions)),
               'reid_embeddings': len(attrs),
               'global_gender': dict(Counter(r.get('gender') or 'unknown' for r in global_people)),
               'global_age': dict(Counter(r.get('age') or 'unknown' for r in global_people)),
               'tracking_errors': 0,
               'association_candidates': len(associations),
               'accepted_associations': sum(r['accepted'] == '1' for r in associations),
               'spatial_events': len(spatial_events),
               'output_root': str(root.resolve())}
    pptx = Path(cfg.get('pptx', {}).get('output') or root / 'report' / f'{cfg["site"]}.pptx')
    if cfg.get('pptx', {}).get('template'):
        with zipfile.ZipFile(pptx) as archive:
            assert archive.testzip() is None
            slides = [name for name in archive.namelist() if name.startswith('ppt/slides/slide') and name.endswith('.xml')]
        with zipfile.ZipFile(cfg['pptx']['template']) as template:
            original = [name for name in template.namelist() if name.startswith('ppt/slides/slide') and name.endswith('.xml')]
        assert len(slides) == len(original), 'Template slide count changed'
        summary.update(pptx=str(pptx.resolve()), slides=len(slides))
    (root / 'e2e_verification.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--duration-seconds', type=float)
    parser.add_argument('--fps', type=float)
    args = parser.parse_args()
    verify(args.config, args.output, args.duration_seconds, args.fps)
