#!/usr/bin/env python3
"""Run one MiVOLO process per dataset, preserving local_uid in every result."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--persons', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--options', type=Path, required=True)
    args = parser.parse_args()
    import cv2
    from mcmot.mivolo_backend import SnapshotEstimator
    from mcmot.mivolo_statistics import write_statistics
    options = json.loads(args.options.read_text())
    allowed = ('device', 'threads', 'face_conf', 'min_face_size', 'body_only', 'detector_size', 'model_dir', 'repository')
    selected = {k: options[k] for k in allowed if k in options}
    if options.get('detector'):
        selected['detector_path'] = options['detector']
    persons = list(csv.DictReader(args.persons.open(encoding='utf-8')))
    estimator = SnapshotEstimator(**selected) if persons else None
    results = []
    for index, person in enumerate(persons, 1):
        row = {'local_uid': person['local_uid'], 'file': person.get('crop_path', '')}
        image = cv2.imread(row['file']) if row['file'] else None
        try:
            if image is None:
                raise ValueError('crop unavailable')
            result, _ = estimator.predict(image)
            row.update(result, gender=result['gender'].lower(), status='ok', error='')
        except Exception as exc:
            row.update(status='error', error=f'{type(exc).__name__}: {exc}')
        results.append(row)
        if index % 100 == 0:
            print(f'MiVOLO: {index}/{len(persons)}', flush=True)
    fields = ['local_uid', 'file', 'status', 'age', 'gender', 'gender_score', 'male_score', 'female_score', 'input_mode', 'face_status', 'face_bbox', 'face_detection_score', 'image_width', 'image_height', 'detection_ms', 'preprocess_ms', 'inference_ms', 'pipeline_ms', 'error']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.csv.partial')
    with partial.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(results)
    partial.replace(args.output)
    write_statistics(results, args.output.parent / 'statistics')


if __name__ == '__main__':
    main()
