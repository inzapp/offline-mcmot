import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from mcmot.attributes import infer_all, normalize_mivolo
from mcmot.config import load_config
from mcmot.inventory import build_manifest
from mcmot.preparation import camera_matrix, convert_bev, show_status, validate_inputs
from mcmot.raw import KEYPOINT_NAMES, csv_columns, merged_detections, result_rows
from mcmot.tracker import load_roi


class PreparationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.slot = self.root / '7/2026-09-03'
        self.slot.mkdir(parents=True)
        self.stem = '2026-09-03_DEVICE_100000'
        self.raw = self.slot / (self.stem + '.csv')
        self.bev = self.slot / (self.stem + '_bev.csv')
        self.cfg = {'site': 'sample', 'dates': ['2026-09-03'], 'data_path': str(self.root),
                    'video_root': str(self.root), 'config_path': str(self.root / 'config.yaml')}
        square = [[0, 0], [1, 0], [1, 1], [0, 1]]
        (self.root / '7/7_roi.json').write_text(json.dumps({'rois': [
            {'image1_vertices_normalized': square, 'image2_vertices_normalized': square}]}))
        cv2.imwrite(str(self.root / 'bev.png'), np.zeros((10, 10, 3), dtype=np.uint8))
        cv2.imwrite(str(self.root / '7/7.jpg'), np.zeros((10, 10, 3), dtype=np.uint8))
        self.write_raw()

    def write_raw(self):
        with self.raw.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=csv_columns()); writer.writeheader()
            for x in (.5, -0.2):
                row = dict.fromkeys(csv_columns(), 0)
                row.update(timestamp='2026-09-03 10:00:00.000', frame_index=0, confidence=.9, cx=x, cy=.4, w=.2, h=.2)
                writer.writerow(row)

    def test_conversion_preserves_schema_row_order_and_unclipped_coordinates(self):
        convert_bev(self.raw, self.bev, np.eye(3))
        with self.bev.open() as stream:
            reader = csv.DictReader(stream); rows = list(reader)
        self.assertEqual(reader.fieldnames, ['timestamp', 'frame_index', 'x', 'y'])
        self.assertEqual([float(r['x']) for r in rows], [.5, -.2])
        self.assertEqual([float(r['y']) for r in rows], [.5, .5])
        original = self.bev.stat().st_mtime_ns
        convert_bev(self.raw, self.bev, np.eye(3))
        self.assertEqual(original, self.bev.stat().st_mtime_ns)
        with self.assertRaises(FileExistsError):
            convert_bev(self.raw, self.bev, np.diag([2, 2, 1]))

    def test_failed_force_conversion_keeps_previous_complete_output(self):
        convert_bev(self.raw, self.bev, np.eye(3))
        expected = self.bev.read_bytes()
        self.raw.write_text('timestamp,frame_index,cx,cy,h\nt,0,nan,.4,.2\n')
        with self.assertRaises(ValueError):
            convert_bev(self.raw, self.bev, np.eye(3), force=True)
        self.assertEqual(self.bev.read_bytes(), expected)
        self.assertFalse(self.bev.with_suffix('.csv.partial').exists())

    def test_pixel_matrix_direction_and_normalized_scaling(self):
        calibration = self.root / '7/calibration'
        calibration.mkdir()
        # BEV 200x100 → camera 1000x500: scale 5, then camera x +100 pixels.
        (calibration / 'homography.json').write_text(json.dumps({
            'direction': 'image1_to_image2', 'image_sizes': [[200, 100], [1000, 500]],
            'H': [[5, 0, 100], [0, 5, 0], [0, 0, 1]]}))
        matrix = camera_matrix(self.cfg, '7')
        projected = matrix @ np.array([.6, .5, 1])
        np.testing.assert_allclose(projected[:2] / projected[2], [.5, .5])

    def test_flat_inventory_and_status_matching_counts(self):
        self.raw.with_suffix('.mp4').write_bytes(b'video')
        convert_bev(self.raw, self.bev, np.eye(3))
        records = build_manifest(self.cfg)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].status, 'ok')
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            show_status(self.cfg)
        text = captured.getvalue()
        self.assertIn('Complete 1/1', text)
        self.assertIn('✓ 7.jpg [camera image]', text)
        self.assertIn('└──', text)
        self.assertNotIn('(missing)', text)

    def test_deep_validation_rejects_same_count_but_misaligned_rows(self):
        self.raw.with_suffix('.mp4').write_bytes(b'video')
        convert_bev(self.raw, self.bev, np.eye(3))
        with patch('mcmot.inventory.video_stats', return_value=(10, 17900, 1790)):
            self.assertTrue(validate_inputs(self.cfg, deep=True))
            self.bev.write_text(self.bev.read_text().replace(',0,', ',1,', 1))
            self.assertFalse(validate_inputs(self.cfg, deep=True))

    def test_config_resolves_section_paths_and_model_names(self):
        config = self.root / 'config.yaml'
        config.write_text('data_path: .\noutput_root: output\ndates: ["2026-09-03"]\nextraction:\n  det_model: yolov8m.pt\n')
        cfg = load_config(config)
        self.assertEqual(cfg['video_root'], str(self.root))
        self.assertEqual(cfg['bev_image'], str(self.root / 'bev.png'))
        self.assertEqual(cfg['extraction']['det_model'], 'yolov8m.pt')
        (self.root / 'yolov8m.pt').write_bytes(b'downloaded model')
        self.assertEqual(load_config(config)['extraction']['det_model'], 'yolov8m.pt')

    def test_combined_roi_supports_multiple_rois_per_camera(self):
        combined = self.root / 'roi.json'
        payload = json.loads((self.root / '7/7_roi.json').read_text())
        combined.write_text(json.dumps({'7': payload}))
        camera, bev = load_roi(combined, '7')
        self.assertEqual(camera, [payload['rois'][0]['image2_vertices_normalized']])
        self.assertEqual(bev, [payload['rois'][0]['image1_vertices_normalized']])

    def test_calibration_recovers_saved_points_and_registers_roi(self):
        from mcmot.pipeline import calibrate
        output = self.root / '7/calibration'
        draft = output / 'bev.png__7.jpg/previous/homography.json'
        draft.parent.mkdir(parents=True)
        draft.write_text('{}')
        shared = self.root / 'roi.json'
        self.cfg['roi_file'] = str(shared)
        payload = json.loads((self.root / '7/7_roi.json').read_text())

        def gui(arguments):
            self.assertIn('--require-roi', arguments)
            self.assertEqual(arguments[arguments.index('--load') + 1], draft)
            saved = draft.parent.parent / 'new'
            saved.mkdir()
            (saved / 'homography.json').write_text('{"H": [[1,0,0],[0,1,0],[0,0,1]]}')
            (saved / 'roi.json').write_text(json.dumps(payload))

        with patch.dict('os.environ', {'DISPLAY': ':0'}), patch('mcmot.pipeline.run_process', side_effect=gui):
            calibrate(self.cfg, '7')
        self.assertTrue((output / 'homography.json').is_file())
        self.assertEqual(json.loads(shared.read_text())['7'], payload)

    def test_calibration_save_requires_roi_before_writing_files(self):
        from tools.homography_tool import HomographyTool
        output = self.root / 'calibration'
        tool = HomographyTool([self.root / 'bev.png', self.root / '7/7.jpg'], output)
        tool.matrix = np.eye(3)
        tool.require_roi = True
        with self.assertRaisesRegex(ValueError, 'ROI'):
            tool.save()
        self.assertFalse(output.exists())

    def test_report_visuals_support_nested_and_legacy_roi_payloads(self):
        from mcmot.report_visuals import _roi_specs, _camera_roi_images, _bev_overview, _boundary_images
        payload = json.loads((self.root / '7/7_roi.json').read_text())
        shared = self.root / 'roi.json'
        self.cfg.update(roi_file=str(shared), bev_image=str(self.root / 'bev.png'),
                        global_matching={'boundary_quantile': .15})
        assets = self.root / 'assets'
        for nested in (False, True):
            rois = payload['rois'] * 2
            shared.write_text(json.dumps({'7': {'rois': rois} if nested else rois[0]}))
            specs = _roi_specs(self.cfg)
            images = _camera_roi_images(self.cfg, assets, specs)
            self.assertEqual(images[0]['vertices'], 8 if nested else 4)
            self.assertTrue((assets / 'camera_7_roi.jpg').is_file())
            self.assertTrue(all(_bev_overview(self.cfg, assets, specs, [])))
            people = [{'camera_id': '7', 'local_uid': 'uid', 'start_boundary_distance': '.01',
                       'end_boundary_distance': '.02'}]
            self.assertEqual(len(_boundary_images(self.cfg, assets, specs, people, {})), 1)


class InferenceContractTest(unittest.TestCase):
    def test_empty_pose_without_keypoint_confidence_is_valid(self):
        def tensor(values):
            return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: values))
        result = SimpleNamespace(boxes=SimpleNamespace(xywhn=tensor([]), conf=tensor([])),
                                 keypoints=SimpleNamespace(conf=None))
        self.assertEqual(result_rows(result, .02, True), [])
        result.boxes = SimpleNamespace(xywhn=tensor([[.5, .5, .2, .4]]), conf=tensor([.8]))
        with self.assertRaises(ValueError):
            result_rows(result, .02, True)

    def test_mivolo_failure_preserves_reid_embedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'person.jpg'
            cv2.imwrite(str(image), np.zeros((64, 64, 3), dtype=np.uint8))
            persons = root / 'persons.csv'
            with persons.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['local_uid', 'crop_path']); writer.writeheader()
                writer.writerows([{'local_uid': uid, 'crop_path': str(image)} for uid in ('local1', 'local2')])

            def inference(arguments, **kwargs):
                output = Path(arguments[arguments.index('--output') + 1])
                output.write_text('local_uid,file,status,age,gender,gender_score\n'
                                  f'local1,{image},ok,36.2,female,.9\n'
                                  f'local2,{image},error,,,\n')

            with patch('mcmot.attributes.subprocess.run', side_effect=inference), patch('mcmot.attributes.ReIDModel') as reid:
                reid.return_value.embed.return_value = np.ones((2, 4), dtype=np.float32)
                output = root / 'attributes/inference_results.csv'
                embeddings = infer_all(persons, output, {'attributes': {'backend': 'mivolo'}, 'reid': {'batch_size': 32}})
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['age'], '30-39')
            self.assertEqual(float(rows[0]['age_estimate']), 36.2)
            self.assertEqual(rows[1]['age'], 'unknown')
            self.assertEqual(rows[1]['status'], 'attributes_unavailable')
            self.assertTrue(Path(rows[1]['reid_embedding_path']).is_file())
            self.assertEqual(set(embeddings), {'local1', 'local2'})
            self.assertFalse(output.with_suffix('.csv.partial').exists())

    def test_pose_has_priority_over_higher_confidence_detector(self):
        pose = dict(confidence=.03, cx=.5, cy=.5, w=.2, h=.2)
        det = dict(confidence=.99, cx=.5, cy=.5, w=.2, h=.2)
        separate = dict(confidence=.8, cx=.8, cy=.5, w=.1, h=.2)
        self.assertEqual(merged_detections([pose], [det, separate], .45), [pose, separate])

    def test_mivolo_age_is_binned_but_continuous_value_is_preserved(self):
        row = normalize_mivolo({'status': 'ok', 'age': '27.5', 'gender': 'Female', 'gender_score': '.92'})
        self.assertEqual(row['age'], '20-29')
        self.assertEqual(row['age_estimate'], 27.5)
        self.assertEqual(row['gender'], 'female')
        self.assertEqual(normalize_mivolo({'status': 'error'})['age'], 'unknown')
        self.assertEqual(normalize_mivolo({'status': 'ok', 'age': 'nan'})['age'], 'unknown')


if __name__ == '__main__':
    unittest.main()
