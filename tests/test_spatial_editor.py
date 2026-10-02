import copy
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from mcmot.config import load_config
from mcmot.spatial import build_spatial
from mcmot.spatial_editor import KINDS, SpatialConfig, SpatialEditor, validate_polygon
from mcmot.tracker import add_gaze_fields


START = [[.3, .3], [.7, .3], [.7, .7], [.3, .7]]
END = [[.3, .0], [.7, .0], [.7, .2], [.3, .2]]


class SpatialEditorTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'config.yaml'
        self.cfg = {'data_path': 'data/sample', 'output_root': 'output_sample',
                    'attributes': {'backend': 'mivolo', 'model_dir': '../weights'},
                    'spatial': {'custom_option': 42, 'routes': [], 'regions': [],
                                'gaze': {'keypoint_confidence': .7, 'targets': []}}}
        self.path.write_text('# Original comment\n' + yaml.safe_dump(self.cfg, allow_unicode=True), encoding='utf-8')
        self.original = self.path.read_bytes()
        self.model = SpatialConfig(self.path)
        self.model.spatial['routes'].append({'name': '관람 동선', 'bidirectional': True,
                                            'start_gate': copy.deepcopy(START), 'end_gate': copy.deepcopy(END)})
        self.model.spatial['regions'].append({'name': '관람 공간', 'polygon': copy.deepcopy(START)})
        self.model.spatial['gaze']['targets'].append({'name': '상단 시선', 'source_polygon': copy.deepcopy(START),
                                                    'target_polygon': copy.deepcopy(END)})

    def test_save_preserves_settings_relative_paths_and_first_backup(self):
        backup = self.model.save()
        saved = yaml.safe_load(self.path.read_text())
        self.assertEqual(saved['attributes'], self.cfg['attributes'])
        self.assertEqual(saved['data_path'], 'data/sample')
        self.assertEqual(saved['output_root'], 'output_sample')
        self.assertEqual(saved['spatial']['custom_option'], 42)
        self.assertEqual(saved['spatial']['gaze']['keypoint_confidence'], .7)
        self.assertEqual(backup.read_bytes(), self.original)
        self.assertFalse(self.model.dirty)
        self.model.spatial['routes'][0]['name'] = '새 동선'
        self.assertTrue(self.model.dirty)
        self.model.save()
        self.assertEqual(backup.read_bytes(), self.original)
        self.assertFalse(list(self.root.glob('*.partial')))

    def test_concurrent_change_is_not_overwritten(self):
        self.path.write_text('data_path: changed\n', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, '변경'):
            self.model.save()
        self.assertEqual(self.path.read_text(), 'data_path: changed\n')
        self.assertFalse(self.path.with_name(self.path.name + '.spatial.bak').exists())

    def test_pipeline_spatial_dispatch_uses_bev_without_inference(self):
        from mcmot.pipeline import main
        cfg = copy.deepcopy(self.cfg)
        cfg['bev_image'] = 'bev.png'
        (self.root / 'bev.png').touch()
        self.path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
        before = self.path.read_bytes()
        with patch('mcmot.pipeline.show_status'), patch('mcmot.spatial_editor.launch') as launch:
            main(['spatial', '--config', str(self.path)])
        launch.assert_called_once_with(str(self.path), self.root / 'bev.png')
        self.assertEqual(self.path.read_bytes(), before)

    def test_missing_polygon_or_duplicate_name_prevents_write(self):
        self.model.spatial['routes'][0]['end_gate'] = []
        with self.assertRaisesRegex(ValueError, 'end_gate'):
            self.model.save()
        self.assertEqual(self.path.read_bytes(), self.original)
        self.model.spatial['routes'][0]['end_gate'] = END
        self.model.spatial['routes'].append(copy.deepcopy(self.model.spatial['routes'][0]))
        with self.assertRaisesRegex(ValueError, '중복'):
            self.model.save()
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_polygons_reject_invalid_or_intersecting_shapes(self):
        for points in ([], [[0, 0], [1, 1]], [[0, 0], [.5, .5], [1, 1]],
                       [[0, 0], [1, 0], [1, float('nan')]],
                       [[0, 0], [1, 0], [2, 1]], [[0, 0], [1, 0], [True, 1]],
                       [[0, 0], [1, 0], [1, 1], [1, 0]],
                       [[0, 0], [1, .8], [0, 1], [.8, 0]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                validate_polygon(points)
        validate_polygon(START)
        validate_polygon(START + [START[0]])
        validate_polygon([[0, 0], [1, 0], [.4, .4], [1, 1], [0, 1]])

    def test_canvas_letterboxing_undo_and_finish_store_normalized_coordinates(self):
        editor = SpatialEditor.__new__(SpatialEditor)
        editor.model = self.model
        editor.kind = next(iter(KINDS))
        editor.index = 0
        editor.draft_key = 'start_gate'
        editor.draft = []
        # Landscape canvas displaying an image with margins around it.
        editor.transform = (100, 50, 400, 200)
        editor.status = Mock()
        editor.redraw = Mock()
        editor.refresh = Mock()
        editor.messagebox = Mock()
        editor.click(SimpleNamespace(x=50, y=50))  # Outside the image.
        self.assertEqual(editor.draft, [])
        for x, y in ((100, 50), (500, 50), (500, 250), (200, 100)):
            editor.click(SimpleNamespace(x=x, y=y))
        editor.undo()
        editor.finish()
        self.assertEqual(self.model.spatial['routes'][0]['start_gate'], [[0, 0], [1, 0], [1, 1]])
        self.assertIsNone(editor.draft)
        editor.messagebox.showerror.assert_not_called()
        self.model.save()
        self.assertEqual(yaml.safe_load(self.path.read_text())['spatial']['routes'][0]['start_gate'], [[0, 0], [1, 0], [1, 1]])

    def test_cancel_drawing_retains_old_polygon_and_does_not_save(self):
        editor = SpatialEditor.__new__(SpatialEditor)
        editor.model = self.model
        editor.draft = [[0, 0], [1, 0]]
        editor.draft_key = 'start_gate'
        editor.messagebox = Mock()
        editor.messagebox.askyesno.return_value = True
        editor.status = Mock()
        editor.redraw = Mock()
        editor.cancel_draw()
        self.assertIsNone(editor.draft)
        self.assertEqual(self.model.spatial['routes'][0]['start_gate'], START)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_gaze_priority_and_direction_change_reach_analysis(self):
        self.model.spatial['gaze']['targets'].append({'name': '동일 위치 후순위', 'source_polygon': START, 'target_polygon': END})
        self.model.save()
        cfg = load_config(self.path)
        row = {'left_ear_conf': 1, 'right_ear_conf': 1, 'left_eye_conf': 1,
               'right_eye_conf': 1, 'nose_conf': 1, 'left_ear_x': .4, 'left_ear_y': .5,
               'right_ear_x': .6, 'right_ear_y': .5, 'nose_x': .5, 'nose_y': .4,
               'bev_x': .5, 'bev_y': .5}
        add_gaze_fields(row, np.eye(3), cfg['spatial']['gaze'])
        self.assertEqual(row['gaze_target'], '상단 시선')
        targets = self.model.spatial['gaze']['targets']
        targets[0], targets[1] = targets[1], targets[0]
        self.model.save()
        cfg = load_config(self.path)
        add_gaze_fields(row, np.eye(3), cfg['spatial']['gaze'])
        self.assertEqual(row['gaze_target'], '동일 위치 후순위')
        output = self.root / 'analysis'
        (output / 'global').mkdir(parents=True)
        def write(name, rows):
            with (output / 'global' / name).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        write('persons.csv', [{'global_id': '1', 'date': '2026-09-03', 'gender': 'female', 'age': '30s'}])
        # Reverse traversal END -> START must count only when bidirectional is enabled.
        write('tracks.csv', [{'global_id': '1', 'timestamp': '2026-09-03 10:00:00', 'is_observed': '1',
                              'bev_x': .5, 'bev_y': .1, 'gaze_target': ''},
                             {'global_id': '1', 'timestamp': '2026-09-03 10:00:01', 'is_observed': '1',
                              'bev_x': .5, 'bev_y': .5, 'gaze_target': row['gaze_target']}])
        path = build_spatial(output, cfg)
        with path.open() as stream:
            events = list(csv.DictReader(stream))
        self.assertEqual({e['dimension'] for e in events}, {'route', 'region', 'gaze'})
        cfg['spatial']['routes'][0]['bidirectional'] = False
        path = build_spatial(output, cfg)
        with path.open() as stream:
            self.assertNotIn('route', {e['dimension'] for e in csv.DictReader(stream)})


if __name__ == '__main__':
    unittest.main()
