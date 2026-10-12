import contextlib
import io
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mcmot.inventory import build_manifest, preflight_videos
from mcmot.pipeline import main
from tools import raw_batch


class VideoPreflightTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        slot = root / '7/2026-09-03'
        slot.mkdir(parents=True)
        self.bad = slot / '2026-09-03_DEVICE_100000.mp4'
        self.good = slot / '2026-09-03_DEVICE_103000.mp4'
        for video in (self.bad, self.good):
            video.write_bytes(b'video')
            video.with_suffix('.csv').write_text('raw')
            video.with_name(video.stem + '_bev.csv').write_text('bev')
        self.cfg = {'site': 'sample', 'dates': ['2026-09-03'], 'data_path': str(root),
                    'video_root': str(root), 'config_path': str(root / 'config.yaml')}

    def test_probe_lists_missing_moov_and_releases_good_capture(self):
        # Recheck previous exclusions so repaired files can rejoin a later run.
        self.cfg['pipeline'] = {'excluded_videos': [str(self.good)]}
        capture = Mock()
        capture.isOpened.return_value = True
        results = [SimpleNamespace(returncode=1, stderr='moov atom not found'),
                   SimpleNamespace(returncode=0, stderr='')]
        with patch('mcmot.inventory.subprocess.run', side_effect=results) as probe, \
                patch('cv2.VideoCapture', return_value=capture) as open_video, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            excluded = preflight_videos(self.cfg)
        self.assertEqual(excluded, [str(self.bad)])
        self.assertEqual(probe.call_count, 2)
        open_video.assert_called_once_with(str(self.good))
        capture.release.assert_called_once()
        self.assertIn(f'WARNING: moov atom not found — 녹화 전체 skip: {self.bad}', output.getvalue())

    @unittest.skipUnless(shutil.which('ffprobe'), 'ffprobe required')
    def test_real_mp4_without_moov_is_warning_only(self):
        self.good.unlink()
        # A real MP4 container header and mdat with no moov metadata.
        self.bad.write_bytes(struct.pack('>I4s4sI4s', 20, b'ftyp', b'isom', 0, b'isom')
                             + struct.pack('>I4s', 8, b'mdat'))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(preflight_videos(self.cfg), [str(self.bad)])
        self.assertIn('WARNING: moov atom not found', output.getvalue())

    def test_deep_manifest_excludes_existing_csvs_before_probing(self):
        self.cfg['pipeline'] = {'excluded_videos': [str(self.bad)]}
        with patch('mcmot.inventory.video_stats', return_value=(10, 17900, 1790)) as stats, \
                patch('mcmot.inventory.csv_stats', return_value=(0, '', '')) as csv_stats:
            records = build_manifest(self.cfg, deep=True)
        self.assertEqual([r.video_path for r in records], [str(self.good)])
        stats.assert_called_once_with(self.good)
        self.assertEqual(csv_stats.call_count, 2)

    def test_other_open_errors_still_fail_after_reporting_moov_list(self):
        results = [SimpleNamespace(returncode=1, stderr='moov atom not found'),
                   SimpleNamespace(returncode=1, stderr='Permission denied')]
        with patch('mcmot.inventory.subprocess.run', side_effect=results), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError, 'Permission denied'):
                preflight_videos(self.cfg)
        self.assertIn(str(self.bad), output.getvalue())

    def test_all_passes_exclusions_to_raw_and_every_following_stage(self):
        extractor = Mock()
        bootstrap = SimpleNamespace(analytics_environment=Mock(), ultralytics_environment=Mock())

        def preflight(cfg):
            bootstrap.analytics_environment.assert_not_called()
            bootstrap.ultralytics_environment.assert_not_called()
            return [str(self.bad)]

        def run_raw(arguments):
            exclusion_file = Path(arguments[arguments.index('--excluded-videos') + 1])
            self.assertEqual(json.loads(exclusion_file.read_text()), [str(self.bad)])
            with patch.object(sys, 'argv', list(map(str, arguments[1:]))), \
                    patch('tools.raw_batch.load_config', return_value=dict(self.cfg)), \
                    patch('tools.raw_batch.Extractor', return_value=extractor):
                raw_batch.main()

        def check_filtered(cfg, *args):
            self.assertEqual([r.video_path for r in build_manifest(cfg)], [str(self.good)])
            return True

        def analyze(cfg, args):
            check_filtered(cfg)
            return cfg

        with patch.dict('sys.modules', {'bootstrap': bootstrap}), \
                patch('mcmot.pipeline.load_config', return_value=self.cfg), \
                patch('mcmot.pipeline.show_status'), \
                patch('mcmot.pipeline.preflight_videos', side_effect=preflight), \
                patch('mcmot.pipeline.run_process', side_effect=run_raw), \
                patch('mcmot.pipeline.prepare_bev', side_effect=check_filtered), \
                patch('mcmot.pipeline.validate_inputs', side_effect=check_filtered), \
                patch('mcmot.pipeline.analysis', side_effect=analyze), \
                contextlib.redirect_stdout(io.StringIO()):
            main(['all', '--config', self.cfg['config_path'], '--force', '--deep', '--skip-inference'])
        extractor.convert.assert_called_once_with(str(self.good), str(self.good.with_suffix('.csv')), True)

    def test_all_excluded_exits_without_installation_or_processing(self):
        with patch('mcmot.pipeline.load_config', return_value=self.cfg), \
                patch('mcmot.pipeline.show_status'), \
                patch('mcmot.pipeline.preflight_videos', return_value=[str(self.bad), str(self.good)]), \
                patch('mcmot.pipeline.run_process') as run, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            main(['all', '--config', self.cfg['config_path']])
        run.assert_not_called()
        self.assertIn('WARNING: 처리 가능한 녹화가 없어', output.getvalue())


if __name__ == '__main__':
    unittest.main()
