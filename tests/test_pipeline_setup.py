import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

from mcmot.pipeline import ROOT, main


class PipelineSetupTest(unittest.TestCase):
    def setUp(self):
        self.analytics = Mock()
        self.ultralytics = Mock()
        self.calls = Mock()
        self.calls.attach_mock(self.analytics, 'analytics')
        self.calls.attach_mock(self.ultralytics, 'ultralytics')
        bootstrap = SimpleNamespace(analytics_environment=self.analytics,
                                    ultralytics_environment=self.ultralytics)
        self.mock_module = patch.dict('sys.modules', {'bootstrap': bootstrap})
        self.mock_module.start()
        self.addCleanup(self.mock_module.stop)

    def test_setup_without_config_installs_both_environments_and_skips_data(self):
        with patch('mcmot.pipeline.load_config') as load, patch('mcmot.pipeline.show_status') as status:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(['setup'])
        load.assert_not_called()
        status.assert_not_called()
        self.assertEqual([call[0] for call in self.calls.mock_calls], ['analytics', 'ultralytics'])
        self.ultralytics.assert_called_once_with({'backend': 'mivolo', 'repository': str(ROOT / 'vendor/mivolo')})
        self.assertIn('Setup complete', output.getvalue())

    def test_setup_config_overrides_install_options_without_dataset_or_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump({'attributes': {
                'repository': 'custom/mivolo', 'torch_version': '2.7.1',
                'torchvision_version': '0.22.1',
                'torch_index_url': 'https://download.pytorch.org/whl/cpu',
                'model_dir': 'missing/weights'}}), encoding='utf-8')
            with patch('mcmot.pipeline.show_status') as status, contextlib.redirect_stdout(io.StringIO()):
                main(['setup', '--config', str(config)])
            options = self.ultralytics.call_args.args[0]
            self.assertEqual(options['backend'], 'mivolo')
            self.assertEqual(options['repository'], str(root / 'custom/mivolo'))
            self.assertEqual(options['torch_index_url'], 'https://download.pytorch.org/whl/cpu')
            self.assertEqual(options['model_dir'], str(root / 'missing/weights'))
            self.analytics.assert_called_once_with()
            status.assert_not_called()

    def test_setup_includes_mivolo_with_legacy_config(self):
        with patch('mcmot.pipeline.load_config', return_value={'attributes': {'backend': 'onnx'}}):
            with contextlib.redirect_stdout(io.StringIO()):
                main(['setup', '--config', 'legacy.yaml'])
        self.assertEqual(self.ultralytics.call_args.args[0]['backend'], 'mivolo')

    def test_other_commands_still_require_config_before_setup_or_data_access(self):
        for command in ('status', 'calibrate', 'spatial', 'raw', 'bev', 'validate', 'run', 'pptx', 'all'):
            with self.subTest(command=command):
                with patch('mcmot.pipeline.load_config') as load, contextlib.redirect_stderr(io.StringIO()) as output:
                    with self.assertRaises(SystemExit) as caught:
                        main([command])
                self.assertEqual(caught.exception.code, 2)
                self.assertIn('--config is required', output.getvalue())
                load.assert_not_called()
        self.analytics.assert_not_called()
        self.ultralytics.assert_not_called()


if __name__ == '__main__':
    unittest.main()
