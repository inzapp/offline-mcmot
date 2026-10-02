import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mcmot.pptx_agent import _run_codex


class PptxProgressTest(unittest.TestCase):
    def run_worker(self, script, timeout=3):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        job = Path(temporary.name)
        output = io.StringIO()
        command = [sys.executable, '-u', '-c', script]
        return job, output, command, timeout

    def test_events_and_heartbeat_are_displayed_and_full_logs_preserved(self):
        script = '''import sys, time
assert sys.stdin.read() == "report prompt"
sys.stdout.write('{"type":"turn.')
sys.stdout.flush()
time.sleep(.08)
print('started"}')
print('{"type":"item.started","item":{"type":"command_execution","command":"build_report.py"}}')
time.sleep(.1)
print('{"type":"turn.completed"}')
print('diagnostic', file=sys.stderr)
'''
        job, output, command, timeout = self.run_worker(script)
        with contextlib.redirect_stdout(output):
            _run_codex(command, 'report prompt', job, timeout, heartbeat_seconds=.03, poll_seconds=.01)
        text = output.getvalue()
        self.assertIn('Generation started', text)
        self.assertIn('Still generating...', text)
        self.assertIn('Running command: build_report.py', text)
        self.assertIn('Codex report generation finished', text)
        events = [json.loads(line) for line in (job / 'events.jsonl').read_text().splitlines()]
        self.assertEqual(events[0]['type'], 'turn.started')
        self.assertEqual(len(events), 3)
        self.assertIn('diagnostic', (job / 'stderr.log').read_text())

    def test_failed_worker_reports_error_and_raises(self):
        job, output, command, timeout = self.run_worker('import sys; sys.stdin.read(); sys.exit(2)')
        with contextlib.redirect_stdout(output), self.assertRaises(subprocess.CalledProcessError) as caught:
            _run_codex(command, '', job, timeout, poll_seconds=.01)
        self.assertEqual(caught.exception.returncode, 2)
        self.assertIn('Generation failed', output.getvalue())

    def test_timeout_stops_worker_and_preserves_logs(self):
        job, output, command, timeout = self.run_worker('import sys,time; sys.stdin.read(); time.sleep(10)', timeout=.15)
        with contextlib.redirect_stdout(output), self.assertRaises(subprocess.TimeoutExpired):
            _run_codex(command, '', job, timeout, poll_seconds=.01)
        self.assertIn('Generation timed out', output.getvalue())
        self.assertTrue((job / 'events.jsonl').is_file())
