"""Feed a prepared prompt and local report artifacts to non-interactive Codex."""
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile


def _event_message(event):
    kind = event.get('type')
    if kind == 'thread.started':
        return 'Codex session started'
    if kind == 'turn.started':
        return 'Codex is preparing the report'
    item = event.get('item', {})
    if kind == 'item.started' and item.get('type') == 'command_execution':
        command = ' '.join(item.get('command', '').split())
        return 'Running command: ' + command[:140] + ('...' if len(command) > 140 else '')
    if kind == 'item.completed':
        if item.get('type') == 'agent_message':
            message = ' '.join(item.get('text', '').split())
            return message[:180] + ('...' if len(message) > 180 else '')
        if item.get('type') == 'file_change':
            return 'Report files updated'
        if item.get('type') == 'command_execution':
            return f'Command finished (exit code: {item.get("exit_code")})'
    if kind == 'turn.completed':
        return 'Codex report generation finished'
    if kind in ('error', 'turn.failed'):
        return 'Codex reported an error; see events.jsonl and stderr.log'
    return None


def _run_codex(command, prompt, job, timeout, heartbeat_seconds=30, poll_seconds=.1):
    """Keep full event logs on disk and display progress even during silent periods."""
    started = time.monotonic()
    last_heartbeat = started
    activity = 'Waiting for Codex'
    event_path = job / 'events.jsonl'
    print('PPT: Generation started (Codex CLI)', flush=True)
    print(f'PPT: Progress logs: {event_path} | {job / "stderr.log"}', flush=True)
    with event_path.open('w') as events, (job / 'stderr.log').open('w') as errors:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=events, stderr=errors, text=True) as process:
            try:
                process.stdin.write(prompt)
                process.stdin.close()
                with event_path.open() as reader:
                    def read_events():
                        nonlocal activity
                        while True:
                            position = reader.tell()
                            line = reader.readline()
                            if not line.endswith('\n'):
                                reader.seek(position)
                                break
                            try:
                                message = _event_message(json.loads(line))
                            except (ValueError, TypeError, AttributeError):
                                continue
                            if message:
                                activity = message
                                print(f'PPT [{time.monotonic() - started:.0f}s]: {message}', flush=True)

                    while process.poll() is None:
                        read_events()
                        now = time.monotonic()
                        if now - started >= timeout:
                            print(f'PPT: Generation timed out after {timeout}s; logs: {job}', flush=True)
                            raise subprocess.TimeoutExpired(command, timeout)
                        if now - last_heartbeat >= heartbeat_seconds:
                            print(f'PPT [{now - started:.0f}s]: Still generating... Last activity: {activity}', flush=True)
                            last_heartbeat = now
                        time.sleep(min(poll_seconds, max(0, timeout - (now - started))))
                    read_events()
                if process.returncode:
                    print(f'PPT: Generation failed; inspect {event_path} and {job / "stderr.log"}', flush=True)
                    raise subprocess.CalledProcessError(process.returncode, command)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()


def generate(cfg):
    settings = cfg.get('pptx', {})
    template = Path(settings.get('template', ''))
    prompt = Path(settings.get('prompt', ''))
    if not template.is_file() or not prompt.is_file():
        raise ValueError('pptx.template 및 pptx.prompt 파일을 YAML에 지정하세요')
    if shutil.which('codex') is None:
        raise RuntimeError('Codex CLI가 설치되어 있지 않습니다')
    root = Path(__file__).resolve().parents[1]
    output_root = Path(cfg['output_root'])
    report = output_root / 'report/index.html'
    if not report.exists():
        raise FileNotFoundError(f'분석 보고서 없음: {report}')
    destination = Path(settings.get('output') or output_root / 'report' / f'{cfg["site"]}.pptx')
    if destination.exists():
        raise FileExistsError(f'PPT가 이미 존재합니다. 새 출력 경로를 지정하세요: {destination}')
    job = output_root / 'report/pptx_job'
    job.mkdir(parents=True, exist_ok=True)
    staged_template = job / 'template.pptx'
    artifact = job / 'generated.pptx'
    shutil.copy2(template, staged_template)
    if artifact.exists():
        raise FileExistsError(f'이전 PPT 작업 결과가 존재합니다: {artifact}')
    text = prompt.read_text(encoding='utf-8') + '\n\n' + json.dumps({
        'template': str(staged_template.resolve()), 'report': str(report.resolve()),
        'global_persons': str((output_root / 'global/persons.csv').resolve()),
        'spatial_summary': str((output_root / 'spatial/demographics_summary.csv').resolve()),
        'statistics': str((output_root / 'attributes/statistics').resolve()),
        'output_pptx': str(artifact.resolve()), 'python': str(root / '.venv/bin/python'),
        'plot_python': str(root / '.venv_ultralytics/bin/python')}, ensure_ascii=False, indent=2)
    (job / 'prompt.txt').write_text(text, encoding='utf-8')
    command = ['codex', 'exec', '--ephemeral', '--sandbox', 'workspace-write', '--json', '-C', str(root),
               '-o', str((job / 'final_message.txt').resolve())]
    if not output_root.resolve().is_relative_to(root):
        command += ['--add-dir', str(output_root.resolve())]
    if settings.get('model'):
        command += ['--model', settings['model']]
    command += ['-']
    print(f'PPT: Template: {template}', flush=True)
    print(f'PPT: Output: {destination}', flush=True)
    _run_codex(command, text, job, settings.get('timeout_seconds', 1800))
    print('PPT: Validating generated PPTX file', flush=True)
    with zipfile.ZipFile(artifact) as archive:
        if archive.testzip() or 'ppt/presentation.xml' not in archive.namelist():
            raise RuntimeError('생성된 PPTX 형식 검증 실패')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact, destination)
    print(f'PPT 생성: {destination}', flush=True)
