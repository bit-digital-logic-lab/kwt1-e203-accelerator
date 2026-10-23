#!/usr/bin/env python3
"""Bounded DeepSeek queue; logs real exits and Git state, never merges."""

import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import threading


HERE = Path(__file__).resolve().parent
ROOT = Path('/home/yang/kwt-worktrees')
PROJECT = Path('/home/yang/Desktop/ic设计竞赛')
CHECK = PROJECT / '.opencode/skills/quarter-hour-check/scripts/check.py'
MODEL = 'opencode-go/deepseek-v4.1-flash'
MAX_SECONDS = 10800
TASKS = ('traffic', 'bus64', 'bus128', 'resident_a', 'resident_w', 'output', 'layout', 'tail', 'array8x8', 'array8x16', 'feed8x8', 'feed8x16')
REPORTS = {name: f'graduate/ds_tasks/r6_{name}/results.json' for name in (*TASKS, 'integration', 'crosscheck')}

lock = threading.Lock()
state = {'model': MODEL, 'started_at': None, 'finished_at': None,
         'state': 'not_started', 'tasks': {}}


def now():
    return datetime.now(timezone.utc).isoformat()


def git(path, *args):
    result = subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else f'ERROR({result.returncode}): {result.stderr.strip()}'


def save():
    with lock:
        temp = HERE / 'STATUS.tmp'
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temp.replace(HERE / 'STATUS.json')


def hook(name, status, result, note, pid=None):
    cmd = ['python3', str(CHECK), 'update', '--id', f'kwt-ds-r6-{name}',
           '--state', status, '--result', str(result), '--note', note]
    if name == 'dispatcher':
        cmd += ['--service', 'kwt-ds-r6-20261023.service']
    if pid is not None:
        cmd += ['--pid', str(pid)]
    subprocess.run(cmd, cwd=PROJECT, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=False)


def worker(name):
    path = ROOT / f'ds-r6-{name}'
    prompt = (HERE / 'prompts' / f'{name}.md').read_text()
    log = HERE / 'logs' / f'{name}.log'
    result = HERE / 'status' / f'{name}.json'
    log.parent.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)
    start_head = git(path, 'rev-parse', 'HEAD')
    record = {'task': name, 'worktree': str(path), 'branch': git(path, 'branch', '--show-current'),
              'start_head': start_head, 'started_at': now(), 'log': str(log),
              'result': str(path / REPORTS[name]), 'state': 'running'}
    with lock:
        state['tasks'][name] = record
    save()
    with log.open('w') as output:
        env = os.environ.copy()
        env['PWD'] = str(path)
        try:
            process = subprocess.Popen(
                ['opencode', 'run', '--standalone', '--auto', '--model', MODEL, prompt],
                cwd=path, stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.STDOUT, env=env, start_new_session=True)
            hook(name, 'running', result, f'OpenCode running; log={log}', process.pid)
            try:
                code = process.wait()
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
                output.write('\nWORKER_TIMEOUT\n')
        except Exception as exc:
            code = 125
            output.write(f'\nDISPATCH_EXCEPTION {type(exc).__name__}: {exc}\n')
    record.update(finished_at=now(), exit_code=code,
                  head=git(path, 'rev-parse', 'HEAD'),
                  dirty_status=git(path, 'status', '--porcelain'),
                  report_exists=(path / 'WORKER_REPORT.md').is_file(),
                  results_exists=(path / REPORTS[name]).is_file())
    record['report_exists'] = record['report_exists'] or (path / REPORTS[name]).with_name('WORKER_REPORT.md').is_file()
    record['new_commit'] = record['head'] != start_head
    record['clean'] = not bool(record['dirty_status'])
    record['submitted'] = bool(code == 0 and record['new_commit'] and
                               record['clean'] and record['results_exists'] and record['report_exists'])
    record['state'] = 'submitted' if record['submitted'] else 'needs_review'
    result.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    with lock:
        state['tasks'][name] = record
    save()
    hook(name, 'complete' if record['submitted'] else 'failed', result,
         f'exit={code} commit={record["new_commit"]} clean={record["clean"]} results={record["results_exists"]}')
    return record


def main():
    state['started_at'] = now()
    state['state'] = 'running'
    save()
    hook('dispatcher', 'running', HERE / 'STATUS.json', 'twelve concurrent experiments then integration and independent crosscheck', os.getpid())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(TASKS)) as pool:
            futures = [pool.submit(worker, name) for name in TASKS]
            for future in concurrent.futures.as_completed(futures):
                print(json.dumps(future.result(), ensure_ascii=False), flush=True)
        print('FIRST_WAVE_COMPLETE, START_INTEGRATION', flush=True)
        print(json.dumps(worker('integration'), ensure_ascii=False), flush=True)
        print(json.dumps(worker('crosscheck'), ensure_ascii=False), flush=True)
        state['finished_at'] = now()
        state['state'] = 'needs_primary_review'
        save()
        hook('dispatcher', 'complete', HERE / 'STATUS.json', 'queue finished; primary review still required')
    except BaseException as exc:
        state['finished_at'] = now()
        state['state'] = 'failed'
        state['error'] = f'{type(exc).__name__}: {exc}'
        save()
        hook('dispatcher', 'failed', HERE / 'STATUS.json', state['error'])
        raise


if __name__ == '__main__':
    main()
