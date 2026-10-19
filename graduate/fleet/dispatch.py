#!/usr/bin/env python3
"""Bounded, resumable OpenCode Go DeepSeek fleet for graduate KWT-1 work."""
import argparse
import concurrent.futures
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/home/yang/Desktop/ic设计竞赛')
ROOT = Path('/home/yang/kwt-worktrees')
STATE = ROOT / 'ds-fleet-20261019'
MODEL = 'opencode-go/deepseek-v4.1-flash'
BASE = 'codex/kwt-grad-encoder'
CHECK = PROJECT / '.opencode/skills/quarter-hour-check/scripts/check.py'
PREPARE_LOCK = threading.Lock()

# Each job has its own branch and worktree; workers never edit a peer's checkout.
TASKS = [
 ('spec', '审核题目截图第4-7项与当前研究生实现的覆盖矩阵；只写 graduate/ds_tasks/spec。'),
 ('dataset', '核查训练/验证划分、哈希、样本重叠和现有校准泄漏；只写 graduate/ds_tasks/dataset。'),
 ('attn_scale', '修复注意力从验证样本求 sx/fit_shift 的问题：用训练划分冻结尺度，独立验证多样本；只改 graduate/encoder/model/attn_int_model.py 及 graduate/ds_tasks/attn_scale。'),
 ('postnorm_scale', '修复 postnorm 从验证 sample_00 取中间张量尺度的问题：训练集冻结尺度与 LUT，验证集只评估；只改 graduate/encoder/model/kwt_postnorm_ref.py 及 graduate/ds_tasks/postnorm_scale。'),
 ('layer_params', '为12层生成冻结整数权重/偏置/尺度 manifest，训练集标定、验证集测试，不改原模型；只写 graduate/ds_tasks/layer_params。'),
 ('gelu_lut', '独立评估 GELU LUT 的训练集构造、误差界、验证集误差与硬件 ROM 大小；只写 graduate/ds_tasks/gelu_lut。'),
 ('attn_samples', '独立复跑注意力 RTL 多个真实验证样本及极值测试，保留逐位与FP32双指标；只写 graduate/ds_tasks/attn_samples。'),
 ('nonlinear_samples', '独立复跑 GELU/FFN2/LayerNorm RTL 多样本与近零方差边界；只写 graduate/ds_tasks/nonlinear_samples。'),
 ('encoder_gate', '修复 verify_encoder.py 只打印 NRMSE 却未在超3%时失败的门禁；加负例注入；只改 graduate/encoder/model/verify_encoder.py 与 graduate/ds_tasks/encoder_gate。'),
 ('meta_determinism', '修复 grad-encoder ACCEPTANCE.json 的 meta.json 哈希不可复现；从干净 Git 树独立复跑；只改 graduate/encoder/vectors/gen_vectors.py、ACCEPTANCE.json 与 graduate/ds_tasks/meta_determinism。'),
 ('encoder_12numeric', '设计并实际运行12层不同权重的数值链独立参考/样本测试，不能把调度smoke称数值通过；只写 graduate/ds_tasks/encoder_12numeric。'),
 ('encoder_transport', '验证12层参数逐层切换与层间99x64数据搬运，故障注入防止重复layer0权重蒙混；只写 graduate/ds_tasks/encoder_transport。'),
 ('head_boundary', '验证12层最终输出到题目规定的软件分类头的边界与标签，至少真实样本smoke；只写 graduate/ds_tasks/head_boundary。'),
 ('runner_paths', '修复研究生分阶段 runner 对新 graduate/rtl 路径误判 SKIP 的问题；只改 graduate/verification/ 与 graduate/ds_tasks/runner_paths。'),
 ('negative_tests', '为逐位/NRMSE/12层换参门禁做独立故障注入，证明测试能检测错误；只写 graduate/ds_tasks/negative_tests。'),
 ('e203_iface', '分析研究生整编码器 E203/NICE 或总线真实接口、驱动缺口；可新增独立可运行模型，不冒充官方SoC；只写 graduate/ds_tasks/e203_iface。'),
 ('sram_budget', '量化注意力99x99、QKV、FFN中间值和双缓冲在题目指定SRAM宏的容量端口开销；只写 graduate/ds_tasks/sram_budget。'),
 ('synth', '使用可用开源综合工具对研究生 RTL 做可复现结构/面积/时序初查；没有目标PDK则明确MISSING；只写 graduate/ds_tasks/synth。'),
 ('activity', '生成研究生 RTL 真实仿真的 VCD/活动统计和边界说明，不能称实际功耗；只写 graduate/ds_tasks/activity。'),
 ('cycles', '测实跑注意力/单层/12层的阶段周期、吞吐与测试口径；不得把计算段误作任务总周期；只写 graduate/ds_tasks/cycles。'),
 ('traffic', '量化参数装载、SRAM读写、层间搬运字节和总线下界；只写 graduate/ds_tasks/traffic。'),
 ('metrics', '建立研究生第7项同一冻结配置的精度/周期/面积/功耗证据矩阵，MISSING明确列出；只写 graduate/ds_tasks/metrics。'),
 ('golden', '独立核对官方TF/NumPy参考的注意力与12层中间张量，多真实样本，保留hash/误差；只写 graduate/ds_tasks/golden。'),
 ('dispatch_review', '审核并改进 OpenCode Go DeepSeek 调度器的状态、依赖、超时、退出码和hook，写可运行单元测试；只写 graduate/ds_tasks/dispatch_review。'),
 ('final_audit', '在前24项结束后独立审阅所有分支/日志/实际测试，形成研究生第4-7项验收矩阵与阻塞清单，至少复跑两项测试；只写 graduate/ds_tasks/final_audit。'),
]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def call(cmd, **kw):
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def checkin(state, note):
    call([sys.executable, str(CHECK), 'update', '--id', 'kwt-grad-ds-fleet',
          '--state', state, '--service', 'kwt-grad-ds-fleet.service',
          '--result', str(STATE / 'summary.json'), '--note', note], cwd=PROJECT)


def prepare(task_id):
    with PREPARE_LOCK:
        branch = f'codex/kwt-ds-grad-{task_id}'
        worktree = ROOT / f'ds-grad-{task_id}'
        if not worktree.exists():
            r = call(['git', '-C', str(PROJECT), 'worktree', 'add', '-b', branch,
                      str(worktree), BASE])
            if r.returncode != 0:
                raise RuntimeError(f'{task_id}: worktree creation failed: {r.stderr}')
        actual = call(['git', '-C', str(worktree), 'branch', '--show-current'])
        if actual.stdout.strip() != branch:
            raise RuntimeError(f'{task_id}: branch mismatch: {actual.stdout.strip()}')
        return branch, worktree


def run_one(task):
    task_id, scope = task
    branch, worktree = prepare(task_id)
    path = STATE / 'status' / f'{task_id}.json'
    log = STATE / 'logs' / f'{task_id}.log'
    prior = json.loads(path.read_text()) if path.exists() else None
    if prior and prior.get('terminal'):
        return prior
    before = call(['git', '-C', str(worktree), 'rev-parse', 'HEAD']).stdout.strip()
    status = {'task': task_id, 'branch': branch, 'worktree': str(worktree),
              'base_head': before, 'started_at': stamp(), 'state': 'running',
              'terminal': False, 'log': str(log)}
    write_json(path, status)
    prompt = (f'你是 OpenCode Go DeepSeek 独立工作者，任务 {task_id}。只在当前独立 worktree 工作；'
              '禁止开子智能体、改其他 worktree、merge/rebase/push。先读 AGENTS.md、题目原截图和现有研究生报告。'
              f'具体任务：{scope} '
              '必须真实运行相关命令并保存退出码/日志/样本数；软件参考、RTL逐位、E203、综合、官方验收分别标证据层级。'
              '训练划分可做无梯度标定，验证划分仅评估；不得用验证样本求尺度/阈值。'
              '根目录 WORKER_REPORT.md 写三段：结果、复现命令与退出码、未证事项。'
              '只 git add 本任务文件并 commit，保持工作树干净。输出最多一行 commit 与报告路径，然后退出。'
              '可只读使用主工程 /home/yang/Desktop/ic设计竞赛 的 gitignored 数据，不得将大数据、密钥或缓存提交。')
    cmd = ['opencode', 'run', '--standalone', '--auto', '--model', MODEL, prompt]
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open('w') as f:
        try:
            result = subprocess.run(cmd, cwd=worktree, stdin=subprocess.DEVNULL,
                                    stdout=f, stderr=subprocess.STDOUT, timeout=2700)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
            f.write('\nTIMEOUT 2700 seconds\n')
    after = call(['git', '-C', str(worktree), 'rev-parse', 'HEAD']).stdout.strip()
    dirty = call(['git', '-C', str(worktree), 'status', '--porcelain']).stdout.strip()
    report = worktree / 'WORKER_REPORT.md'
    report_changed = bool(call(['git', '-C', str(worktree), 'diff', '--name-only',
                                f'{before}..{after}', '--', 'WORKER_REPORT.md']).stdout.strip())
    status.update({'finished_at': stamp(), 'elapsed_seconds': round(time.monotonic()-started, 2),
                   'exit_code': code, 'head': after, 'new_commit': after != before,
                   'clean': not bool(dirty), 'report_exists': report.is_file(),
                   'report_changed': report_changed,
                   'state': 'submitted' if code == 0 and after != before and not dirty and report.is_file() and report_changed else 'failed',
                   'terminal': True})
    write_json(path, status)
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parallel', type=int, default=6)
    a = p.parse_args()
    if not 1 <= a.parallel <= 8:
        p.error('--parallel must be 1..8')
    STATE.mkdir(parents=True, exist_ok=True)
    lock = (STATE / '.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('fleet already running')
    write_json(STATE / 'summary.json', {'state':'running', 'started_at':stamp(),
                                       'model':MODEL, 'task_count':len(TASKS), 'parallel':a.parallel})
    checkin('running', f'{len(TASKS)} DeepSeek tasks queued; max parallel={a.parallel}')
    results = []
    try:
        # The final auditor runs after all 24 producers terminate.
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.parallel) as pool:
            futures = {pool.submit(run_one, task): task[0] for task in TASKS[:-1]}
            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    failure = {'task': task_id, 'state':'failed', 'terminal':True,
                               'error':f'{type(exc).__name__}: {exc}', 'finished_at':stamp()}
                    write_json(STATE / 'status' / f'{task_id}.json', failure)
                    results.append(failure)
                write_json(STATE / 'summary.json', {'state':'running', 'updated_at':stamp(),
                           'model':MODEL, 'task_count':len(TASKS), 'finished':len(results),
                           'submitted':sum(x['state']=='submitted' for x in results),
                           'failed':sum(x['state']=='failed' for x in results)})
        results.append(run_one(TASKS[-1]))
    finally:
        final = {'state':'complete', 'updated_at':stamp(), 'model':MODEL,
                 'task_count':len(TASKS), 'finished':len(results),
                 'submitted':sum(x['state']=='submitted' for x in results),
                 'failed':sum(x['state']=='failed' for x in results),
                 'results':[{k:x.get(k) for k in ('task','state','head','exit_code','error')} for x in results]}
        write_json(STATE / 'summary.json', final)
        checkin('complete' if final['failed']==0 and final['finished']==len(TASKS) else 'failed',
                f"fleet submitted={final['submitted']} failed={final['failed']}")
    return 0 if final['failed']==0 and final['finished']==len(TASKS) else 1


if __name__ == '__main__':
    sys.exit(main())
