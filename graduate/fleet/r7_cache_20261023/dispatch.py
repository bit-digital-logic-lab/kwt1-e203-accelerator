#!/usr/bin/env python3
"""DeepSeek cache experiments; numerical preflight precedes cache RTL branches.

No worker branch is merged, pushed, or marked as accepted by this dispatcher.
"""
import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
ROOT=Path('/home/yang/kwt-worktrees')
BASE='534fe5af931cc486e3bf724ae410a9705db9310a'
MODEL='opencode-go/deepseek-v4.1-flash'
SERVICE='kwt-ds-r7-cache-20261023.service'
CHECK=PROJECT/'.opencode/skills/quarter-hour-check/scripts/check.py'
TASKS=json.loads((HERE/'tasks.json').read_text())
POST=('w_pages','resident_w','resident_both','fifo','cacc','packing','macro','array8x8')
LOCK=threading.RLock()
PREPARE_LOCK=threading.Lock()
PROCESSES={}
STATE=dict(state='not_started',model=MODEL,source_base=BASE,tasks={})

COMMON="""You are local OpenCode Go deepseek-v4.1-flash. Do not start subagents or othermodelworkers. ReadAGENTS.md, competition题目/README.md andoriginalscreenshots. Workonlyyourworktree, do noteditotherworktrees, merge/rebase/push. No pruning/retraining/calibrationchange; frozenA16W12 parameters. PCaudio/MFCC/projection/classifier remainFP32. Targettrue8x8=64 and8x16=128physicalmultipliers, nothistorical16mult rowbroadcast. BiasandaccumulationcentralCACC. Cacheclearedperofficialcoldtask, loadingthroughlastwritecounts, nohiddenpreload. PrivateSRAM16bitx128/1024/4096SP/DP, shared512KB isseparate; accelerator+interface+privatemacros<=2.25mm2. Actualtargetmacroareas/PDK/officialSoC unavailable: neverclaimarea/timing/powerclosure. Read main graduate/fleet/r7_cache_20261023/README.md, capacity.json, and acceptance_20261023/recheck/README.md. Automaticgate ispreflightonly.
Your task deliverables: graduate/ds_tasks/r7_TASK/run.sh, results.json, WORKER_REPORT.md plus experimentcode inhardware/experiments/r7_TASK whenapplicable. Record source/basecommit, commands+realexitcodes, hashes, numericcontract, macrocapacityversuslogicregisters, limitations, actualtestcounts. Produce results eveniffailed. Newcode mustbeselfcontainedexceptexplicitimmutabledata/symlinks. NoemptyPASS orbackgroundteststhatoutliveworker withoutdurabletrackedservice. Runactualtests tocompletionthencommitallowncode/results andleavecleanworktree; no buildbinaries/cachevectorsdumps inGit. AllCPU-heavybuild/sim/synth through python3 /home/yang/Desktop/ic设计竞赛/graduate/fleet/r6_20261023/heavy.py -- commandargs, jobs<=2, OPENBLAS/OMP1; do notnestedlockwrapperandworkerchildwrapperor deadlock. Modelsconcurrent, onlyheavytoolsqueued. Shortfinalreport. Otherresultbranches arenotacceptedbecauseworkerprintedPASS. Frozenpackread-only:/home/yang/kwt-worktrees/ds-p027-precision/graduate/ds_tasks/p027_precision. Originalfixturedataread-only:/home/yang/kwt-worktrees/ds-r6-array8x16/graduate/ds_tasks/r6_array8x16/fixtures. Keepfullwidthbaselineunchanged inpostfixexperiments; copytoownnewexperiment. Compareequalarithmetic/equaloutputformats, reportbothbus64/128andallcoldbytes. Primaryreviewpending.\n"""


def now():return datetime.now(timezone.utc).isoformat()


def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()


def save():
    with LOCK:
        temp=HERE/'STATUS.tmp'
        temp.write_text(json.dumps(STATE,ensure_ascii=False,indent=2)+'\n')
        temp.replace(HERE/'STATUS.json')


def hook(name,state,result,note,pid=None):
    cmd=['python3',str(CHECK),'update','--id',f'kwt-ds-r7-{name}',
         '--state',state,'--result',str(result),'--note',note]
    if name=='dispatcher':cmd+=['--service',SERVICE]
    if pid:cmd+=['--pid',str(pid)]
    subprocess.run(cmd,cwd=PROJECT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def prepare(name,base):
    path=ROOT/f'ds-r7-{name}'
    subprocess.run(['git','-C',str(PROJECT),'worktree','add','-b',f'codex/kwt-ds-r7-{name}',str(path),base],check=True)
    for rel in ('资料','software/weights','software/.build','software/.venv'):
        target=path/rel
        if not target.exists() and not target.is_symlink() and (PROJECT/rel).exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            target.symlink_to(PROJECT/rel,target_is_directory=True)
    return path


def worker(name,base):
    report=HERE/'status'/f'{name}.json'
    log=HERE/'logs'/f'{name}.log'
    record=dict(task=name,base_commit=base,started_at=now(),state='starting',submitted=False)
    code=125
    try:
        with PREPARE_LOCK:path=prepare(name,base)
        result=path/f'graduate/ds_tasks/r7_{name}/results.json'
        prompt=COMMON.replace('TASK',name)+'\nRead experiment plan at absolute path '+str(HERE/'README.md')+' and '+str(HERE/'capacity.json')+'.\nTASK '+name+'\n'+TASKS[name]+'\nBASE_COMMIT='+base+'\n'
        (HERE/'prompts'/f'{name}.md').write_text(prompt)
        record.update(worktree=str(path),result=str(result),log=str(log),state='running')
        with LOCK:STATE['tasks'][name]=record
        save()
        env=os.environ.copy();env.update(PWD=str(path),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with log.open('w') as stream:
            proc=subprocess.Popen([shutil.which('opencode') or '/home/yang/.nvm/versions/node/v24.18.0/bin/opencode',
                'run','--standalone','--auto','--model',MODEL,prompt],cwd=path,env=env,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            with LOCK:PROCESSES[name]=proc
            hook(name,'running',report,f'DeepSeek running; log={log}',proc.pid)
            code=proc.wait()
            with LOCK:PROCESSES.pop(name,None)
        head=git(path,'rev-parse','HEAD');dirty=git(path,'status','--porcelain')
        descends=subprocess.run(['git','-C',str(path),'merge-base','--is-ancestor',base,head]).returncode==0
        record.update(exit_code=code,head=head,dirty_status=dirty,
            results_exists=result.is_file(),report_exists=result.with_name('WORKER_REPORT.md').is_file(),
            run_exists=result.with_name('run.sh').is_file(),descends_from_base=descends)
        # Submission integrity, deliberately not an assertion that tests passed.
        record['submitted']=bool(code==0 and head!=base and not dirty and descends and
            record['results_exists'] and record['report_exists'] and record['run_exists'])
        record['state']='submitted_not_accepted' if record['submitted'] else 'needs_review'
    except BaseException as exc:
        record.update(state='dispatch_failed',exit_code=code,error=f'{type(exc).__name__}: {exc}')
    record['finished_at']=now()
    report.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    with LOCK:STATE['tasks'][name]=record
    save()
    hook(name,'complete' if record['submitted'] else 'failed',report,
         f"exit={record.get('exit_code')} submitted={record['submitted']}; primary review pending")
    return record


def terminate(signum,frame):
    with LOCK:
        for proc in PROCESSES.values():
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    raise SystemExit(128+signum)


def main():
    for d in ('logs','status','prompts'):(HERE/d).mkdir(exist_ok=True)
    if (HERE/'STATUS.json').exists():raise SystemExit('Existing run: inspect status; do not silently replay')
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,terminate)
    STATE.update(state='running',started_at=now());save()
    hook('dispatcher','running',HERE/'STATUS.json','fullwidth repair + capacity/trace; preflight then8cache tasks',os.getpid())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(TASKS)) as pool:
            early=[pool.submit(worker,n,BASE) for n in ('capacity','trace')]
            fixed=pool.submit(worker,'fullwidth',BASE).result()
            if not fixed['submitted']:raise RuntimeError('Fullwidth worker not submitted; dependent RTL tasks blocked')
            hook('dispatcher','running',HERE/'STATUS.json','primary-authored fullwidth numerical preflight',os.getpid())
            with (HERE/'logs/fullwidth_gate.log').open('w') as stream:
                gate=subprocess.run(['python3',str(HERE/'gate_fullwidth.py'),fixed['worktree'],fixed['head']],
                                    stdout=stream,stderr=subprocess.STDOUT)
            STATE['fullwidth_preflight_exit']=gate.returncode;save()
            if gate.returncode:raise RuntimeError('Fullwidth preflight failed; dependent cache RTL tasks blocked')
            STATE['cache_base_commit']=fixed['head'];save()
            later=[pool.submit(worker,n,fixed['head']) for n in POST]
            for future in concurrent.futures.as_completed(early+later):
                r=future.result();print(json.dumps(r,ensure_ascii=False),flush=True)
        worker('crosscheck',STATE['cache_base_commit'])
        STATE.update(state='needs_primary_review',finished_at=now());save()
        bad=[n for n,r in STATE['tasks'].items() if not r.get('submitted')]
        hook('dispatcher','failed' if bad else 'complete',HERE/'STATUS.json',
             'Finished; primary acceptance pending; incomplete='+','.join(bad))
    except BaseException as exc:
        STATE.update(state='failed',finished_at=now(),error=f'{type(exc).__name__}: {exc}');save()
        hook('dispatcher','failed',HERE/'STATUS.json',STATE['error']);raise


if __name__=='__main__':main()
