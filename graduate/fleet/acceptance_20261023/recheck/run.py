"""Verify completed worker artifacts from immutable snapshots."""
from pathlib import Path
import subprocess,tarfile,io,tempfile,json,os,importlib.util,concurrent.futures
import numpy as np
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];logs=HERE/'logs';logs.mkdir(exist_ok=True)
def snapshot(commit,paths):
 p=Path(tempfile.mkdtemp(prefix='kwt-recheck-'))
 raw=subprocess.check_output(['git','-C',str(REPO),'archive',commit,*paths])
 with tarfile.open(fileobj=io.BytesIO(raw)) as t:t.extractall(p,filter='data')
 return p

def metric():
 pin='455f7f2ef3c3f94f8536e1e93e72d087ec38ad7e';p=snapshot(pin,['graduate/ds_tasks/r5_gates','graduate/ds_tasks/r4_rtl','graduate/ds_tasks/p027_precision','graduate/ds_tasks/r5_metric_floor'])
 vec=p/'graduate/ds_tasks/r4_rtl/vectors'
 if vec.exists():raise RuntimeError('unexpected raw vectors in archive')
 vec.symlink_to('/home/yang/kwt-worktrees/ds-r5-gates/graduate/ds_tasks/r4_rtl/vectors',target_is_directory=True)
 task=p/'graduate/ds_tasks/r5_gates';(task/'logs').mkdir(exist_ok=True)
 spec=importlib.util.spec_from_file_location('fixed_metric',task/'model/common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
 rng=np.random.default_rng(8);checked=0
 for n in (1,2,64,257):
  for scale in (0.,1e-12,1e-8,1e-4,1.,1e4):
   ref=rng.normal(size=n)*scale;est=ref+rng.normal(size=n)*1e-7
   want=np.sqrt(np.mean((est-ref)**2))/max(np.sqrt(np.mean(ref**2)),1e-8)
   assert np.isclose(c.nrmse(est,ref)[0],want,rtol=1e-12,atol=1e-15);checked+=1
 for est,ref in (([],[]),([1],[1,2]),([float('nan')],[0]),([0],[float('inf')])):
  try:c.nrmse(est,ref)
  except ValueError:checked+=1
  else:raise AssertionError('invalid metric input accepted')
 assert np.isclose(c.nrmse(np.full(2,1e-7),np.zeros(2))[0],10,rtol=1e-14)
 cmds=[]
 for name in ('units','negative_tests'):
  cmd=['python3',str(task/f'model/{name}.py'),'--out-json',str(HERE/f'metric_{name}.json')]
  r=subprocess.run(cmd,cwd=task,capture_output=True,text=True)
  (logs/f'metric_{name}.log').write_text(r.stdout+r.stderr);cmds.append(dict(argv=cmd,exit=r.returncode))
  if r.returncode:raise RuntimeError(f'{name} failed')
 data=dict(commit=pin,clean_archive=str(p),independent_formula_cases=checked+1,counterexample_nrmse=10.,commands=cmds,status='PASS_METRIC_FIX_ONLY')
 (HERE/'metric_acceptance.json').write_text(json.dumps(data,indent=2)+'\n');return data

def banks():
 pin='5812ec231a';p=snapshot(pin,['hardware/experiments/r5_banks','graduate/ds_tasks/r5_banks'])
 work=Path(tempfile.mkdtemp(prefix='kwt-bank-work-'));env=os.environ.copy();env.update(R5_BANKS_WORK=str(work),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 cmd=['python3',str(REPO/'graduate/fleet/r6_20261023/heavy.py'),'--','bash',str(p/'graduate/ds_tasks/r5_banks/run.sh')]
 with (logs/'banks.log').open('w') as f:r=subprocess.run(cmd,cwd=p,env=env,stdout=f,stderr=subprocess.STDOUT)
 f=p/'graduate/ds_tasks/r5_banks/results.json';out=dict(commit=pin,command=cmd,exit=r.returncode,clean_archive=str(p))
 if f.exists():out['results']=json.loads(f.read_text())
 (HERE/'banks_acceptance.json').write_text(json.dumps(out,indent=2)+'\n');return dict(task='banks',exit=r.returncode)

if __name__ == '__main__':
 import sys
 if '--metric-only' in sys.argv:
  print(json.dumps(metric()),flush=True)
 else:
  errors=[]
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(f) for f in (metric,banks)]
   for f in concurrent.futures.as_completed(futures):
    try:print(json.dumps(f.result()),flush=True)
    except Exception as exc:errors.append(str(exc));print('VERIFICATION_ERROR',str(exc),flush=True)
  if errors:raise SystemExit(1)
