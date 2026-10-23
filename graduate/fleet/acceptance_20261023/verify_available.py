"""Primary independent verification; reads immutable Git snapshots, not worker prose."""
from pathlib import Path
import subprocess,tarfile,io,tempfile,json,importlib.util,os
import numpy as np
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[2]
logs=HERE/'logs';logs.mkdir(exist_ok=True)
w=Path(tempfile.mkdtemp(prefix='kwt-primary-accept-'))
gate='25868b271a99f2ee083898056bcda69d37b9b7d0'
common=w/'common.py';common.write_bytes(subprocess.check_output(['git','-C',str(REPO),'show',f'{gate}:graduate/ds_tasks/r5_gates/model/common.py']))
spec=importlib.util.spec_from_file_location('r5_common_snapshot',common);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
checks=[]
for n,delta in ((1,1e-7),(2,1e-7),(64,1e-7),(2,1.0)):
 ref=np.zeros(n);est=np.full(n,delta)
 got=float(c.nrmse(est,ref)[0]);want=float(np.sqrt(np.mean((est-ref)**2))/max(float(np.sqrt(np.mean(ref**2))),1e-8))
 checks.append(dict(elements=n,error=delta,got=got,expected=want,pass_worker=got<=.0027,pass_correct=want<=.0027,formula_matches=bool(np.isclose(got,want))))
(HERE/'nrms_floor_check.json').write_text(json.dumps(dict(commit=gate,checks=checks,verdict='FAIL: RMS floor applied to unnormalised squared sum'),indent=2)+'\n')
print('NRMSE_FLOOR',json.dumps(checks),flush=True)
ctrl='63a17a6ff0b0d535d60a2b91296bfa821fedaa22';dest=w/'control';dest.mkdir()
blob=subprocess.check_output(['git','-C',str(REPO),'archive',ctrl,'hardware/experiments/r5_control','graduate/ds_tasks/r5_control'])
with tarfile.open(fileobj=io.BytesIO(blob)) as t:t.extractall(dest,filter='data')
(dest/'software').mkdir();(dest/'software/weights').symlink_to(REPO/'software/weights',target_is_directory=True)
cmd=['python3',str(REPO/'graduate/fleet/r6_20261023/heavy.py'),'--','bash',str(dest/'graduate/ds_tasks/r5_control/run.sh')]
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (logs/'control_rerun.log').open('w') as out:
 p=subprocess.run(cmd,cwd=dest,env=env,stdout=out,stderr=subprocess.STDOUT)
r=dict(source_commit=ctrl,command=cmd,exit=p.returncode,clean_archive=str(dest),read_only_external=['software/weights','p027 frozen pack'],scope='R5 16-multiplier serialized FFN1 control only; not target8x8/8x16 optimized result')
result=dest/'graduate/ds_tasks/r5_control/results.json'
if result.exists():
 data=json.loads(result.read_text());r['state']=data.get('state');r['checks']=data.get('checks');r['required_measurement']=data.get('required_measurement');r['fault_injection']=data.get('fault_injection');r['precision']=data.get('precision');r['synthesis']=data.get('synthesis')
(HERE/'control_rerun.json').write_text(json.dumps(r,indent=2)+'\n')
print('CONTROL_EXIT',p.returncode,flush=True)
raise SystemExit(p.returncode)
