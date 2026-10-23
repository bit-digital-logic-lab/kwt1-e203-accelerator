from pathlib import Path
import os,subprocess,json
from run import snapshot,HERE,REPO,logs
pin='005f656d8c';p=snapshot(pin,['software','graduate/ds_tasks/r5_software','资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1'])
log=logs/'software';log.mkdir(exist_ok=True);res=HERE/'software_results.json'
env=os.environ.copy();env.update(R5_PYTHON=str(REPO/'software/.venv/bin/python'),R5_DATASET_ARCHIVE=str(REPO/'资料/数据集/speech_commands_v0.02.tar.gz'),R5_CACHE=str(REPO/'software/.build/quant_samples_12labels_40.npz'),R5_P027_PACK='/home/yang/kwt-worktrees/ds-p027-precision/graduate/ds_tasks/p027_precision/frozen',R5_RESULTS=str(res),R5_LOG_DIR=str(log),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
cmd=['python3',str(REPO/'graduate/fleet/r6_20261023/heavy.py'),'--','bash',str(p/'graduate/ds_tasks/r5_software/run_clean.sh')]
with (logs/'software_run.log').open('w') as f:r=subprocess.run(cmd,cwd=p,env=env,stdout=f,stderr=subprocess.STDOUT)
(HERE/'software_acceptance.json').write_text(json.dumps(dict(commit=pin,archive=str(p),argv=cmd,exit=r.returncode,dependency_mounts={k:v for k,v in env.items() if k.startswith('R5_')},results=str(res)),indent=2)+'\n')
print('SOFTWARE_EXIT',r.returncode,flush=True)
raise SystemExit(r.returncode)
