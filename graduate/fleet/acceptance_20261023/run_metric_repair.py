from pathlib import Path
import subprocess,json,os
h=Path(__file__).resolve().parent;r=h.parents[2];w=Path('/home/yang/kwt-worktrees/ds-r5-metric-floor');check=r/'.opencode/skills/quarter-hour-check/scripts/check.py';result=h/'metric_repair_status.json'
def hook(s,n,pid=None):
 cmd=['python3',str(check),'update','--id','ds-r5-metric-floor','--state',s,'--result',str(result),'--service','kwt-ds-metric-floor-20261023.service','--note',n]
 if pid:cmd+=['--pid',str(pid)]
 subprocess.run(cmd,stdout=subprocess.DEVNULL)
try:
 with (h/'logs/metric_repair.log').open('w') as log:
  p=subprocess.Popen(['opencode','run','--standalone','--auto','--model','opencode-go/deepseek-v4.1-flash',(h/'metric_floor_prompt.md').read_text()],cwd=w,stdout=log,stderr=subprocess.STDOUT)
  hook('running','repair confirmed RMS denominator bug',p.pid);code=p.wait()
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip()
 result.write_text(json.dumps(dict(exit=code,head=head,result=str(w/'graduate/ds_tasks/r5_metric_floor/results.json'),primary_acceptance='pending'),indent=2)+'\n')
 hook('complete' if code==0 else 'failed',f'worker exit={code}, primary acceptance pending')
except BaseException as e:
 result.write_text(json.dumps(dict(error=str(e)))+'\n');hook('failed',str(e));raise
