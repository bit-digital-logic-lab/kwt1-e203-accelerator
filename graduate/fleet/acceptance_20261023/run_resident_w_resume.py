from pathlib import Path
import subprocess,json,os
h=Path(__file__).resolve().parent;r=h.parents[2];w=Path('/home/yang/kwt-worktrees/ds-r6-resident_w');check=r/'.opencode/skills/quarter-hour-check/scripts/check.py';result=h/'resident_w_resume_status.json'
def hook(s,n,pid=None):
 cmd=['python3',str(check),'update','--id','ds-r6-resident-w-rescue','--state',s,'--result',str(result),'--service','kwt-ds-r6-resident-w-rescue-20261023.service','--note',n]
 if pid:cmd+=['--pid',str(pid)]
 subprocess.run(cmd,stdout=subprocess.DEVNULL)
try:
 with (h/'logs/resident_w_resume.log').open('w') as log:
  p=subprocess.Popen(['opencode','run','--standalone','--auto','--model','opencode-go/deepseek-v4.1-flash',(h/'recheck/resident_w_resume_prompt.md').read_text()],cwd=w,stdout=log,stderr=subprocess.STDOUT)
  hook('running','finish abandoned uncommitted resident_w task',p.pid);code=p.wait()
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip()
 result.write_text(json.dumps(dict(exit=code,head=head,result=str(w/'graduate/ds_tasks/r6_resident_w/results.json'),primary_acceptance='pending'),indent=2)+'\n')
 hook('complete' if code==0 else 'failed',f'worker exit={code}, primary acceptance pending')
except BaseException as e:
 result.write_text(json.dumps(dict(error=str(e)))+'\n');hook('failed',str(e));raise
