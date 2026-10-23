from pathlib import Path
import time,subprocess,json
H=Path(__file__).resolve().parent;REPO=H.parents[3];check=REPO/'.opencode/skills/quarter-hour-check/scripts/check.py';result=H/'array8x16_preview.json'
def hook(state,note):subprocess.run(['python3',str(check),'update','--id','primary-array8x16-preview','--state',state,'--service','kwt-primary-array-preview-watch-20261023.service','--result',str(result),'--note',note],stdout=subprocess.DEVNULL)
def alive():
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   args=(p/'cmdline').read_bytes().split(b'\x00')
   if len(args)>1 and args[1].endswith(b'/array8x16_preview.py'):return True
  except OSError:pass
 return False
hook('running','128bit twelve layers verified; 64bit recompilation queued; full-range32bit export rejected')
while alive():time.sleep(5)
if result.exists():
 r=json.loads(result.read_text());good=len(r.get('cases',[]))==24 and all(x['mismatches']==0 for x in r['cases'])
 hook('complete' if good else 'failed','preview completed; full-range blocker remains, not final architecture acceptance')
else:hook('failed','preview process ended without full result; 128bit partial evidence remains')
