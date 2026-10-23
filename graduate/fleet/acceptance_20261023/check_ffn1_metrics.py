"""Independent RMS formula, full raw-output comparison and bigint spot oracle."""
from pathlib import Path
import json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
source=Path('/home/yang/kwt-worktrees/ds-r5-gates/graduate/ds_tasks/r4_rtl/vectors/main')
pack=Path('/home/yang/kwt-worktrees/ds-p027-precision/graduate/ds_tasks/p027_precision/frozen/frozen_calib.npz')
z=np.load(pack,allow_pickle=False);rows=240*99;cols=256;total=12*rows
files={k:source/v for k,v in dict(dut='rtl_out.bin',gold='golden.bin',fp='fp32_h.bin',inputs='inputs.bin').items()}
d=np.memmap(files['dut'],dtype='<i8',mode='r',shape=(total,cols));g=np.memmap(files['gold'],dtype='<i8',mode='r',shape=(total,cols));fp=np.memmap(files['fp'],dtype='<f4',mode='r',shape=(total,cols));x=np.memmap(files['inputs'],dtype='<i2',mode='r',shape=(total,64))
result=[];badall=badspots=0
for k in range(12):
 key=f'layer{k:02d}';scale=z[key+'_sh1'];w=z[key+'_w1'];b=z[key+'_b1'];sse=refsum=0.;n=0;bad=0
 for start in range(0,rows,2048):
  sl=slice(k*rows+start,k*rows+min(start+2048,rows))
  di=np.asarray(d[sl]);ref=np.asarray(fp[sl],dtype=np.float64);est=di.astype(np.float64)*scale[None,:]
  bad+=int(np.count_nonzero(di!=g[sl]));sse+=float(np.sum((est-ref)**2));refsum+=float(np.sum(ref**2));n+=ref.size
 spotbad=0
 for r in (0,98,rows//2,rows-1):
  a=[int(v) for v in x[k*rows+r]]
  for c in range(cols):
   want=int(b[c])+sum(a[t]*int(w[t,c]) for t in range(64))
   spotbad+=int(want!=int(d[k*rows+r,c]))
 correct=(sse/n)**.5/max((refsum/n)**.5,1e-8)
 old=(sse/max(refsum,1e-8))**.5
 result.append(dict(layer=k,elements=n,mismatch_raw_vs_saved_golden=bad,bigint_spot_mismatches=spotbad,nrmse=correct,nrmse_percent=correct*100,old_floor_nrmse=old,old_vs_correct_abs=abs(correct-old)))
 badall+=bad;badspots+=spotbad
 print('layer',k,'NRMSE%',correct*100,'mismatch',bad,'spot',spotbad,flush=True)
r=dict(scope='72,990,720 raw DUT values vs saved golden; independent bigint oracle12,288 outputs; independent RMS on all FP32 raw references',source=str(source),layers=result,total_elements=total*cols,bigint_outputs=12*4*256,mismatches=badall,bigint_mismatches=badspots,worst_nrmse_percent=max(v['nrmse_percent'] for v in result),source_hashes={})
for name,p in files.items():
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 r['source_hashes'][name]=h.hexdigest()
(HERE/'ffn1_metric_rerun.json').write_text(json.dumps(r,indent=2)+'\n')
if badall or badspots:raise SystemExit(1)
