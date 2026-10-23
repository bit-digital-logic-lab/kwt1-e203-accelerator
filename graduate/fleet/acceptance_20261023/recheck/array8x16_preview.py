"""Preliminary immutable copy of uncommitted candidate; never counts as final delivery."""
from pathlib import Path
import json,hashlib,shutil,tempfile,subprocess
import numpy as np
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];source=Path('/home/yang/kwt-worktrees/ds-r6-array8x16');report=json.load(open(source/'graduate/ds_tasks/r6_array8x16/results.json'));w=Path(tempfile.mkdtemp(prefix='kwt-array-preview-'));logs=HERE/'logs/array_preview';logs.mkdir(exist_ok=True)
paths=['hardware/experiments/r6_array8x16/rtl/'+f for f in ['kwt_product_array_tagged.sv','kwt_cacc_cellbank.sv','kwt_array8x16_stream_engine.sv']]+['hardware/experiments/r6_array8x16/tb/tb_kwt_array8x16_stream.sv']
hashes={}
for rel in paths:
 data=(source/rel).read_bytes();h=hashlib.sha256(data).hexdigest();assert h==report['sources'][rel],f'worker source changed: {rel}';hashes[rel]=h;(w/Path(rel).name).write_bytes(data)
fix=source/'graduate/ds_tasks/r6_array8x16/fixtures/ffn1_fixtures.npz';shutil.copyfile(fix,w/'fixture.npz');z=np.load(w/'fixture.npz',allow_pickle=False)
records=[];commands=[]
for bus in (128,64):
 obj=w/f'obj{bus}';cmd=['verilator','--binary','--timing','--assert','-Wno-fatal','-j','2','--top-module','tb_kwt_array8x16_stream','--Mdir',str(obj),f'-GBUS_W={bus}',*[str(w/Path(s).name) for s in paths]]
 with (logs/f'build{bus}.log').open('w') as f:p=subprocess.run(['python3',str(REPO/'graduate/fleet/r6_20261023/heavy.py'),'--',*cmd],stdout=f,stderr=subprocess.STDOUT)
 commands.append(dict(argv=cmd,exit=p.returncode));assert p.returncode==0
 for L in range(12):
  d=w/f'input{L}';d.mkdir(exist_ok=True);a=z['out1'][0,L].astype(np.int64);weight=z['w1'][L];b=z['b1'][L]
  golden=np.array([[int(b[c])+sum(int(a[r,k])*int(weight[k,c]) for k in range(64)) for c in range(256)] for r in range(99)],dtype=np.int64)
  assert np.max(np.abs(golden))<2**31,'candidate32bit contract exceeded'
  padded=np.zeros((104,64),dtype=np.int64);padded[:99]=a
  for name,arr,bits in [('a',padded,16),('w',weight,16),('bias',b,40),('exp',golden,32)]:
   (d/f'{name}.hex').write_text(''.join(f'{int(x)&((1<<bits)-1):0{bits//4}x}\n' for x in arr.flat))
  out=d/f'out{bus}.txt';cmd=[str(obj/'Vtb_kwt_array8x16_stream'),f'+A={d}/a.hex',f'+W={d}/w.hex',f'+BIAS={d}/bias.hex',f'+EXPECTED={d}/exp.hex',f'+OUT={out}','+OUT_BP=1']
  p=subprocess.run(cmd,capture_output=True,text=True);(logs/f'bus{bus}_layer{L}.log').write_text(p.stdout+p.stderr);assert p.returncode==0
  got=np.fromstring(out.read_text(),sep='\n',dtype=np.int64).reshape(99,256);m=int(np.count_nonzero(got!=golden));assert m==0
  records.append(dict(bus=bus,layer=L,mismatches=m,output_count=got.size,log=p.stdout));print('PREVIEW',bus,L,'exact',flush=True)
r=dict(verdict='PRELIMINARY_NUMERIC_ONLY',source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),source_sha256=hashes,source_has_uncommitted_changes=bool(subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()),fixture_sha256=hashlib.sha256((w/'fixture.npz').read_bytes()).hexdigest(),commands=commands,cases=records,outputs_compared=sum(x['output_count'] for x in records),limitations=['Copied source hashes match worker intermediate report; original branch not yet delivered.','32bit output contract fits these model samples; not full-range34bit accumulator output proof.','TB supplies transactions/ready-page metadata; no official SoC bus/PDK/macro closure.'])
(HERE/'array8x16_preview.json').write_text(json.dumps(r,indent=2)+'\n')
