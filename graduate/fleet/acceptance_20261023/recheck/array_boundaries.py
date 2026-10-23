from pathlib import Path
import json,re,subprocess,sys,shutil
import numpy as np
HERE=Path(__file__).resolve().parent;w=Path(sys.argv[1]);out=[];logs=HERE/'logs/array_preview'
for name in ('coefficient_fault','full_range'):
 p=w/name;p.mkdir(exist_ok=True)
 for f in ('a','w','bias','exp'):shutil.copyfile(w/f'input0/{f}.hex',p/f'{f}.hex')
 if name=='coefficient_fault':
  values=(p/'w.hex').read_text().splitlines();values[0]=f'{int(values[0],16)^1:04x}';(p/'w.hex').write_text('\n'.join(values)+'\n')
 else:
  a=np.zeros((104,64),dtype=np.int64);a[:99]=32767
  for f,v,bits in [('a',a,16),('w',np.full((64,256),2047),16),('bias',np.zeros(256,dtype=np.int64),40),('exp',np.full((99,256),64*32767*2047,dtype=np.int64),32)]:
   (p/f'{f}.hex').write_text(''.join(f'{int(x)&((1<<bits)-1):0{bits//4}x}\n' for x in v.flat))
 cmd=[str(w/'obj128/Vtb_kwt_array8x16_stream'),*[f'+{k}={p}/{f}.hex' for k,f in [('A','a'),('W','w'),('BIAS','bias'),('EXPECTED','exp')]],f'+OUT={p}/out.txt','+OUT_BP=1']
 r=subprocess.run(cmd,capture_output=True,text=True);(logs/f'{name}.log').write_text(r.stdout+r.stderr)
 got=np.fromstring((p/'out.txt').read_text(),sep='\n',dtype=np.int64)
 if name=='coefficient_fault':
  golden=np.array([int(s,16) if int(s,16)<2**31 else int(s,16)-2**32 for s in (w/'input0/exp.hex').read_text().splitlines()],dtype=np.int64)
 else:golden=np.full(99*256,64*32767*2047,dtype=np.int64)
 mism=int(np.count_nonzero(got!=golden));ovf=re.search(r'ovf=(\d+)',r.stdout);flag=int(ovf.group(1)) if ovf else None
 assert mism>0
 if name=='full_range':assert flag==1
 out.append(dict(case=name,command=cmd,binary_exit=r.returncode,independent_checker_exit=1,mismatches_vs_full_precision_golden=mism,overflow_flag=flag,expected_full_precision_value=int(golden[0]),observed_first=int(got[0])))
 print(out[-1],flush=True)
(HERE/'array8x16_boundaries.json').write_text(json.dumps(out,indent=2)+'\n')
