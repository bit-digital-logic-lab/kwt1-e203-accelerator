#!/usr/bin/env python3
"""Primary-authored preflight: committed RTL, fresh build, unbounded integer goldens.

Uses the worker's bus driver, but owns all input/golden files and independently
parses all exported values. This is a numerical preflight, not final acceptance.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
SOURCES=['hardware/experiments/r6_array8x16/rtl/'+x for x in
         ('kwt_product_array_tagged.sv','kwt_cacc_cellbank.sv','kwt_array8x16_stream_engine.sv')]
SOURCES+=['hardware/experiments/r6_array8x16/tb/tb_kwt_array8x16_stream.sv']


def main():
    worker=Path(sys.argv[1]);commit=sys.argv[2]
    logs=HERE/'logs/fullwidth_gate';logs.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='kwt-r7-width-gate-'))
    record=dict(commit=commit,verdict='running',source_sha256={},commands=[],cases=[],
                limitations=['Worker bus TB reused; primary owns independent stimulus/golden/comparison.',
                             'No official SRAM, timing, area, SoC acceptance.'])
    def save():
        (HERE/'fullwidth_gate.json').write_text(json.dumps(record,indent=2)+'\n')
    def run(cmd,name):
        with (logs/name).open('w') as stream:
            p=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
        record['commands'].append(dict(argv=[str(x) for x in cmd],exit=p.returncode,log=str(logs/name)))
        save();return p.returncode
    try:
        for rel in SOURCES:
            data=subprocess.check_output(['git','-C',str(worker),'show',f'{commit}:{rel}'])
            (root/Path(rel).name).write_bytes(data)
            record['source_sha256'][rel]=hashlib.sha256(data).hexdigest()
        cases=[]
        for name in ('max_positive','max_negative_operands','cancellation_large_bias'):
            a=[[0]*64 for _ in range(104)];w=[[0]*256 for _ in range(64)];b=[0]*256
            for r in range(99):
                a[r]=[32767 if name=='max_positive' else -32768 for k in range(64)]
                if name=='cancellation_large_bias':a[r]=[32767 if (r+k)%2 else -32768 for k in range(64)]
            for k in range(64):
                w[k]=[2047 if name=='max_positive' else -2048 for c in range(256)]
                if name=='cancellation_large_bias':w[k]=[2047 if (k+c)%2 else -2048 for c in range(256)]
            if name=='cancellation_large_bias':b=[(1 if c%2 else -1)*(1<<38) for c in range(256)]
            golden=[b[c]+sum(a[r][k]*w[k][c] for k in range(64)) for r in range(99) for c in range(256)]
            assert all(-(1<<39)<=x<(1<<39) for x in golden)
            d=root/name;d.mkdir()
            for f,values,bits in [('a',[x for row in a for x in row],16),('w',[x for row in w for x in row],16),('bias',b,40),('exp',golden,64)]:
                (d/f'{f}.hex').write_text(''.join(f'{x&((1<<bits)-1):0{bits//4}x}\n' for x in values))
            cases.append((name,d,golden))
        for bus in (64,128):
            obj=root/f'obj{bus}'
            cmd=['python3',str(PROJECT/'graduate/fleet/r6_20261023/heavy.py'),'--',
                 'verilator','--binary','--timing','--assert','-Wno-fatal','-j','2',
                 '--top-module','tb_kwt_array8x16_stream','--Mdir',str(obj),f'-GBUS_W={bus}',
                 *[str(root/Path(x).name) for x in SOURCES]]
            assert run(cmd,f'build{bus}.log')==0,'fresh Verilator build failed'
            for name,d,golden in cases:
                out=d/f'out{bus}.txt'
                cmd=[str(obj/'Vtb_kwt_array8x16_stream'),f'+A={d}/a.hex',f'+W={d}/w.hex',
                     f'+BIAS={d}/bias.hex',f'+EXPECTED={d}/exp.hex',f'+OUT={out}','+OUT_BP=1']
                rc=run(cmd,f'{name}_{bus}.log')
                got=[int(x) for x in out.read_text().split()] if out.exists() else []
                mismatch=sum(x!=y for x,y in zip(got,golden))+abs(len(got)-len(golden))
                record['cases'].append(dict(case=name,bus_bits=bus,exit=rc,count=len(got),
                    mismatches=mismatch,minimum=min(golden),maximum=max(golden),
                    output_sha256=hashlib.sha256(out.read_bytes()).hexdigest() if out.exists() else None))
                save();assert rc==0 and mismatch==0,'full width comparison failed'
            name,d,golden=cases[0]
            wrong=[golden[0]^1,*golden[1:]]
            (d/'wrong.hex').write_text(''.join(f'{x&((1<<64)-1):016x}\n' for x in wrong))
            badcmd=[str(obj/'Vtb_kwt_array8x16_stream'),f'+A={d}/a.hex',f'+W={d}/w.hex',
                    f'+BIAS={d}/bias.hex',f'+EXPECTED={d}/wrong.hex',f'+OUT={d}/bad{bus}.txt','+OUT_BP=1']
            rc=run(badcmd,f'wrong_expected_{bus}.log')
            assert rc!=0,'faulted golden was not rejected by bus TB'
        record['verdict']='preflight_pass_not_final_acceptance'
        record['outputs_compared']=sum(x['count'] for x in record['cases'])
        save()
    except BaseException as exc:
        record.update(verdict='failed',error=f'{type(exc).__name__}: {exc}')
        save();raise


if __name__=='__main__':main()
