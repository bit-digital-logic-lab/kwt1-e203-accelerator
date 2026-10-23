"""Probe immutable existing 8x8/8x16 array engines, K=64; not final accelerator timing."""
from pathlib import Path
import tempfile,subprocess,json,hashlib
ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent
rtl=ROOT/'hardware/rtl/pulse_array'; logs=HERE/'probe_logs';logs.mkdir(exist_ok=True)
res=[]
for cols in (8,16):
 w=Path(tempfile.mkdtemp(prefix=f'kwt-array8x{cols}-'))
 text=(rtl/'tb/tb_kwt_pingpong_cacc_engine.sv').read_text().replace('COLS = 16','COLS = '+str(cols)).replace('KLEN = 2','KLEN = 64').replace('kwt_pingpong_cacc_engine dut','kwt_pingpong_cacc_engine #(.ROWS(ROWS), .COLS(COLS)) dut')
 text=text.replace('    int checks = 0;', '''    int checks = 0;
    int ticks=0, launches=0, first_tick=0, last_tick=0, gap_min=9999, gap_max=0, busy_ticks=0;
    always @(posedge clk) if(rst_n) begin
        ticks++;
        if(busy) busy_ticks++;
        if(dut.array.launch) begin
            if(launches==0) first_tick=ticks;
            else if(launches % 64 != 0) begin
                if(ticks-last_tick < gap_min) gap_min=ticks-last_tick;
                if(ticks-last_tick > gap_max) gap_max=ticks-last_tick;
            end
            last_tick=ticks;
            launches++;
        end
    end''')
 text=text.replace('$display("PASS pingpong+CACC:', '$display("PROBE cols=%0d busy=%0d launches=%0d gap_min=%0d gap_max=%0d", COLS, busy_ticks, launches, gap_min, gap_max);\n        $display("PASS pingpong+CACC:')
 text=text.replace('total = (r+1)*(3*c+8) + (c-4);', 'total = c-4; for(int k=0;k<KLEN;k++) total += ((k==0)?r+1:2*(r+1))*(c+2+k);').replace('total = -((r+1)*(2*c+3)) + 7;', 'total = 7; for(int k=0;k<KLEN;k++) total -= (r+1)*(c+1+k);')
 tb=w/'tb.sv';tb.write_text(text)
 sources=[rtl/'kwt_product_array.sv',rtl/'kwt_cacc.sv',ROOT/'hardware/rtl/sram/kwt_operand_pingpong.sv',rtl/'kwt_pingpong_cacc_engine.sv',tb]
 cmd=['verilator','--binary','--timing','--assert','-Wno-fatal','-j','2','--top-module','tb_kwt_pingpong_cacc_engine','--Mdir',str(w/'obj'),*[str(p) for p in sources]]
 p=subprocess.run(cmd,capture_output=True,text=True);(logs/f'build{cols}.log').write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError('build failed')
 run=[str(w/'obj/Vtb_kwt_pingpong_cacc_engine')];p=subprocess.run(run,capture_output=True,text=True);(logs/f'run{cols}.log').write_text(p.stdout+p.stderr)
 print(p.stdout,flush=True)
 res.append(dict(rows=8,cols=cols,build_cmd=cmd,build_exit=0,run=run,exit=p.returncode,output=p.stdout,source_sha256={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources[:-1]}))
 if p.returncode:raise RuntimeError('simulation failed')
(HERE/'primary_array_probe.json').write_text(json.dumps(res,indent=2)+'\n')
