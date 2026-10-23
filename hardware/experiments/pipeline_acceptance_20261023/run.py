"""Primary-agent handshake/bandwidth measurement on immutable R4 source.
Copies pinned Git blobs into a private build directory; never edits DS worktrees.
"""
from pathlib import Path
import hashlib,importlib.util,json,re,subprocess,tempfile
HERE=Path(__file__).resolve().parent
REPO=Path('/home/yang/Desktop/ic设计竞赛')
PIN='8490830ef4c0446f41f6a50c7e3185b51a54f0f0'
BASE='hardware/experiments/integrated_singlemac_bias/'
FILES=[BASE+'rtl/kwt_single_mac_dual_cacc.sv','hardware/rtl/pulse_array/kwt_tile_output_stage.sv',BASE+'rtl/kwt_bias_bank.sv',BASE+'rtl/kwt_shared_bus_arbiter.sv',BASE+'tb/tb_kwt_integrated_task.sv']
work=Path(tempfile.mkdtemp(prefix='kwt-primary-pipeline-'))
logs=HERE/'logs'; logs.mkdir(exist_ok=True)
commands=[]
def execute(args,log):
 p=subprocess.run([str(a) for a in args],capture_output=True,text=True)
 (logs/log).write_text(p.stdout+p.stderr)
 commands.append({'argv':[str(a) for a in args],'exit':p.returncode,'log':str(logs/log)})
 if p.returncode: raise RuntimeError(f'{log} exit={p.returncode}')
 return p.stdout
sources={}
for path in FILES:
 raw=subprocess.check_output(['git','-C',str(REPO),'show',f'{PIN}:{path}'])
 sources[path]=hashlib.sha256(raw).hexdigest();(work/Path(path).name).write_bytes(raw)
helper=work/'helper_src/nested/helper.py'
helper.parent.mkdir(parents=True)
helper.write_bytes(subprocess.check_output(['git','-C',str(REPO),'show',f'{PIN}:hardware/experiments/r4_synth/run_r4.py']))
spec=importlib.util.spec_from_file_location('r4_pinned',helper); R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
tb=work/'tb_kwt_integrated_task.sv';text=tb.read_text()
observer='''
    longint mon_cycles=0, mon_beats=0, mon_macs=0, mon_gap=0;
    longint mon_internal_idle=0, mon_max_internal_gap=0;
    always @(posedge clk) begin
        if (rst_n) begin
            mon_cycles++;
            if (stream_valid && in_ready && !clear_bit) begin
                if (mon_beats != 0) begin
                    mon_internal_idle += mon_gap;
                    if (mon_gap > mon_max_internal_gap) mon_max_internal_gap=mon_gap;
                end
                mon_gap=0;
                mon_beats++;
                mon_macs += $countones(rm);
            end else mon_gap++;
        end
    end
'''
text=text.replace('    string a_path,',observer+'\n    string a_path,')
text=text.replace('                $display("ARCH_METRICS', '                $display("MON cycles=%0d beats=%0d useful_macs=%0d internal_idle=%0d max_internal_gap=%0d", mon_cycles, mon_beats, mon_macs, mon_internal_idle, mon_max_internal_gap);\n                $display("ARCH_METRICS')
tb.write_text(text)
execute(['verilator','--binary','--timing','--assert','-Wno-fatal','-j','2','--top-module','tb_kwt_integrated_task','--Mdir',work/'obj','-GROWS=16',*[work/Path(p).name for p in FILES]],'build.log')
case=R.make_case('full');inputs=R.prepare_inputs(work/'inputs',case)
cases=[]
for name,pre,bp,gap in [('normal',3,1,0),('out2',3,2,0),('out3',3,3,0),('out6',3,6,0),('bus2',3,1,2),('bus7',3,1,7),('pre1',1,1,0),('pre7',7,1,0)]:
 out=work/f'{name}.txt'
 stdout=execute([work/'obj/Vtb_kwt_integrated_task',f'+A={inputs["a"]}',f'+W={inputs["w"]}',f'+BIAS={inputs["bias"]}',f'+EXPECTED={inputs["expected"]}',f'+OUTPUT={out}','+M=99','+K=64','+N=256','+GAPS=0',f'+A_PRELOAD_TILES={pre}',f'+OUT_BP={bp}',f'+BUS_GAP={gap}'],name+'.log')
 metrics=R.parse_metrics(stdout)
 line=next(l for l in stdout.splitlines() if l.startswith('MON ')); mon={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
 check=R.verify_actual(out,case['expected'],99,256)
 assert check['mismatches']==0 and check['count']==25344
 assert mon['beats']==7*16*64*16 and mon['useful_macs']==99*64*256
 grants=sum(metrics['bus_grant_'+k] for k in ('out','bias','a','w'))
 assert grants+metrics['bus_idle_cycles']==metrics['bus_cycles']
 n=metrics['bus_cycles']
 ratios=dict(bus_occupied=grants/n,mac_feed_active=mon['beats']/n,mac_useful=mon['useful_macs']/(16*n),masked_lanes=1-mon['useful_macs']/(16*mon['beats']),feed_bubble_fraction=1-mon['beats']/metrics['compute_wall_cycles'])
 cases.append(dict(name=name,metrics=metrics,observer=mon,ratios=ratios,output_check=check,grants=grants))
 print(name,'cycles',n,'bus',round(100*ratios['bus_occupied'],2),'feed',round(100*ratios['mac_feed_active'],2),'useful',round(100*ratios['mac_useful'],2),flush=True)
report=dict(source_commit=PIN,source_sha256=sources,observer_sha256=hashlib.sha256(tb.read_bytes()).hexdigest(),commands=commands,input_sha256=case['input_sha256'],golden_sha256=case['golden_sha256'],cases=cases,scope='Functional R4 INT16/mod32 task harness, 16-bit ideal bus. TB owns scheduling/A/W. Not frozen A16W12 integrated RTL, official bus or physical timing.',build_dir=str(work))
(HERE/'results.json').write_text(json.dumps(report,indent=2)+'\n')
