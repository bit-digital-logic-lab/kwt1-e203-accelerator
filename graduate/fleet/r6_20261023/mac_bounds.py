"""Target8x8/8x16 analytic bounds. No simulated/physical timing claims."""
import json,math
from pathlib import Path
M,K,N=99,64,256
traffic={'legacy_INT16_INT32':M*K*2+K*N*2+N*4+M*N*4,'frozen_container16_bias64_output48':M*K*2+K*N*2+N*8+M*N*6,'frozen_packed12_bias64_output48':M*K*2+(K*N*12+7)//8+N*8+M*N*6}
a=[]
for cols in (8,16):
 mac=8*cols;ideal=math.ceil(M*K*N/mac);tiles=math.ceil(M/8)*math.ceil(N/cols);tiled=tiles*K
 a.append(dict(rows=8,cols=cols,physical_multipliers=mac,ideal_mac_cycles=ideal,masked_tiles=tiles,tiled_k_cycles=tiled,row_lane_efficiency=M/(math.ceil(M/8)*8),local_operand_bytes_per_launch=2*(8+cols),current_serial_read_fsm_issue_interval=max(8,cols)+2,
  perfect_overlap_lowerbound={str(bw):max(tiled,math.ceil(traffic['frozen_container16_bias64_output48']/(bw//8))) for bw in (64,128)}))
r=dict(shape=[M,K,N],macs=M*K*N,targets=a,byte_traffic=traffic,bus_beats={k:{str(w):math.ceil(n/(w//8)) for w in (64,128)} for k,n in traffic.items()},assumptions=['FP32 PC frontend/head excluded. Hardware frozen numeric contract must be checked.','48bit raw output width/bias64 are candidates; actual range and packing need tests.','Each operand and raw output crosses task bus once; metadata, alignment and protocol overhead excluded.','No cross-task warm cache or repeated transfers credited as utilization.','Ideal counts exclude array skew/fill/drain/CACC/writeback; tiled counts include99to104rowmask only.','Serial-read II is a static FSM count; probe actual cycles independently.'])
Path(__file__).with_name('mac_bounds.json').write_text(json.dumps(r,indent=2)+'\n')
for x in a:print(x)
