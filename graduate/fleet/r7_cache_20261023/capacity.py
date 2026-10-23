#!/usr/bin/env python3
"""Capacity and traffic bounds; no invented SRAM area or RTL cycle estimates."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEPTHS = (128, 1024, 4096)


def bank(depth, width=16, banks=1):
    """Min allocated words, then macro count, per equal-width independent bank."""
    units = (depth + 127) // 128
    options = []
    for n4096 in range((units + 31) // 32 + 1):
        for n1024 in range((units + 7) // 8 + 1):
            n128 = max(0, units - 32*n4096 - 8*n1024)
            options.append((32*n4096 + 8*n1024 + n128,
                            n4096+n1024+n128, n128, n1024, n4096))
    words, count, n128, n1024, n4096 = min(options)
    fewest=min(options,key=lambda x:(x[1],x[0]))
    slices = (width+15)//16
    return dict(logical_bytes=(depth*width*banks+7)//8,
                allocated_bytes=words*128*2*slices*banks,
                banks=banks, depth=depth, width=width,
                macro_counts={str(d):n*slices*banks for d,n in
                              zip(DEPTHS,(n128,n1024,n4096))},
                fewest_instances_alternative=dict(allocated_bytes=fewest[0]*128*2*slices*banks,
                    macro_counts={str(d):n*slices*banks for d,n in zip(DEPTHS,fewest[2:])}),
                port_feasibility='not implied by capacity; must be verified separately')


def main():
    data = dict(scope='FFN1 99x64 times 64x256, A16/W12 in16bit slots, accumulator40',
        spec=dict(shared_bytes_assuming_KiB=512*1024,
                  private_byte_cap=None, total_accelerator_area_limit_mm2=2.25,
                  private_max_status='unknown without specified macro areas and mapped logic area',
                  capacity_source='题目/640.jpg 4, page3/14',
                  area_source='题目/640.jpg 13, page12/14',
                  area_constraint='logic_mm2 + interface_mm2 + sum(count_i * macro_area_i_mm2) <= 2.25',
                  cold_task='task data/parameter caches cleared before each official performance task'),
        logical_bytes=dict(A=99*64*2,A_padded104=104*64*2,
                           W16=64*256*2,W12_packed=64*256*12//8,
                           bias40=256*5, CACC_two_pages40=8*16*2*5),
        macro_allocations={}, traffic_bounds=[], shared_examples=[],
        caveats=['Macro counts minimize allocated bits, then number of instances, not area.',
                 'Single/dual port macros of equal bit capacity can have different areas.',
                 'CACC cell registers belong in mapped logic area; raw bytes are not SRAM area.',
                 'No claim that inferred memories map to the target SRAM macros or meet150MHz.',
                 'Traffic is an optimistic lower bound: no row/tile/beat padding, contention or CPU overhead.'])
    alloc=data['macro_allocations']
    alloc['A_full_8banks']=bank(13*64,banks=8)
    alloc['W_two_pages_DP_combined']=bank(2*64,banks=16)
    alloc['W_two_pages_SP_separate']=bank(64,banks=32)
    alloc['W_full_16banks']=bank(16*64,banks=16)
    alloc['bias_40bit_256deep']=bank(256,40)
    for bus in (64,128):
        for depth in (8,16,32,64,128,256,512):
            alloc[f'FIFO_{bus}bit_{depth}beats']=bank(depth,bus)
    # If somebody tried to map each simultaneously updated CACC cell to SRAM:
    alloc['CACC_naive_cell_per_bank_NOT_recommended']=bank(2,40,banks=128)
    alloc['CACC_packed_wide_word_lower_capacity_only']=bank(2,128*40)
    for outbits in (32,40,48,64):
        for wbits in (12,16):
            y=(99*256*outbits+7)//8
            total=99*64*2+64*256*wbits//8+256*5+y
            for bus in (64,128):
                data['traffic_bounds'].append(dict(out_bits=outbits,weight_storage_bits=wbits,
                    output_bytes=y,total_payload_bytes=total,bus_bits=bus,
                    minimum_bus_beats=(total+bus//8-1)//(bus//8),
                    compute_tiled_8x8_cycles=26624,compute_tiled_8x16_cycles=13312,
                    output32_full_range_legal=(outbits!=32)))
        # A/W/Y +bias only: runtime/stack/descriptors must still be reserved.
        used=99*64*2+64*256*2+256*5+(99*256*outbits+7)//8
        data['shared_examples'].append(dict(out_bits=outbits,used_bytes=used,
            remaining_before_CPU_code_stack_descriptors=512*1024-used))
    for name,w in [('A_full_W_pingpong_DP','W_two_pages_DP_combined'),
                   ('A_full_W_pingpong_SP','W_two_pages_SP_separate'),
                   ('A_full_W_full','W_full_16banks')]:
        parts=['A_full_8banks',w,'bias_40bit_256deep','FIFO_128bit_64beats']
        alloc[name]=dict(parts=parts,allocated_bytes=sum(alloc[p]['allocated_bytes'] for p in parts),
                        excludes='CACC registers, tags, control, queues and pipeline registers')
    (HERE/'capacity.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v['allocated_bytes'] for k,v in alloc.items()
                      if k.startswith('A_full_W_')},ensure_ascii=False))


if __name__=='__main__':main()
