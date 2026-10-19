# Bias-overlap one-descriptor schedule — bias at CACC export, no pre-MAC barrier

Worktree: `ds-p027-bias` · branch `codex/kwt-ds-streaming-bias`, based on the
accepted `streaming_mac` commit `c337966`. This directory is new; every prior
`streaming_mac` artifact is unchanged and was only read.

## 1. Short answer

The accepted dual-CACC task preloads bias **serially** (`LOAD_A → LOAD_B →
LOAD_BIAS → RUN_TILE → STORE_OUT`) before any MAC. The user correction is
implemented: **A, W and bias are prepared concurrently wherever local SRAM
ports allow, and the bias add happens at CACC export, not as a pre-MAC
barrier.** The new one-descriptor schedule streams A row tiles, prefetches the
next W page, and prefetches bias column groups into a dedicated bank; compute
starts without bias and only an export waits for its bias group.

Measured on the same unchanged core `kwt_row_broadcast_tile.sv`, the same
unchanged `kwt_tile_output_stage.sv`, the same 16-bit bus and the same clock:

| one FFN1 layer (99×64 × 64×256 + bias), 16 rows | cycles | vs baseline | vs accepted dual |
| --- | ---: | ---: | ---: |
| baseline single-CACC (accepted, RTL) | **154 240** | — | — |
| accepted dual-CACC (accepted, RTL) | **125 603** | −28 637 (−18.6 %) | — |
| **bias-overlap, A_PRELOAD_TILES=3 (RTL)** | **121 915** | **−32 325 (−20.96 %)** | **−3 688 (−2.94 %)** |
| bias-overlap full (cycle model, actual vs modeled below) | 121 913 | −32 327 | −3 690 |

The full RTL run reproduces the committed baseline **154 240 exactly** and the
accepted dual **125 603 exactly**, so the comparison really is same-clock /
same-bus / same core. The gain is real but modest: the old bias barrier was
only 512 shared-bus slots out of 7 920 initial-load slots, and the accepted
dual already hid the rest. All numbers are cycle counts of a functional/task
simulation; they are **not** an E203 run and **not** a 150 MHz STA result.

## 2. What "simultaneous" means here (precise)

* **Concurrent local SRAM ports.** A compute tile reads A and W from their own
  banks (and accumulates in its CACC page) every MAC cycle. The A fill port,
  the W fill port, the bias fill port and the output-stage read port are
  separate local SRAM ports; a fill and a compute read may therefore happen in
  the same cycle. This is *local* concurrency.
* **One shared 16-bit bus.** All external traffic (A fill, W fill, bias fill,
  result write-back) leaves/enters through **one 16-bit bus that carries at
  most one word/slot per cycle**. Four requesters therefore need arbitration.
  `kwt_shared_bus_arbiter.sv` grants exactly one request per enabled cycle
  (fixed priority `OUT > BIAS > A > W`) and the TB asserts that
  `bus_grant_a+w+bias+out` equals the architectural slot count. **No path in
  this experiment claims two requesters move data on that bus in one cycle.**
* **Bias is not a compute barrier.** A column tile needs 16 INT32 bias values
  = **32 shared 16-bit slots** (low half then high half). The only deadline is
  that group *ct* must be valid before the **first CACC export** of column
  tile *ct*. Compute of tile *ct* may run and finish while the group is still
  filling; only the export engine waits.
* **Back-pressure output.** A 32-bit result needs **two** bus slots. The TB
  requests a slot per cycle and pulses `stage_out_ready` on the second slot,
  i.e. one result per two cycles, the same average bandwidth as the accepted
  harness.

## 3. Schedule (one descriptor)

One command descriptor carries `A base`, `W base`, `bias base`, `out base`,
`M=99`, `K=64`, `N=256`. There is no `LOAD_BIAS` state.

| engine | rule | validity |
| --- | --- | --- |
| A resident fill | sequential 16-bit words; a row tile is computable when its own `ROWS*K` words are resident | `a_row_ready(rt)`; `+A_PRELOAD_TILES` gates the first tile only |
| W page prefetch | two pages, one per column group; next group fills the idle page while the current group computes | `w_page_valid[p] && w_page_group[p]==ct`; page cannot be reused until its previous group finished compute |
| bias group fill | 256×32 dedicated bank, group = 16 INT32 = 32 slots; lookahead `<= compute_col+1` | `group_valid[ct]` rises only after all 32 half-writes; export waits otherwise |
| MAC | row-major within a column group, alternating CACC pages; reads A/W local banks | starts when `a_row_ready(rt)` and W page valid; never waits on bias |
| CACC export | dual pages; export tile *T* on page *p* while tile *T+1* computes on `1-p`; bias read from the dedicated bank and added inside `dump_data` | starts when `stage_start_ready && group_valid[ct]`; page freed at `capture_done` |

### Valid / ready and back-pressure

* Bus: `req[i]` is held with stable data/address until `gnt[i]`; one `gnt` per
  enabled cycle; `enable=0` is a no-loss contention cycle (global bus gap).
* Compute: a tile starts only when its W page and A row tile are valid and its
  CACC page is free (`cacc_busy` is cleared at `capture_done`, not at capture
  start).
* Export: if the bias group is not valid the export engine stalls (counted as
  `bias_stall_cycles`); if the output stage is not idle it stalls
  (`stage_stall_cycles`). Compute stalls only when both CACC pages are pending
  (`cacc_stall_cycles`) or a load is late (`load_stall_cycles`).

### Reset

`rst_n` clears the CACC pages, the bias-bank validity bits and the arbiter
grants. The task-level `clear` strobes are deliberately separate: the CACC
`clear` (start of each tile) must **not** clear bias, because bias for the
next tile is already resident. The unit microtest proves: bank validity is 0
after reset, a partial group never becomes valid, reset during a partial fill
invalidates it, and a post-reset refill is correct.

### Hazards covered

half-written bias group used for export; A row tile read before resident; W
page overwritten while its group computes; CACC page overwritten while
exporting; output stage overflow (capture only when idle); shared-bus double
grant; stale bias after reset. Each is either gated in the controller or
checked in `tb_kwt_bias_arbiter_unit.sv`.

### Latency if bias misses its deadline

Export of tile *ct* stalls until `group_valid[ct]`; compute can continue on
the other CACC page. With the normal lookahead, group *ct* starts filling when
compute reaches column *ct-1*, i.e. roughly one column group (~16 tile
computes, ≈16k cycles) before it is needed, so the deadline is met by a wide
margin. If it is missed, the penalty is bounded and data stays correct:

| full FFN1 RTL case | cycles | bias stall | CACC stall | outputs |
| --- | ---: | ---: | ---: | --- |
| normal, preload=3 | 121 915 | 0 | 0 | 25 344 INT32 exact |
| bias held off for 6 000 cycles | 121 927 | 889 | 100 | exact |
| bias held off for 20 000 cycles | 135 927 | 14 889 | 14 100 | exact |
| output back-pressure `/3` (one result per 6 cycles) | 186 136 | 0 | 61 806 | exact |
| global bus gap every 7th cycle | 123 929 | 0 | 0 | exact |

The 6 000-cycle delay costs only **+12 cycles**: the bias miss is absorbed by
the export/compute slack and by using the other CACC page. A 20 000-cycle
delay costs +14 012 cycles but is still exact. Output back-pressure is the only
case that changes the critical path materially; correctness is preserved.

## 4. A-prefetch sweep (full FFN1, RTL)

| `A_PRELOAD_TILES` | initial load | compute wall | load stalls | total cycles |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2 080 | 119 480 | 2 104 | 121 915 |
| 2 | 3 104 | 118 456 | 1 080 | 121 915 |
| **3** | **4 128** | **117 432** | **56** | **121 915** |
| 4 | 5 152 | 117 376 | 0 | 122 883 |
| 5 | 6 176 | 117 376 | 0 | 123 907 |
| 6 | 7 200 | 117 376 | 0 | 124 931 |
| 7 (=all A) | 7 392 | 117 376 | 0 | 125 123 |

Preload 1/2/3 all give 121 915 because the shared bus, not compute, is the
early bottleneck. Preload 3 is recommended: same total as preload 1/2 with the
fewest in-compute stalls (56) and a deterministic 4 128-cycle initial load.
Preload 7 is the conservative "all A resident before compute" variant and
still beats the accepted dual (125 123 vs 125 603).

## 5. Actual RTL vs model, kept separate

`model/bus_event_model.py` is a **cycle-level** event model. It first
reproduces the accepted RTL with fitted constants (full FFN1 baseline
**154 240**, dual **125 603**, both exact; tiny/ragged/twotile/overflow within
±1 cycle), then predicts the new schedule over one shared 16-bit bus.

| quantity | RTL (measured) | model (predicted) | delta |
| --- | ---: | ---: | ---: |
| baseline full | 154 240 | 154 240 | 0 |
| accepted dual full | 125 603 | 125 603 | 0 |
| bias-overlap full, preload=3 | **121 915** | **121 913** | +2 (0.002 %) |
| delayed bias 6 000 | 121 927 | 121 923 | +4 |
| delayed bias 20 000 | 135 927 | 135 923 | +4 |
| output back-pressure /3 | 186 136 | 186 135 | +1 |
| bus gap /7 | 123 929 | 123 927 | +2 |

The model also shows the bus is never oversubscribed: full-frame slots are
`OUT 50 688 + A 6 336 + W 16 384 + bias 512 = 73 920`, while the run is
121 915 cycles, i.e. ≈60.6 % slot occupancy (`bus_idle_cycles=47 995`).
Back-pressure/gap runs increase stalls, not bus violations.

## 6. Area / SRAM macros / ports (capacity only)

| buffer | size | 16-bit macro budget | ports |
| --- | ---: | --- | --- |
| **new bias bank** 256×32 | 1 024 B | 2 × `16b×1024` (or 512 16-bit registers) | 1 × 16-bit fill W + 1 × 32-bit export R |
| **new arbiter** | 4 requesters | combinational, no data macro | 1 grant/cycle |
| A resident 99×64×16 (unchanged) | 12 672 B | 2 × `16b×4096` | 1R compute + 1W fill |
| W ping-pong 2×64×16×16 (unchanged) | 4 096 B | 2 × `16b×1024` | compute reads active page, fill writes idle page |
| CACC dual 2×16×16×32 (unchanged) | 2 048 B | 4 × `16b×1024` if macro-mapped | 1R export + 1W accumulate page |
| output stage 256×32 (unchanged) | 1 024 B | 2 × `16b×1024` | 1W capture + 1R drain |

The dedicated 256×32 bias bank reuses the bias storage the accepted task
already held; the new logic is the group-valid bits and the arbiter. The
16-bit bus width is the binding constraint; capacity counts only.

**Unresolved:** 150 MHz closure, exact SRAM macro bank/port arbitration and
the 2.25 mm² accelerator+SRAM fit are **not** proven here. The 150 MHz numbers
below are conditional arithmetic on simulated cycles, not STA.

| at 150 MHz (conditional) | one layer | 12 layers (model) |
| --- | ---: | ---: |
| baseline | 1.0283 ms | 12.34 ms |
| accepted dual | 0.8374 ms | 10.05 ms |
| bias-overlap | 0.8128 ms | 9.75 ms |

## 7. Exact commands and exits

```bash
bash hardware/experiments/streaming_mac/bias_overlap/run.sh          # exit 0
#   builds: baseline TB, accepted dual TB, bias-overlap TB, bias/arbiter unit
#   runs:   tiny/ragged/twotile/overflow/full, preload 1..7, delayed bias
#           6000/20000, OUT_BP 3/6, BUS_GAP 7, combined
#   writes: bias_overlap/results.json, logs/, model/model_results.json
```
Last run (this worktree): `run.sh` exit **0**; baseline full = 154 240
(matches committed), accepted dual full = 125 603 (matches committed),
bias-overlap full = 121 915, all outputs INT32-exact, `unit=PASS`.

Individual commands (paths relative to repo root; Verilator 5.047):
```bash
# bias-overlap task build
verilator --binary --timing --assert -Wno-fatal -j 4 \
  --top-module tb_kwt_bias_overlap_task -GROWS=16 --Mdir /tmp/bo \
  hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv \
  hardware/rtl/pulse_array/kwt_tile_output_stage.sv \
  hardware/experiments/streaming_mac/bias_overlap/rtl/kwt_shared_bus_arbiter.sv \
  hardware/experiments/streaming_mac/bias_overlap/rtl/kwt_bias_bank.sv \
  hardware/experiments/streaming_mac/bias_overlap/tb/tb_kwt_bias_overlap_task.sv
# unit microtest
verilator --binary --timing --assert -Wno-fatal -j 4 \
  --top-module tb_kwt_bias_arbiter_unit --Mdir /tmp/u \
  hardware/experiments/streaming_mac/bias_overlap/rtl/kwt_shared_bus_arbiter.sv \
  hardware/experiments/streaming_mac/bias_overlap/rtl/kwt_bias_bank.sv \
  hardware/experiments/streaming_mac/bias_overlap/tb/tb_kwt_bias_arbiter_unit.sv
# cycle model
python3 hardware/experiments/streaming_mac/bias_overlap/model/bus_event_model.py \
  --out hardware/experiments/streaming_mac/bias_overlap/model/model_results.json \
  --rtl-results hardware/experiments/streaming_mac/bias_overlap/results.json
```

## 8. What is proven and what is not

**Proven (RTL + unit):** one-descriptor concurrent A/W/bias fill; bias added at
CACC dump; independent INT32 outputs at CACC dump for normal, delayed-bias,
contention/back-pressure and reset (unit) cases; one-hot shared-bus grants and
architectural slot accounting; exact reproduction of the accepted baseline and
dual full-FFN1 cycles; a 3 688-cycle (2.94 %) improvement over the accepted
dual for full FFN1.

**Not proven:** E203/System-Bus/NICE integration; 150 MHz STA; area/power
sign-off or a 2.25 mm² fit; 12-layer quantised numerics and layer switching;
silicon-level SRAM macro timing. The A row-tile streaming result assumes
separate A fill and compute read ports (stated in the port table); a
single-port A macro would need a different bank split.

## 9. Files

```
hardware/experiments/streaming_mac/bias_overlap/
├── README.md
├── run.sh
├── run_bias_overlap.py
├── results.json
├── logs/                             per-build/run logs, unit + model logs
├── rtl/
│   ├── kwt_shared_bus_arbiter.sv     4-requester, one-hot, fixed priority
│   └── kwt_bias_bank.sv              256×32 dedicated bias bank + group valid
├── tb/
│   ├── tb_kwt_bias_overlap_task.sv   one-descriptor task harness
│   └── tb_kwt_bias_arbiter_unit.sv   bank/arbiter unit microtest
└── model/
    ├── bus_event_model.py            cycle-accurate shared-bus event model
    └── model_results.json            model output
```
