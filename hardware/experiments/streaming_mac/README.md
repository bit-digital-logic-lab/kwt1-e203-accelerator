# Streaming MAC experiment — does the KWT-1 matrix unit wait for all data?

Worktree: `ds-p027-streaming` · branch `codex/kwt-ds-streaming-mac`.
All files here were produced in this worktree; the arithmetic core and the
baseline testbench are used **unmodified**.

## 1. Short answer

The MAC unit does **not** wait for all data — under ideal supply the 16-row
broadcast core already accepts one `(A-row, B-word, column)` group **every
cycle (II = 1)**; `hardware/architecture/compute_only_pipeline.md` shows 1 024
back-to-back inputs, 16 384 MACs, all outputs exact. Deepening the multiplier
pipeline therefore does **not** raise task throughput here.

What *does* wait is the **task/accumulator schedule**:

* the old 8×16 tile system (`hardware/rtl/control/kwt_tile_system_model.sv`)
  is strictly sequential per tile — `LOAD_A → LOAD_B → LOAD_BIAS → RUN_TILE →
  STORE_OUT` — with `compute_page`/`load_page` hard-wired to 0, so it never
  overlaps load/compute/store;
* the 16-row staged task model overlaps the **output write-back** with the next
  tile's compute, but still pays a **serial CACC export** (~29 008 cycles =
  **18.8 %** of one FFN1 layer) because there is only one accumulator page.

The delivered fix is a **dual-CACC (two-page accumulator) task schedule**: the
CACC export of tile *T* runs under the compute window of tile *T+1*. Measured
on the same RTL core, same bus model, same clock.

| one FFN1 layer (99×64 × 64×256 + bias), 16 rows | base = single-CACC | dual-CACC | delta |
| --- | ---: | ---: | ---: |
| total cycles | **154 240** | **125 603** | **−28 637 (−18.6 %)**, 1.228× |
| initial load | 7 872 | 7 872 | — |
| compute | 117 040 | 117 040 | — |
| CACC export / drain | 29 008 | 356 (tail only) | overlapped |
| stall | — | 0 | — |
| shared bytes | 147 840 | 147 840 | — |
| valid outputs checked (INT32 exact) | 25 344 | 25 344 | — |

Baseline reproduces the committed `hardware/architecture/staged_task_row16_summary.json`
`full` value **154 240 exactly**, so the comparison really is same-clock /
same-bus. Pure-MAC lower bound for 16 rows is 114 688 cycles, so the baseline
task model was 34.5 % above the arithmetic bound and the dual version is 9.5 %
above it.

Gain appears only when the export fits inside the compute window, i.e.
`K·COLS > ROWS·COLS` (`K > 16` for 16 rows). Tiny / ragged / two-tile /
overflow shapes (K ≤ 8) show ≈ 0 % (dual is sometimes 1 cycle longer) — this is
reported, not hidden.

## 2. The four "matrix units" in this repo (kept distinct)

| unit | file | property |
| --- | --- | --- |
| old 8×16 tile system | `rtl/control/kwt_tile_system_model.sv` | sequential LOAD→RUN→STORE per tile, no overlap; not a throughput design |
| 16-row broadcast MAC core | `rtl/pulse_array/kwt_row_broadcast_tile.sv` | II = 1 under ideal supply; one B word/cycle, 16 MACs/cycle |
| 16-row staged task model | `architecture/tb/tb_kwt_row_broadcast_staged_task.sv` | A-resident + W ping-pong + output buffer; serial CACC export remains |
| graduate single-row FFN1 | `graduate/rtl/encoder/kwt_ffn1_row.sv` | 1 multiplier, INT48, streams one row behind LayerNorm; **not** a throughput engine |

## 3. Experiment design

* **Baseline (true baseline, unmodified):** `hardware/architecture/tb/tb_kwt_row_broadcast_staged_task.sv`
  + `kwt_row_broadcast_tile.sv` + `kwt_tile_output_stage.sv`. A resident, W
  ping-pong prefetch, one CACC page, output buffer drain overlapped with the
  next compute. Export is serialised: compute → capture CACC → next compute.
* **Modified (this experiment):** `hardware/experiments/streaming_mac/tb/tb_kwt_streaming_task.sv`.
  The **same, unchanged** `kwt_row_broadcast_tile.sv` is instantiated **twice**
  as a two-page accumulator. While page *A* streams tile *T+1*, page *B* is
  read out into the **same, unchanged** `kwt_tile_output_stage.sv`. Honest
  cost: 2× the MAC array is instantiated (16 extra 16×16 multipliers); a
  CACC-only second page (registers / two 16-bit macros per 32-bit word) is the
  area-optimal mapping and needs no second multiplier array. Using the repo's
  Nangate45 **relative** compute-core stat (`hardware/architecture/README.md`,
  16-row core ≈ 120,434.692 library units, SRAM unmapped), two core instances
  ≈ 240,869 units — i.e. roughly double the compute-core area for this
  schedule proof.
* Bus / memory model is identical in both TBs: one 16-bit word per cycle for
  load, one 16-bit slot per cycle for store at duty 1/2 (two slots per INT32
  result). Outputs are checked per `(row, col)` against an **independent numpy
  INT32 reference** (`(a@w + bias) & 0xFFFFFFFF`), including INT16/INT32
  saturation-wrap and ragged shapes; access counts are asserted.

### Stage counters (full FFN1 layer)

| counter | baseline | dual |
| --- | ---: | ---: |
| cycles | 154 240 | 125 603 |
| initial_load_cycles | 7 872 | 7 872 |
| compute_cycles | 117 040 | 117 040 |
| drain/capture_cycles | 29 008 | 356 |
| stall_cycles | 0 | 0 |
| overlapped_captures (tiles) | 0 | 111 / 112 |
| overlapped_output_words | 24 297 | 25 296 |
| a_load_words / w_load_words / bias_slots | 6 336 / 16 384 / 512 | same |
| out_slots | 50 688 | 50 688 |

### Deep back-pressure / reset / ordering

`tb_kwt_dualacc_unit.sv` (independent microtest, exit 0) proves reset clears
both pages, export runs while the other page accumulates, a mid-run reset
invalidates both pages, and dump-request gaps (back-pressure) preserve value
and ordering. The task TB additionally runs `+BACKPRESSURE=4` (tiny) and
`=3` (ragged): both INT32-exact; ragged at duty 1/3 costs 597 stall cycles —
still correct.

## 4. Bytes and required SRAM macros / ports

Per one FFN1 layer, required shared traffic = A 6 336 words (12 672 B) +
W 16 384 words (32 768 B) + bias 512 slots + output 50 688 slots =
**147 840 B** with each A/W/bias read once and each valid output written once.

| buffer | size | 16-bit macro budget | ports |
| --- | ---: | --- | --- |
| A resident 99×64×16 | 12 672 B | 2 × `16b×4096` | 1R compute + 1W load |
| W ping-pong 2 × 64×16×16 | 4 096 B | 2 × `16b×1024` | compute reads active page, DMA writes spare |
| CACC dual 2 × 16×16×32 | 2 048 B (regs, or 4 × `16b×1024` if macro-mapped) | 4 | 1R dump page + 1W accumulate page per row |
| output stage 256×32 | 1 024 B | 2 × `16b×1024` (width-cascade) | 1W capture + 1R drain |

Capacity counts only; bank/port arbitration and 150 MHz closure are **not**
proven here.

## 5. Whole 12-layer task — MODEL ONLY, separated from RTL

`model/event_schedule.py` is a discrete-event scheduler (COMPUTE / CAPTURE /
DRAIN resources) whose per-tile compute is **calibrated only from the baseline
RTL**, then used to *predict* the dual-CACC layer, and extended to 12 layers.
It reproduces the baseline layer exactly and predicts the dual layer at
**125 824** vs RTL **125 603** (error **+0.18 %**) — a genuine out-of-sample
check. Model 12-layer totals: baseline 1 850 880, dual 1 509 888 (1.226×);
conditional 150 MHz = **12.34 ms → 10.07 ms**. These are **model numbers**;
inter-layer E203 reprogram and shared-bus arbitration are assumed 0 and are
unproven.

## 6. Which ideas transfer from the primary sources

Sources (inspected 2026-10-19):
* NVDLA unit description — <https://nvdla.org/hw/v1/ias/unit_description.html>
* NVDLA hardware architecture — <https://nvdla.org/hw/v1/hwarch.html>
* Gemmini README — <https://github.com/ucb-bar/gemmini/blob/master/README.md>
* Gemmini paper (DAC'21) — <https://people.eecs.berkeley.edu/~alonamid/papers/gemmini-arxiv-1911.09925.pdf> · abstract <https://arxiv.org/abs/1911.09925>

**Transfers (and is now RTL-proven in this worktree):**
* NVDLA CACC separates an **assembly SRAM group** (accumulate) from a
  **delivery SRAM group** (export) and uses a **valid-credit** back-pressure
  protocol — exactly the "independent output buffer / dual CACC" direction
  measured here (NVDLA unit_description, §Convolution Accumulator).
* NVDLA **ping-pong register programming** hides CPU reprogram latency between
  layers; Gemmini `LoopMatmul`/`LoopConv` **double-buffer** matmul/conv tiles
  and monitor load/store/execute balance in a ROB. This is the same
  double-buffering idea applied to our W page and 12-layer parameter switch.
* Gemmini **decoupled access/execute** (separate load/store/execute controllers)
  and NVDLA's pipeline of CDMA→CBUF→CSC→CMAC→CACC justify overlapping the
  weight prefetch and result write-back with compute — this is what the
  baseline and dual schedules implement.
* NVDLA **multi-batch** hides fully-connected weight-reload bubbles by reusing
  weights across activations. In our layer the 99 rows already form a natural
  "batch" that shares one W — the reason A-resident + W-broadcast is the right
  first move.

**Would exceed this project's port / area budget:**
* NVDLA CBUF = **512 KB in 16 × 32 KB banks with 2 write + 3 read ports**; the
  Convolution DMA uses 512-bit interfaces with burst ≤ 4. None of that fits
  `2.25 mm²` with only `16b×128/1024/4096` macros, and the competition does not
  specify a wide bus. A "longer burst" only helps the **load** stage
  (7 872 / 154 240 = 5.1 %), so it is not the throughput lever.
* Gemmini's full systolic mesh + banked scratchpad **and** a separate banked
  accumulator with adders + TLB + ROB + two DMAs is a far larger area than the
  budget; only its **scheduling** concepts transfer, not its memory system.
* NVDLA is sized for int8/int16 with 1024 MACs and sparse/Winograd features;
  Winograd is irrelevant to a dense `64→256` FC, and sparse compression needs
  encoders/ports we cannot afford.

## 7. Exact commands and exits

```bash
bash hardware/experiments/streaming_mac/run.sh          # exit 0
#   builds: baseline (existing TB) + dual TB + dual-CACC unit microtest
#   runs:   tiny/ragged/twotile/overflow/full FFN1 shapes
#   writes: hardware/experiments/streaming_mac/results.json, logs/, model/model_12layer.json

python3 hardware/experiments/streaming_mac/model/event_schedule.py \
    --baseline-rtl 154240 --dual-rtl 125603            # exit 0 (MODEL ONLY)
```

Last run (this worktree): `run.sh` exit **0**; `PASS streaming-mac experiment:
5 cases, unit=PASS`; baseline `full` = 154 240 (matches the committed
`staged_task_row16_summary.json`), dual `full` = 125 603, all 25 344 full-case
outputs INT32-exact.

## 8. What is NOT proven (do not claim)

* Not an E203 / System-Bus / NICE run; the load/store buses are timing models.
* Not a 150 MHz STA, not an area/power sign-off; the 150 MHz ms figures are
  conditional arithmetic on simulated cycles.
* No target SRAM macro mapping / port or bank arbitration for the CACC pages.
* The dual dataflow is proven for one layer's *integer* schedule; 12-layer
  quantised numerics and parameter switching are only modelled.
* Only a 16-row configuration is measured here; the win is schedule-level, not
  a MAC-pipeline or Fmax result.

## 9. Files

```
hardware/experiments/streaming_mac/
├── README.md                      this report
├── run.sh                         one-command reproducible run
├── run_streaming_mac.py           builds/runs both TBs, INT32 checks, results.json
├── results.json                   machine-readable results
├── logs/                          per-case run/build logs + unit + model logs
├── tb/tb_kwt_streaming_task.sv    dual-CACC task harness
├── tb/tb_kwt_dualacc_unit.sv      dual-CACC primitive microtest (reset/overlap/back-pressure)
└── model/
    ├── event_schedule.py          MODEL-ONLY 12-layer scheduler
    └── model_12layer.json         model output
```
