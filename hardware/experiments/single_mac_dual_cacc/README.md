# One-MAC-array / two-CACC-page experiment

Worktree `ds-p027-singlemac` · branch `codex/kwt-ds-singlemac-dualcacc`.
All new files live under `hardware/experiments/single_mac_dual_cacc/`. The
accepted original RTL (`kwt_row_broadcast_tile.sv`, `kwt_tile_output_stage.sv`)
and the accepted TBs are used **unmodified**; the old arithmetic core is only
instantiated by a comparison wrapper, never by the new implementation.

## 1. Short answer

The `streaming_mac` dual-CACC schedule (154 240 → 125 603 cycles) instantiates
`kwt_row_broadcast_tile` **twice**, i.e. **two complete 16-lane MAC arrays
(32 multipliers)** just to obtain two accumulator pages. That is not area
efficient.

This experiment replaces the second core with a second **accumulator page**:

* **one** shared 16-lane multiplier array (`prod_q[0..15]`),
* **two** independent `16×16×32` CACC pages selected per feed beat (`in_page`),
* bias added only at **dump/export** (`dump_bias`), never at the MAC input,
* the **same unchanged** `kwt_tile_output_stage` result buffer.

| one FFN1 layer (99×64 × 64×256 + bias), 16 rows | cycles | 16-bit multipliers |
| --- | ---: | ---: |
| old one-core baseline (serial CACC export) | 154 240 | 16 |
| old two-core dual harness | 125 603 | **32** |
| **new: one MAC array + two CACC pages** | **125 603** | **16** |
| arithmetic lower bound (model) | 114 688 | — |

The new engine reproduces the 125 603-cycle dual schedule with **half** the
elaborated multiplier cells. The gain is schedule-level and only appears when
the export fits in the compute window (`K·COLS > ROWS·COLS`, i.e. `K > 16`);
tiny/ragged shapes are 462/463 and 1261/1262 cycles respectively and are
reported, not hidden.

## 2. Dataflow

```
                     one shared multiplier feed (16 multipliers)
  a_rows[15:0] ─┐
  b_word      ──┼─►  stage 1: prod_q[r] <= a[r]*b        (registers)
  in_page     ──┘            │  prod_valid_q, page, col, mask
                             ▼
  stage 2 (per page, the only psum write port)
   psum_page0[16][16] INT32   psum_page1[16][16] INT32   + valid bits
        │ 1R dump (page selected by dump_page)
        ▼
   export register: dump_data <= (valid ? psum : 0) + dump_bias
        │
   kwt_tile_output_stage (unchanged)  ──►  out_valid/out_ready/out_data
```

## 3. Interface and exact contract

| topic | definition |
| --- | --- |
| **page ownership** | A page is *feed-owned* in the cycle `in_valid && in_ready`, while its beat sits in `prod_valid_q`, and at the following accumulate edge. A page is *export-owned* in the cycle its accepted dump request drives `dump_data` (`dump_valid_q`). One page may be fed while the other is exported; the pages never share a storage port. |
| **feed valid/ready** | `in_ready` is low while `clear[in_page]` is high or while `in_page` is export-owned (`dump_valid_q && dump_valid_page == in_page`). A beat is accepted iff `in_valid && in_ready`. |
| **dump valid/ready** | `dump_ready` is low while `clear[dump_page]` is high, while the requested page is being fed this cycle, or while that page's beat is in `prod_valid_q`. Feed has priority, so `in_ready` never depends on `dump_ready` and there is no combinational loop. |
| **clear** | `clear[p]` is synchronous and per page: it clears page `p` valid bits; a product arriving for `p` at the same edge is dropped; an export delivery for `p` is cancelled (`dump_valid` is gated by `clear`). |
| **MAC latency** | Beat accepted at edge N → products visible in cycle N+1 (`prod_q`/`prod_valid_q`) → accumulated at edge N+1 → CACC contents visible in cycle N+1. Feed-to-CACC latency = **1 cycle (2 clock edges)**. Dump request accepted in cycle M → `dump_valid`/`dump_data` in cycle M+1. |
| **dump address/data** | Request: `dump_page`, `dump_row`, `dump_col`, `dump_bias`. `dump_addr = {dump_valid_page, dump_valid_row, dump_valid_col}` is the address of the datum currently on `dump_data`. Invalid (never written or cleared) cells export `dump_bias` only. |
| **busy conflicts** | Feeding the export-owned page ⇒ `in_ready=0`. Dumping the page being fed or in the product register ⇒ `dump_ready=0`. These are the only two conflicts; they cost at most one bubble. |
| **backpressure** | Dump is a 1 request/cycle streaming port. A request with `dump_ready=0` is simply retried; request gaps produce idle `dump_valid` cycles and never reorder or corrupt data. |
| **reset** | `rst_n` is asynchronous active-low: both pages, both pipeline stages and the export register are cleared; `in_ready` and `dump_valid` are low. |

## 4. RTL evidence

### 4.1 Microtest `tb_kwt_single_mac_unit.sv` (exit 0)

`PASS single-mac unit: reset, ownership, latency, backpressure, clear`

Checks against an independent in-SV INT32 reference: reset clears both pages;
page-0/page-1 ownership; the feed→CACC latency (a fresh product is not
dumpable for exactly one cycle); `in_ready=0` for the exporting page and
`in_ready=1` for the free page; `dump_ready=0` when feeding the dump page;
64-cell export streams with request gaps; a partially cleared page exports
bias only for unwritten cells; a mid-run reset invalidates both pages; and
`dump_addr`/`dump_data` match.

### 4.2 Task harness `tb_kwt_single_mac_task.sv` vs NumPy INT32 golden

`PASS tiny / ragged / full`; every valid output equals
`(a @ w + bias) & 0xFFFFFFFF` computed by NumPy in int64. Access counts are
asserted; `feed_stall_cycles=0` and `dump_not_ready_cycles=0` for all cases.

| shape | new cycles | old 1-core | old 2-core |
| --- | ---: | ---: | ---: |
| tiny 3×4×5 | 463 | 462 | 463 |
| ragged 10×7×19 | 1 261 | 1 262 | 1 261 |
| full 99×64×256 | **125 603** | 154 240 | 125 603 |

Deep output backpressure on the new engine: tiny `+BACKPRESSURE=4` = 493
cycles, ragged `+BACKPRESSURE=3` = 1 450 cycles, both INT32-exact.

### 4.3 Elaborated multiplier count (yosys, real elaboration)

```
yosys -p 'read_verilog -sv <files>; chparam -set ROWS 16 <top>;
         hierarchy -top <top>; proc; opt_expr; opt_clean; stat'   # exit 0
```

| design | `$mul` cells (ROWS=16) |
| --- | ---: |
| new `kwt_single_mac_dual_cacc` | **16** |
| old `kwt_row_broadcast_tile` (one core) | 16 |
| old two-core comparison wrapper | **32** |

The new engine has the same multiplier count as one old core and half the old
two-core harness, while matching the two-core cycle count.

## 5. Macro / port needs (capacity, not a sign-off)

| buffer | size | 16-bit macro budget | ports |
| --- | ---: | --- | --- |
| CACC two pages 2 × 16×16×32 | 2 048 B | as registers here: 2 × 256 × 32 flops + 512 valid bits; if macro-mapped: 4 × `16b×1024` (2 per 32-bit word per page) | per page 1W accumulate (one row lane/cycle) + 1R dump; pages are disjoint so a 1R1W macro per page suffices |
| bias group N=256 | 1 024 B | 2 × `16b×256` (or a slice of the result buffer) | read at export only; added after the CACC read |
| result buffer 16×16×32 | 1 024 B | 2 × `16b×1024` (32-bit width-cascade) | 1W capture + 1R drain (`kwt_tile_output_stage`, unchanged) |

## 6. What is achievable / NOT proven

* Achievable here: an RTL-exact one-MAC-array / two-CACC schedule at the same
  125 603 cycles as the two-core variant, with 16 instead of 32 multipliers.
* **Not** a 150 MHz STA, **not** a synthesised area/power number. Only $mul
  counts, simulated cycles and byte/bit capacity are reported.
* No bank/port arbitration or SRAM-macro mapping is implemented or closed.
* Not an E203 / System-Bus / NICE run; the load/store buses are timing models.
* The 12-layer total is **model only**; see
  `hardware/experiments/streaming_mac/README.md` §5. It is not re-measured here.

## 7. Exact commands and exits

```bash
bash hardware/experiments/single_mac_dual_cacc/run.sh      # exit 0
#   builds: new unit microtest + new task TB + old 1-core baseline + old 2-core dual
#   runs:   unit; tiny/ragged/full against NumPy INT32 golden; backpressure tiny/ragged
#   counts: yosys $mul cells (new, old 1-core, old 2-core)
#   writes: results.json, logs/
```

Last run (this worktree): `run.sh` exit **0**; unit **PASS**; tiny/ragged/full
INT32-exact; full new = 125 603; `$mul`: new 16, old 1-core 16, old 2-core 32.

## 8. Files

```
hardware/experiments/single_mac_dual_cacc/
├── README.md                        this report
├── run.sh                           one-command reproducible run (exit 0)
├── run_single_mac_dual_cacc.py      builds/runs TBs, NumPy golden, $mul counts, results.json
├── results.json                     machine-readable results
├── rtl/kwt_single_mac_dual_cacc.sv  NEW one-MAC-array / two-CACC-page RTL
├── tb/tb_kwt_single_mac_task.sv     new task harness (uses unchanged output stage)
├── tb/tb_kwt_single_mac_unit.sv     reset/ownership/latency/backpressure/clear microtest
├── area/kwt_two_core_compare_wrapper.sv   COMPARISON ONLY: 2× old core, for $mul counting
└── logs/                            build/run/unit/yosys logs
```
