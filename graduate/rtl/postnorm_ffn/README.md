# KWT-1 PostNorm FFN / GELU / residual / LayerNorm prototype

Synthesizable-RTL prototype of the **single-layer encoder** item that the
undergraduate work does not cover ([`题目/640.jpg 9`](../../../题目/640.jpg%209)
item 5): the GELU, the second FFN (256 → 64), the two residual additions and the
two LayerNorms of the official **PostNorm** encoder, plus an independent integer
and FP32 reference.

Scope boundary: the attention block and the FFN1 (64 → 256) matmul are **not**
in this directory. FFN1 already exists (`hardware/rtl/pulse_array/`); this block
plugs in directly behind it. No existing RTL was modified.

## Layout

| path | contents |
| --- | --- |
| `SPEC.md` | frozen numeric specification (scales, rounding, integer datapath) |
| `rtl/kwt_postnorm_pkg.sv` | `mul_sr`, integer `isqrt`, symmetric saturation |
| `rtl/kwt_residual_add.sv` | one residual addition with per-lane scale alignment |
| `rtl/kwt_layernorm.sv` | integer LayerNorm over `d_model = 64` |
| `rtl/kwt_gelu.sv` | Q8 rescale + 257-node interpolated GELU LUT + requantization |
| `rtl/kwt_ffn2.sv` | 256 → 64 INT32 accumulation, loadable weight/bias SRAM |
| `rtl/kwt_postnorm_ffn.sv` | the whole fragment, row sequencer, valid/ready |
| `rtl/kwt_postnorm_cfg.svh` | **generated** coefficients + GELU LUT |
| `model/gen_config.py` | calibrate from the real checkpoint, emit config + vectors |
| `model/kwt_postnorm_ref.py` | independent integer model + FP32 reference |
| `model/verify.py` | bit-exact comparison of RTL dumps vs the reference |
| `tb/tb_kwt_postnorm_ffn.sv`, `tb/run.sh` | Icarus testbench and driver |
| `tb/run_yosys.sh` | Yosys `check -assert` structural check |

## Interfaces

Everything is valid/ready and stalls correctly under backpressure. One row
(`99 × 64`) is processed at a time.

| port | direction | payload | scale |
| --- | --- | --- | --- |
| `x_valid/x_ready/x_data` | in | one 64-lane INT16 row | `s_x` (layer input) |
| `a_valid/a_ready/a_data` | in | one 64-lane INT16 row | `s_a` (attention output) |
| `h_valid/h_ready/h_data` | in | one INT32 per beat, 256 beats/row | `s_h[c]` (FFN1 output) |
| `o1_valid/o1_ready/o1_data` | out | one 64-lane INT16 row | `sx` (LN1 output → FFN1) |
| `y_valid/y_ready/y_data` | out | one 64-lane INT16 row | `s_out` (encoder output) |
| `w_we/w_addr/w_wdata` | in | 16 384 × INT16 | FFN2 weights, `k*256 + j` |
| `b_we/b_addr/b_wdata` | in | 64 × INT32 | FFN2 bias |

`h` is consumed one hidden channel per beat so the block can sit behind the
existing FFN1 output stage; `o1` is handed to the FFN1 input loader. Tensor
storage inside the block is a 256 × INT16 GELU buffer plus the loadable FFN2
weight SRAM (32 KiB) and bias (256 B), i.e. the per-layer parameter volume the
project already budgets for FFN1.

Quantization: A12W10 activations/weights (INT16 lanes, symmetric, zero point 0),
INT32 accumulators, per-tensor activation scales and per-output-channel weight
scales, exactly the convention of `software/results/ffn1_a12w10_static_sym.npz`.
See `SPEC.md` for every scale and every rounding rule.

## Reproduce

```bash
cd graduate/rtl/postnorm_ffn
bash tb/run.sh          # regenerate vectors, Icarus sim, bit-exact check
bash tb/run_yosys.sh    # Yosys check -assert (needs OSS CAD Suite Yosys >= 0.6)
```

`tb/run.sh` runs 109 rows: the **99 real rows** of the frozen validation sample
(`sample_00_encoder.npy` → real attention → real FFN1 pack → this block) plus
**10 extreme rows** (±2047 activations, ±INT16 saturation, exact-zero variance,
near-zero non-zero variance, all zero, INT32 min/max hidden, alternating
extremes). It asserts `rst_n` between the real and extreme rows and checks that
every output is deasserted while reset is low. `o1_ready`/`y_ready` are driven
by an LFSR, so every row is exercised under random backpressure.

`model/verify.py` requires bit-exact equality of `r1`, `out1`, `g`, `f`, `r2`,
`y` between the RTL dumps and the independent integer model, and additionally
re-derives every coefficient and the FFN1 hidden vector from the checkpoint.

## Evidence level and limits

This is **L1** evidence: RTL + testbench simulation with behavioural memories.
Explicitly **not** proven here:

- no synthesis/timing/area/power sign-off (no target PDK, SRAM macros or SDC);
  `tb/run_yosys.sh` only shows the RTL elaborates and infers no latches;
- no SoC / NICE / DMA integration, no per-layer parameter reload, no
  cross-row pipelining (one row at a time, ~530 cycles/row);
- the FFN2 weight bandwidth assumes 64 parallel reads per cycle from a
  behavioural array; a real macro would need 64 banks or time multiplexing;
- the FP32 reference is a NumPy re-implementation of `software/run.py`
  (PyTorch is not installed here), not an independent framework run;
- only encoder layer 1 is calibrated; other layers have different scales;
- the official 600-sample evaluation set is not available.
