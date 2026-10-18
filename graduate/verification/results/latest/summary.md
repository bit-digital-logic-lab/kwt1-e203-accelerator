# Graduate verification stage run

- generated (UTC): 2026-10-18T07:56:49+00:00
- design root: `/home/yang/kwt-worktrees/grad-test-infra`
- design commit: `3d9ae98452e8c307243e4db51c019d718f7a4c7e`
- counts: {'PASS': 5, 'SKIP': 3, 'FAIL': 0}

| stage | class | status | reason | exit | duration s |
| --- | --- | --- | --- | ---: | ---: |
| attention | rtl | **SKIP** | design module(s) absent: /home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/attention/kwt_attention.sv | None | 0.0 |
| encoder1 | rtl | **SKIP** | design module(s) absent: /home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/encoder/kwt_encoder_layer.sv | None | 0.0 |
| encoder12 | rtl | **SKIP** | design module(s) absent: /home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/encoder/kwt_encoder_layer.sv, /home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/encoder/kwt_encoder_top12.sv | None | 0.0 |
| e203 | soc | **PASS** | checked=1 | 0 | 0.946 |
| synth | synth | **PASS** | checked=4 | 0 | 9.28 |
| smoke_rtl_vector | rtl | **PASS** | checked=2 | 0 | 1.659 |
| smoke_golden_faults | rtl | **PASS** | checked=1 | 0 | 3.291 |
| smoke_software_golden | software | **PASS** | checked=19008 | 0 | 0.107 |

## Per-stage evidence

### attention (rtl) — SKIP

- reason: design module(s) absent: /home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/attention/kwt_attention.sv
- command: `bash /home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/attention/tb/run.sh`
- exit code: None
- duration s: 0.0
- sample identity: KWT-1 single-head attention: Q/K/V projection, 99x99 scores, row softmax, probability-weighted values, output projection (dim 64)
- comparison: RTL vs an independent golden authored in the attention branch
- golden: independent golden required from the harness (never derived from DUT output)

### encoder1 (rtl) — SKIP

- reason: design module(s) absent: /home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/encoder/kwt_encoder_layer.sv
- command: `bash /home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/encoder/tb/run_single_layer.sh`
- exit code: None
- duration s: 0.0
- sample identity: One PostNorm encoder layer: attention + 2 residual adds + 2 LayerNorm + 64->256->64 FFN + GELU
- comparison: RTL vs the design software numeric model, per-layer intermediate outputs
- golden: independent golden required from the harness (never derived from DUT output)

### encoder12 (rtl) — SKIP

- reason: design module(s) absent: /home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/encoder/kwt_encoder_layer.sv, /home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/encoder/kwt_encoder_top12.sv
- command: `bash /home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/encoder/tb/run_twelve_layers.sh`
- exit code: None
- duration s: 0.0
- sample identity: All 12 encoder layers with layer-to-layer data passing and per-layer parameter update
- comparison: RTL vs the software numeric model across all 12 layers
- golden: independent golden required from the harness (never derived from DUT output)

### e203 (soc) — PASS

- reason: checked=1
- command: `bash /home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/control/tb/run_e203.sh`
- exit code: 0
- duration s: 0.946
- sample identity: Real E203 CPU executes the KWT program and drives the accelerator over the NICE interface (simplified SoC model, 128 outputs)
- comparison: E203 observed NICE request/response and outputs, cross-checked by the TB against the frozen tile golden
- golden: independent tile golden compiled into tb_kwt_real_e203.sv
- input hashes:
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/control/tb/run_e203.sh` sha256=f11c5be7487d14b07a49de45c6d28511bc212b9a12a6d4b2612683669684f7c4
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/control/tb/tb_kwt_real_e203.sv` sha256=b75179f8bab260dfc8994ac48d974beac89b1efadd6896486b259b2620136c48
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/control/tb/e203_kwt_program.S` sha256=588020f9f72bb90d1a7aa81b77e54c03da259b7c8a1507282b417392bca88598
- design module hashes:
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/control/tb/run_e203.sh` sha256=f11c5be7487d14b07a49de45c6d28511bc212b9a12a6d4b2612683669684f7c4
- log: `/home/yang/kwt-worktrees/grad-test-infra/graduate/verification/results/latest/e203.log`
- output tail:

```
PASS real E203 NICE: requests=1 responses=1 outputs=128 cycles=1055
/home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/control/tb/tb_kwt_real_e203.sv:80: $finish called at 10740 (1s)
```

### synth (synth) — PASS

- reason: checked=4
- command: `/usr/bin/python3 /home/yang/kwt-worktrees/grad-test-infra/graduate/verification/synth/yosys_evidence.py --design-root /home/yang/kwt-worktrees/grad-test-infra --yosys /home/yang/yosys-sta/oss-cad-suite/bin/yosys`
- exit code: 0
- duration s: 9.28
- sample identity: kwt_row_broadcast_tile compute core: 128 (here 8) multipliers, CACC banks, bias add; optional standard-cell mapping of kwt_cacc
- comparison: Yosys elaboration + structural cell counts; optional mapped area with an example library (NOT a timing/area signoff)
- golden: structural expectations ($mul >= 1, $mem_v2 >= 1, check -assert clean)
- input hashes:
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/pulse_array/kwt_product_array.sv` sha256=f3f1b91c1bd962c5fbc39dff700135ae7f1d2e1f9198b7a44091ea28c9e78fa3
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/pulse_array/kwt_cacc.sv` sha256=0febfb08aa977856867bbfb3760830b4292a8138a365194e69a0b088c9ca9da8
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv` sha256=6e1dbac38b1988972b0358aea626eebf5f989527262449ad1d8adf4d902ac117
- design module hashes:
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/pulse_array/kwt_product_array.sv` sha256=f3f1b91c1bd962c5fbc39dff700135ae7f1d2e1f9198b7a44091ea28c9e78fa3
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/pulse_array/kwt_cacc.sv` sha256=0febfb08aa977856867bbfb3760830b4292a8138a365194e69a0b088c9ca9da8
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv` sha256=6e1dbac38b1988972b0358aea626eebf5f989527262449ad1d8adf4d902ac117
- log: `/home/yang/kwt-worktrees/grad-test-infra/graduate/verification/results/latest/synth.log`
- output tail:

```
PASS elaboration + check -assert clean for top kwt_row_broadcast_tile
  structural counts: $mul=8 $mem_v2=8 $add=9
PASS $mul count 8 >= 1
PASS $mem_v2 count 8 >= 1
PASS kwt_cacc mapped with example library Nangate45_typ.lib: Chip area=47861.114000 (NOT a sign-off area)
KWT-GRAD-VERIFY synth checked=4 failed=0
```

### smoke_rtl_vector (rtl) — PASS

- reason: checked=2
- command: `/usr/bin/python3 /home/yang/kwt-worktrees/grad-test-infra/hardware/testbenchplus/common/run_vectors.py`
- exit code: 0
- duration s: 1.659
- sample identity: testbenchplus example vectors (layer0_fp32, layer0_int16); explicit, author-supplied expected matrices, not real KWT-1 layer data
- comparison: integer: bit-exact; fp32: NRMSE <= 3% and max abs error, against the vector 'expected' golden
- golden: independent 'expected' matrices in vectors/*/case.json
- input hashes:
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/testbenchplus/common/run_vectors.py` sha256=be710e9772322ca58b75415baae80ff5baa39e20e9e09bea7159afef3cf18b1a
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/testbenchplus/common/tb_array_vector.sv` sha256=01c181d61e6698af7ad1d8a771258fb7bc76ab539362105f7ad8e63f06e6ec1b
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/pulse_array/kwt_pe.sv` sha256=17794b9f88bcdbc90e83d64a2634de80b60d400f1a8cf371ee3de6f04dc528fa
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/rtl/pulse_array/kwt_pulse_array.sv` sha256=d2aaf5c1e7a59e805b164406a08678fab2439255db2067ddad3f6879ea1ae30d
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/testbenchplus/vectors/layer0_fp32/case.json` sha256=12f395698b873fd7e59436d49dd2e4e1b58cc7d9231fdfd3f705d5c476efe90e
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/testbenchplus/vectors/layer0_int16/case.json` sha256=7460b92a6eecd8df2a3d8264f056b00dc26e3c0a85a6a01307df693d890d1d32
- design module hashes:
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/testbenchplus/common/run_vectors.py` sha256=be710e9772322ca58b75415baae80ff5baa39e20e9e09bea7159afef3cf18b1a
- log: `/home/yang/kwt-worktrees/grad-test-infra/graduate/verification/results/latest/smoke_rtl_vector.log`
- output tail:

```
layer0_fp32: 2x2 * 2x3 fp32
  FP32 NRMSE=0.00000000% max_abs=0 limit=3.00%
  status=2 total_cycles=84 compute_cycles=8 wait_cycles=74 dma_cycles=0
layer0_int16: 2x2 * 2x3 int16
  integer exact=6/6 accuracy=100.00%
  status=2 total_cycles=84 compute_cycles=8 wait_cycles=74 dma_cycles=0
result: 2/2 passed
```

### smoke_golden_faults (rtl) — PASS

- reason: checked=1
- command: `/usr/bin/python3 /home/yang/kwt-worktrees/grad-test-infra/hardware/testbenchplus/common/negative_injection_check.py`
- exit code: 0
- duration s: 3.291
- sample identity: fault injection on both testbenchplus vectors: original golden must pass, golden with one element +1 must be rejected with non-zero exit
- comparison: harness self-check: corrupted independent golden must produce FAIL
- golden: the same vectors/*/case.json goldens, deliberately corrupted
- input hashes:
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/testbenchplus/common/negative_injection_check.py` sha256=17c4f5f81b283b78a3d9360f2e9c5ace5f448b215c9831f700584885188d30df
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/testbenchplus/common/run_vectors.py` sha256=be710e9772322ca58b75415baae80ff5baa39e20e9e09bea7159afef3cf18b1a
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/testbenchplus/vectors/layer0_fp32/case.json` sha256=12f395698b873fd7e59436d49dd2e4e1b58cc7d9231fdfd3f705d5c476efe90e
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/testbenchplus/vectors/layer0_int16/case.json` sha256=7460b92a6eecd8df2a3d8264f056b00dc26e3c0a85a6a01307df693d890d1d32
- design module hashes:
  - `/home/yang/kwt-worktrees/grad-test-infra/hardware/testbenchplus/common/negative_injection_check.py` sha256=17c4f5f81b283b78a3d9360f2e9c5ace5f448b215c9831f700584885188d30df
- log: `/home/yang/kwt-worktrees/grad-test-infra/graduate/verification/results/latest/smoke_golden_faults.log`
- output tail:

```
PASS negative injection layer0_fp32: detected with exit=1
PASS negative injection layer0_int16: detected with exit=1
PASS testbenchplus negative-injection self-check (2 vectors)
```

### smoke_software_golden (software) — PASS

- reason: checked=19008
- command: `/usr/bin/python3 /home/yang/kwt-worktrees/grad-test-infra/graduate/verification/golden/check_encoder_input.py --design-root /home/yang/kwt-worktrees/grad-test-infra`
- exit code: 0
- duration s: 0.107
- sample identity: validation sample 00 (four/750e3e75_nohash_0.wav, label _unknown_): 98x40 MFCC -> 99x64 layer-0 encoder input
- comparison: FP32 max abs / NRMSE between the design convert() result, an independent numpy reference and the stored TF-checkpoint golden
- golden: stored sample_00_encoder.npy produced by the TensorFlow checkpoint path
- input hashes:
  - `/home/yang/kwt-worktrees/grad-test-infra/software/offline_encoder_input.py` sha256=5f68e3b576e7f9bd539143b129bd607a005c159048d46f4adb1f16f53b683659
  - `/home/yang/kwt-worktrees/grad-test-infra/software/weights/kwt1_fp32.npz` sha256=8f359abf7adbe287b8f89fca98af46194f98f90f2146784731bff551a155825a
  - `/home/yang/kwt-worktrees/grad-test-infra/software/validation/real_input_chain_20261017/sample_00_mfcc.npy` sha256=a8a85e2878283a0d3cd40f157243b10aa5034c2544bbfb21d292da9c422aad15
  - `/home/yang/kwt-worktrees/grad-test-infra/software/validation/real_input_chain_20261017/sample_00_encoder.npy` sha256=489dc7c28b2816cdb7cfc35e3f98c2dd674112f71d9357f746db57de4a27c058
- design module hashes:
  - `/home/yang/kwt-worktrees/grad-test-infra/software/offline_encoder_input.py` sha256=5f68e3b576e7f9bd539143b129bd607a005c159048d46f4adb1f16f53b683659
  - `/home/yang/kwt-worktrees/grad-test-infra/software/weights/kwt1_fp32.npz` sha256=8f359abf7adbe287b8f89fca98af46194f98f90f2146784731bff551a155825a
  - `/home/yang/kwt-worktrees/grad-test-infra/software/validation/real_input_chain_20261017/sample_00_mfcc.npy` sha256=a8a85e2878283a0d3cd40f157243b10aa5034c2544bbfb21d292da9c422aad15
  - `/home/yang/kwt-worktrees/grad-test-infra/software/validation/real_input_chain_20261017/sample_00_encoder.npy` sha256=489dc7c28b2816cdb7cfc35e3f98c2dd674112f71d9357f746db57de4a27c058
- log: `/home/yang/kwt-worktrees/grad-test-infra/graduate/verification/results/latest/smoke_software_golden.log`
- output tail:

```
sample=sample_00_mfcc.npy shape=(99, 64) dtype=float32
golden=/home/yang/kwt-worktrees/grad-test-infra/software/validation/real_input_chain_20261017/sample_00_encoder.npy corrupt=False
  max_abs converted_vs_independent=0
  max_abs converted_vs_golden=4.76837158e-07
  max_abs independent_vs_golden=4.76837158e-07
  converted_vs_golden_nrmse_pct=7.84498907e-06
  independent_vs_golden_nrmse_pct=7.84498907e-06
{"checks": {"converted_vs_independent": 0.0, "converted_vs_golden": 4.76837158203125e-07, "independent_vs_golden": 4.76837158203125e-07}, "tolerance": 1e-05}
PASS software golden: design convert, independent numpy and stored TF golden agree within tolerance
KWT-GRAD-VERIFY encoder_input checked=19008 failed=0
```
