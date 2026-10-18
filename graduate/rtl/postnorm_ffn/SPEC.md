# PostNorm FFN / GELU / residual / LayerNorm — numeric specification

Everything below is frozen for **encoder layer 1** (`enc_layers/0`). All scales
and coefficients are produced by [`model/gen_config.py`](model/gen_config.py) and
re-derived independently by [`model/kwt_postnorm_ref.py`](model/kwt_postnorm_ref.py).

## 1. Official PostNorm order

Taken from the ARM KWT-1 sources in this repository and from `software/run.py`
(`prenorm=False`; `heads=1`, `d_model=64`, `mlp_dim=256`, `eps=1e-6`):

```
attn_out = combine_heads( softmax(Q K^T / sqrt(64)) V )
r1       = x + attn_out                       # residual 1
out1     = layernorm1(r1)                     # LayerNorm 1  -> FFN1 input
h        = out1 @ W1 + b1                     # FFN1 (64 -> 256), existing RTL
g        = gelu(h)                            # erf GELU, TFA default
f        = g  @ W2 + b2                       # FFN2 (256 -> 64)
r2       = out1 + f                           # residual 2
y        = layernorm2(r2)                     # LayerNorm 2  -> encoder output
```

LayerNorm is population variance (`mean of squares of deviations`), matching
`tf.keras.layers.LayerNormalization(epsilon=1e-6)`.

## 2. Quantization

| symbol | meaning | value (layer 1) | how frozen |
| --- | --- | ---: | --- |
| `s_x` | layer input `x` | 0.00135625 | per-tensor static max / 2047 (this sample) |
| `s_a` | attention output | 0.00027125 | per-tensor static max / 2047 (this sample) |
| `sx` | FFN1 input (`out1`) | 0.00243742 | **frozen A12W10 `layer01_activation_scale`** (training split) |
| `s_h[c]` | FFN1 hidden `h` | `sx * s_w1[c]` | per-output-channel weight scales of the frozen pack |
| `s_g` | GELU output / FFN2 input | 0.00095650 | per-tensor static max / 2047 |
| `s_h2[k]` | FFN2 output `f` | `s_g * s_w2[k]` | per-output-channel `W2` scales |
| `s_r1` | residual 1 sum | 0.00144993 | per-tensor static max / 2047 |
| `s_r2` | residual 2 sum | 0.00207154 | per-tensor static max / 2047 |
| `s_out` | encoder output `y` | 0.00183158 | per-tensor static max / 2047 |

Activation and weight containers are the project convention: **A12W10** (12-bit
activations, 10-bit weights, zero point 0, stored in INT16 lanes), symmetric
`s = max|.| / (2^(bits-1) - 1)`, INT32 accumulators. `W1`/`b1` come verbatim
from the frozen `software/results/ffn1_a12w10_static_sym.npz` pack, so the
`h` fed to this block is the same integer vector the existing FFN1 RTL produces.
`W2`/`b2` are quantized here with the identical rule.

Rounding of *coefficients* is round-half-to-even; rounding inside the datapath is
`mul_sr` (round-half-up on the signed product, i.e. `(a*m + 2^(q-1)) >>> q`).

## 3. Integer datapath

All blocks saturate activations symmetrically to **[-2047, +2047]**
(`CLAMP_A`, the effective A12 range).

### 3.1 residual add (`kwt_residual_add`)

```
out[i] = sat( mul_sr(a[i], ma, 16) + mul_sr(b[i], mb[i], 16) )
```

`ma = round(s_a/s_r * 2^16)`, `mb[i] = round(s_b[i]/s_r * 2^16)`. Residual 1 adds
two INT16 vectors (A_W = B_W = 16); residual 2 adds an INT16 vector and the INT32
FFN2 output (A_W = 16, B_W = 32). Rounding each operand separately (instead of
the exact sum) is the only approximation introduced by the two-step add.

### 3.2 LayerNorm (`kwt_layernorm`), D = 64

```
sum   = Σ x[i]
mean  = sum  >> 6                       (floor)
ss    = Σ (x[i]-mean)^2
var   = ss   >> 6                       (floor, population variance)
Vq    = (var << 16) + eps_q             eps_q = round(1e-6 / s_ln^2 * 2^16)
sq    = isqrt(Vq)                       (restoring integer sqrt, floor)
isq   = min( (2^(16+8) + sq/2) / sq , 2^24 - 1 )   ; isq = 2^24-1 if sq == 0
t[i]  = (x[i]-mean) * gcoef[i]          gcoef[i] = round(gamma[i] * 2^16)
u[i]  = mul_sr(t[i], isq, 16)           == (x-mean)*gamma / sqrt(var+eps) * 2^16
o[i]  = mul_sr(u[i], mo, 16) + bcoef[i] mo = round(2^16 / (2^16 * s_out)) = round(1/s_out)
                                        bcoef[i] = round(beta[i] / s_out)   (no fractional factor:
                                        it is added directly in the s_out units of `o`)
y[i]  = sat(o[i])
```

`eps_q` is computed from the *integer* input scale (`eps / s_ln^2`), which is
what makes `sqrt(var_q + eps_q)` dimensionally correct. When the input is
perfectly constant (`var = 0`, `eps_q > 0`) `isq` is finite; when `sq` would be
zero it saturates to `ISQ_MAX`, so no division by zero is possible.

### 3.3 GELU (`kwt_gelu`)

The FFN1 hidden vector is in a per-channel scale, so it is first moved into a
single nonlinear domain with 8 fractional bits (Q8 real units):

```
v   = clamp( mul_sr(h, MG[c], 28), -2048, 2047 )    MG[c] = round(s_h[c] * 2^36)
idx = v + 2048                                       idx in [0, 4095]
i   = idx >> 4 ; frac = idx & 15
g8  = LUT[i] + (((LUT[i+1] - LUT[i]) * frac) >> 4)   LUT[i] = round(gelu((i-128)/16) * 256)
g   = sat( mul_sr(g8, MGN, 16) )                     MGN = round(2^16 / (2^8 * s_g))
```

The LUT has 257 nodes on `[-8, +8)` with step 1/16 and linear interpolation; the
per-channel `MG[c]` is the only per-hidden-channel coefficient. `g` is the FFN2
activation in scale `s_g`.

### 3.4 FFN2 (`kwt_ffn2`)

```
acc[k] = Σ_{j<256} g[j] * w2[k][j]      (48-bit signed accumulator)
f[k]   = sat_int32( acc[k] + b2_int[k] )
```

`w2` is INT16 in the `k*256 + j` layout (one `$readmemh` line per output
channel); `b2_int[k] = round(b2[k] / (s_g * s_w2[k]))`. Worst-case magnitude is
`256 * 2047 * 1023 ≈ 5.4e8`, well inside INT32 and INT48, so the accumulator
cannot wrap for any A12W10 input.

## 4. Bit-exact reproduction

`model/kwt_postnorm_ref.py` implements sections 3.1–3.4 in plain Python (own
restoring `isqrt`, own round-half-to-even, own LUT evaluation) and is the oracle
for the RTL dumps. `model/verify.py` additionally re-derives every coefficient
from the checkpoint and requires it to equal the value baked into
`rtl/kwt_postnorm_cfg.svh`, and re-evaluates FFN1 from the quantized input to
require it to equal the `h` handed to the RTL.
