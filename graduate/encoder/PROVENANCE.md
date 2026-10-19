# 来源与改动记录（provenance）

本目录的 RTL/模型为**只读复制**，来源为本次任务允许依赖的三个前置分支终态。
复制后**仅**做了两处与数值无关的改动：（RTL 文件零改动；三个 python 文件只把
“仓库根定位”改为向上搜索 `software/weights/kwt1_fp32.npz`，使其在更浅的
`graduate/encoder/` 路径下可用）。数值代码逐行未改。

> 注：`codex/kwt-grad-nonlinear` 在本目录开始拼接之后、由**其自身 worker** 追加了提交
> `3084222`（把 `kwt_postnorm_ffn` 的 `D/HID` 从 cfg localparam 改为模块 parameter，修
> ANSI 端口表以便 Icarus elaborate）。本目录随即从 `3084222` 重新复制
> `kwt_postnorm_ffn.sv`、`kwt_postnorm_cfg.svh` 两个文件；其余文件的来源仍为 `10aa202`
> 与 `405c876`/`c9a4307`。两次内容在 Verilator 下功能一致（`run.sh` 产物哈希不变）。

## 来源分支（终态提交）

| 分支 | 提交（SHA） |
| --- | --- |
| `codex/kwt-grad-attention` | `405c8768b9ff44f8faaba3e3e76d532ce2641473` |
| `codex/kwt-grad-nonlinear` | `3084222b0247e525eb3c0061134a1fc88d69bff6` |
| `codex/kwt-grad-golden`    | `c9a430734901fc761e8160986400a154f59a3bb3` |

## 复制文件与哈希

### RTL（逐字节相同）

| 目标（本分支） | 源 | 源 sha256 = 目标 sha256 |
| --- | --- | --- |
| `graduate/rtl/encoder/third_party/attn_single_head.sv` | grad-attention `graduate/rtl/attention/rtl/attn_single_head.sv` | `71640553…408f24` |
| `graduate/rtl/encoder/third_party/attn_params.svh` | grad-attention `…/rtl/attn_params.svh` | `6dc3e3fd…d3b064` |
| `graduate/rtl/encoder/third_party/kwt_postnorm_pkg.sv` | grad-nonlinear `…/postnorm_ffn/rtl/kwt_postnorm_pkg.sv` | `387171bd…fc5507` |
| `graduate/rtl/encoder/third_party/kwt_residual_add.sv` | grad-nonlinear `…/rtl/kwt_residual_add.sv` | `914bd169…51a173` |
| `graduate/rtl/encoder/third_party/kwt_layernorm.sv` | grad-nonlinear `…/rtl/kwt_layernorm.sv` | `9270bd19…374edf` |
| `graduate/rtl/encoder/third_party/kwt_gelu.sv` | grad-nonlinear `…/rtl/kwt_gelu.sv` | `affeeb85…cb1bb` |
| `graduate/rtl/encoder/third_party/kwt_ffn2.sv` | grad-nonlinear `…/rtl/kwt_ffn2.sv` | `87b179b7…78f4e` |
| `graduate/rtl/encoder/third_party/kwt_postnorm_ffn.sv` | grad-nonlinear `…/rtl/kwt_postnorm_ffn.sv` | `a13ac115…78f8005` |
| `graduate/rtl/encoder/third_party/kwt_postnorm_cfg.svh` | grad-nonlinear `…/rtl/kwt_postnorm_cfg.svh` | `5179f131…369a899` |

（RTL 文件名后带 `…` 为省略前缀；完整值见 `WORKER_REPORT.md` / `sha256sum` 输出。）

### Python（仅“仓库根定位”改动）

| 目标 | 源 | 源 sha256 | 目标 sha256 | 改动 |
| --- | --- | --- | --- | --- |
| `graduate/encoder/model/attn_int_model.py` | grad-attention `graduate/rtl/attention/model/attn_int_model.py` | `c3b2b674…8a4029` | `2baad345…57c5a` | `WSRC/XSRC` 根定位 |
| `graduate/encoder/model/kwt_postnorm_ref.py` | grad-nonlinear `…/model/kwt_postnorm_ref.py` | `2174dc55…ea86b5` | `0b33fc61…009ff7` | `ROOT` 根定位 |
| `graduate/encoder/reference/kwt1_fp32_golden.py` | grad-golden `graduate/reference/kwt1_fp32_golden.py` | `dd8c98a8…834d64` | `15779887…a10531` | `REPO_ROOT` 根定位 |

## 本次新增（非复制）

- `graduate/rtl/encoder/kwt_ffn1_row.sv`
- `graduate/rtl/encoder/kwt_encoder_layer.sv`
- `graduate/rtl/encoder/kwt_encoder_layers_ctrl.sv`
- `graduate/rtl/encoder/kwt_encoder_top.sv`
- `graduate/encoder/model/verify_encoder.py`
- `graduate/encoder/vectors/gen_vectors.py`
- `graduate/encoder/tb/tb_encoder.sv`、`tb_layers_smoke.sv`、`tb/run.sh`、`tb/run_smoke.sh`
- `graduate/encoder/README.md`、`SOFTWARE_BOUNDARY.md`、`PROVENANCE.md`

## 未改动

- 未修改任何来源分支、主目录、`hardware/`、`software/`、`docs/` 等既有本科核心。
- 未 `merge`/`rebase`/`cherry-pick`/`push`，未合并到 `main`。
- 未开子智能体。
