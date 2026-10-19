# encoder_12numeric — 12 层不同权重的数值链（独立参考 + 样本测试）

本目录是该独立工作者的**唯一**产物目录（任务约定：只写
`graduate/ds_tasks/encoder_12numeric`）。它补齐被现有研究生报告明确标为“未证”
的一项：**12 层编码器的数值正确性**——而不是 12 层调度 smoke。

被纠正的结论（来自 `graduate/encoder/README.md` / 根 `WORKER_REPORT.md`）：

> “12 层 smoke 只证**调度**，不证 12 层数值；第 1–11 层用的是第 0 层的尺度与权重 ROM。”

本任务用 **12 组互不相同的权重**和**逐层训练集标定的尺度**，跑通:

1. **L1 软件参考**：独立整数 12 层编码器链（`model/ref12.py` + `model/chain.py`），
   训练划分标定、验证划分只评估，与 FP32 golden 逐层比较 NRMSE；
2. **L2 RTL 逐位**：把官方 RTL 编码器**逐层重新精化**（每层各自的
   `attn_params.svh` / `kwt_postnorm_cfg.svh` / `M_A`），层间 **RTL 输出直接作为
   下一层输入**，逐层逐级与 L1 整数参考**逐位比较**。

> 这里的“12 层不同权重”是**参数级**的：每一层的注意力 Wq/Wk/Wv/Wo/bias、
> FFN1 W1/b1、FFN2 W2/b2、LayerNorm gamma/beta、以及全部尺度/移位/系数都不同，
> 见 `results.json.audit.layers_have_distinct_weights` 与
> `layers_have_distinct_scales`。

## 目录

```
graduate/ds_tasks/encoder_12numeric/
  model/ref12.py            独立整数原语 + 定点契约（不读 RTL 输出）
  model/chain.py            12 层标定 + 独立整数链（向量化，与标量自测逐位一致）
  model/run_l1.py           L1 驱动：训练标定 → 冻结 → 验证评估（仅评估）
  model/freeze_int.py       确定性重算并冻结整数张量包，校验尺度/M_A 一致
  model/gen_rtl_vectors.py  生成逐层 SV 配置 + hex 激励
  model/run_rtl_chain.py    L2 驱动：逐层精化 RTL、RTL 链式运行、逐位比较
  tb/tb_chain_layer.sv      本任务自有 TB（不改动 graduate/encoder/tb）
  run_all.sh                一键复现（三段 + 退出码）
  logs/                     真实运行日志
  frozen_12layer_calib.json 冻结标定（尺度/系数/LUT/证据字段）
  frozen_12layer_int.npz    冻结整数权重/偏置（12 层）
  results.json              L1 结果与证据层级
  rtl_results.json          L2 结果与证据层级
```

## 数值方案（无梯度，训练标定 / 验证只评估）

- 校准划分：`software/.build/quant_samples_12labels_20.npz` 的 **training** 划分
  （240 条 MFCC）。编码器输入由冻结 ckpt 的 `patch_proj/pos_emb/class_emb`
  投影得到（`software/run.py` 语义）。
- 每层每个张量静态对称：`s = max_训练|stage| / (2^(A-1)-1)`，零点 0，A=12。
- **层间尺度连续**：`s_x[k] = s_out[k-1]`，使 `y_k` 的整数结果正好是
  `x_{k+1}` 的整数输入——与 RTL 控制器的 `y → x_buf` 拷贝语义一致。
- 注意力权重 10 bit/张量、移位 `RQ/…/RO`、`KC/SH_KC`、`M_A=round(so/s_a·2^16)`
  均由**训练划分**冻结；GELU LUT 为解析式 257 点。
- 验证划分（240 条 dataset validation + 6 条真实 WAV 的 encoder 输入）
  **不参与**任何尺度、移位、LUT、阈值或 `M_A` 的选择。

## L1 结果（独立整数链 vs FP32 golden；246 条验证样本）

逐层编码器输出 `y` 的 NRMSE（对每层取验证样本最大值；分母 RMS，下限 1e-8）：

| 层 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset val (%) | 0.97 | 1.43 | 1.81 | 2.03 | 2.24 | 2.17 | 2.34 | 2.32 | 2.39 | 2.52 | 2.70 | **3.78** |
| real val (%) | 0.81 | 1.03 | 1.30 | 1.47 | 1.66 | 1.80 | 1.81 | 1.85 | 1.97 | 2.08 | 2.31 | **3.13** |

- 误差随层数单调累积（注意力 GELU LUT/量化主导）：第 0 层 ~0.8–1.0%，
  第 11 层最差 **3.78%**。
- **11/12 层 < 3%，第 11 层略超 3%（3.78% / 3.13%）**。这是本独立链的实测结论，
  不是官方 600 条评测结论（见“证据层级”）。

## L2 结果（真实 RTL，12 层不同权重，逐位）

每层用**各自**的 `attn_params.svh` / `kwt_postnorm_cfg.svh` / `M_A` 单独精化；
层间用 RTL 的 `y` 作为下一层 `X`。6 条真实验证样本 × 12 层 = **72 个层运行、
7128 行 RTL 输出**，`a/r1/g/f/r2/y` 六个阶段与 L1 整数参考**全部逐位一致**。

| 样本 | 逐位 | 最差层 y NRMSE |
| --- | --- | ---: |
| sample_00 | exact | 2.5163% |
| sample_01 | exact | 2.7870% |
| sample_02 | exact | 2.7963% |
| sample_03 | exact | 2.4854% |
| sample_04 | exact | **3.1279%** |
| sample_05 | exact | 2.7448% |

`rtl_results.json.bit_exact_all = true`。RTL 与 L1 参考逐位一致，因此 RTL 的
FP32 NRMSE 与 L1 的 real-val 数字完全吻合（最差出现在第 11 层）。

## 证据层级（逐项标注）

| 层级 | 状态 | 证据 |
| --- | --- | --- |
| L1 软件参考（12 层独立整数链） | **已证** | `model/run_l1.py`、`results.json`、`logs/run_l1.log` |
| L2 RTL 逐位（12 层不同权重） | **已证** | `model/run_rtl_chain.py`、`rtl_results.json`、`logs/run_rtl_chain.log` |
| L3 E203 SoC（指令流/NICE/总线） | **未证** | 未在官方 E203 上运行；`graduate/encoder/tb/tb_layers_smoke.sv` 只证调度 |
| L4 综合（150 MHz/面积/功耗） | **未证** | 无目标 PDK/SDC；未做 Yosys/DC |
| L5 官方验收（600 条评测） | **未证** | 未跑官方评测脚本；无官方评测集 |

## 复现

```bash
bash graduate/ds_tasks/encoder_12numeric/run_all.sh
```

或分段：

```bash
OPENBLAS_NUM_THREADS=4 python3 graduate/ds_tasks/encoder_12numeric/model/run_l1.py
OPENBLAS_NUM_THREADS=4 python3 graduate/ds_tasks/encoder_12numeric/model/freeze_int.py
python3 graduate/ds_tasks/encoder_12numeric/model/run_rtl_chain.py --samples all
```

真实退出码与样本数见 `logs/exit_codes.txt`。
