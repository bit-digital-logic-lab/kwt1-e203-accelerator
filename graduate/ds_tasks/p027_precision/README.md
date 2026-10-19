# p027_precision — 有界位宽/定点精度搜索（12 层编码器输出 NRMSE）

本目录是**独立的精度实验**，只新增文件，不修改 `graduate/ds_tasks/encoder_12numeric`、
`r2_full_numeric` 或任何 RTL。基线 A12W10 冻结链的 2.7528% 保持原样，本任务不重贴标签。

## 1. 目标与边界

- 目标：**240 条独立 validation 样本、12 层、逐层池化**的编码器输出（`y`）最差层
  NRMSE ≤ **0.27%**，参考为 checkpoint 一致的 FP32 golden。
- 标定：只用 training 划分（240 条 MFCC → 冻结 ckpt 投影后的 encoder 输入），
  无梯度、逐张量静态对称尺度；validation 只做评估；本地 split **没有 test 划分**（加载时断言）。
- 只量化 RCL/硬件阶段；PC 端 MFCC 与 `patch_proj/pos_emb/class_emb` 投影保持 FP32。
- 本任务**不**声明 RTL、E203、综合或官方 600 条评测结论。

## 2. 实验参考实现

`model/prec_ref.py` 是与 RTL 文档契约一致的**参数化整数参考**（新文件，不是基线改动）：

- 激活位宽逐阶段可配：`x/a/r1/ln1/g/r2/y`（`y` 与下一层 `x` 同位宽，保证
  `s_x[k+1] == s_out[k]` 的整数直传）；
- 权重位宽：注意力 `wq/wk/wv/wo`、FFN1、FFN2 分别可配；
- 舍入/饱和：`mul_sr(a,m,q)=(a*m+2^(q-1))>>q`，激活对称饱和，全部为整数运算；
- LayerNorm：整数均值/方差、整数 isqrt、倒数 `isq` 定点；
- GELU：分段线性 LUT（节点 Q(LF)、线性插值、输入 rescale Q(QG)）；
- softmax：exp2 LUT（表项 Q(M_LUT)、可选线性插值）、概率 `P` 位宽、`G_SH` 归一化；
- 残差：`ma1/mb1/ma2/md2` 定点系数（`QB` 小数位）；
- 注意力内部寄存器位宽 `qkv_bits/ctx_bits/out_bits`。

### 2.1 搜索中发现的真实定点瓶颈（不是调参玄学）

把 A 从 12 提到 16/20 时，**只加激活位宽不够**；以下常数位宽必须同步放大，否则误差不降反升：

| 瓶颈 | 现象（实测） | 修正 |
| --- | --- | --- |
| 残差系数小数位 `QB` | A16 时 `md2 = sh2/s_r2·2^16` 被舍入为 **0**，FFN2 残差支路整体丢失 → `r2` 误差 20.5% | `QB = A+16`（A16→32，A20→36），系数位宽实测报告 |
| LayerNorm 倒数 `isq` | `isq = 2^16/√var` 在 A16 时只剩 ~2 bit 有效位 → LN 输出 ~0.8% 噪声 | `isq_frac = 16 + 2(A−12) + 8`，寄存器上限 31 bit |
| GELU 输入 rescale `QG` | 固定 QG=28 时 A20 的 `mg` 只剩 134（0.7% 乘数噪声） | `QG = 28 + (A−12)` |
| GELU LUT 输出 `LF` | LF=8 时 GELU 输出量化 1/256，A16 下成为主误差（1.2%） | LF=12，节点间距 1/256（LUT 4097×Q12） |
| softmax 概率 `P` | 15-bit 概率量化使注意力输出误差 ~0.14% | `prob_bits=20`（exp2 表 256 项 + 4-bit 插值，M_LUT=20） |
| 指数系数 `KC` | 高精度下 `KC` 被压到 320，指数分辨率不足 | 常数乘法器加宽到 ≥2^20（KC 位宽报告） |

上表每一项都由 layer-0 单层对照（`a/r1/g/f/r2/y`）定位，并在**训练划分**上重标定；
validation 只用于最终报告。

## 3. 结果（240 条 validation，逐层池化）

见 `results.json` 与 `logs/summary.md`（自动生成）。下面表格在运行后填入。

全部候选在 240 条 validation 上逐层池化（`y` 为编码器输出）：

| scheme | A bits (x/a/r1/ln1/g/r2) | W attn/w1/w2 | worst layer | pooled y NRMSE | holdout worst | weight KiB/12L | MAC bit-products |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A12W10 | 12/12/12/12/12/12 | 10/10/10 | 11 | 2.7538% | 2.6968% | 745.5 | 10.7e9 |
| A16W10 | 16/16/16/16/16/16 | 10/10/10 | 11 | 0.3301% | 0.3218% | 745.5 | 13.7e9 |
| A16W12 | 16/16/16/16/16/16 | 12/12/12 | 11 | 0.1517% | 0.1286% | 889.5 | 15.5e9 |
| A16W16 | 16/16/16/16/16/16 | 16/16/16 | 11 | 0.1301% | 0.1043% | 1177.5 | 19.3e9 |
| A20W16 | 20/20/20/20/20/20 | 16/16/16 | 11 | 0.0977% | 0.0562% | 1177.5 | 24.7e9 |
| A16W12_lut10 | 16/16/16/16/16/16 | 12/12/12 | 11 | 0.2524% | 0.2377% | 889.5 | 15.5e9 |
| SEL_A20xy_W12 | 20/16/20/16/16/20 | 12/12/12 | 11 | 0.1287% | 0.0981% | 889.5 | 18.2e9 |
| SEL_A20xy_W16 | 20/16/20/16/16/20 | 16/16/16 | 11 | 0.1023% | 0.0640% | 1177.5 | 22.2e9 |
| A16W16_hr105 | 16/16/16/16/16/16 | 16/16/16 | 11 | 0.1078% | 0.1066% | 1177.5 | 19.3e9 |
| SEL_A20xy_W12_hr105 | 20/16/20/16/16/20 | 12/12/12 | 11 | 0.1047% | 0.0983% | 889.5 | 18.2e9 |

说明：

- `A12W10` 是本任务独立重实现的基线契约，240 条实测 **2.7538%**，与冻结包
  `r2_full_numeric` 的 **2.7528%** 相差 0.001 个百分点（独立实现，isq 饱和上限不同）。
- 目标线是 **0.27%**；`A16W10`（16-bit 激活 + 10-bit 权重）为 **0.3301%**，是最接近
  但未达线的方案，说明**只加激活位宽不够**。
- 选中方案 **A16W12**：240 条 validation 最差层 L11 **0.1517%**，train-fit 诊断
  **0.1286%**；相对基线改善 **18.1×**，相对目标余量 **1.78×**。
- `A16W12_lut10` 用 10-bit GELU/softmax LUT 达到 0.2524%，余量仅 7%，按“≥25% 余量”
  规则不选；`A20W16` 达到 0.0977%，但 MAC bit-product 高 59%。
- 所有候选最差层都是第 11 层，误差随层数单调累积，与基线一致。

选中方案逐层（240 条 validation，池化 `y` NRMSE %）：

| 层 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A16W12 | 0.0480 | 0.0591 | 0.1039 | 0.1017 | 0.1055 | 0.1108 | 0.1091 | 0.1115 | 0.1150 | 0.1201 | 0.1300 | **0.1517** |
| A12W10（基线） | 0.7885 | 1.0628 | 1.3032 | 1.4392 | 1.5655 | 1.6943 | 1.7884 | 1.8600 | 1.9586 | 2.0521 | 2.2376 | **2.7538** |

成本代理（12 层）：

| 方案 | 权重 KiB | 激活 KiB/层 | 乘法器位宽（attn/ffn1/ffn2/QK/PV/requant/LN） | 实测累加器位宽（qkv/score/ctx/attn_out/ffn1/ffn2） | MAC bit-product |
| --- | ---: | ---: | --- | --- | ---: |
| A12W10 | 745.5 | 95.9 | 22/22/22/32/31/32/28 | 23/35/31/28/24/23 | 10.74e9 |
| **A16W12** | **889.5** | **125.3** | **28/28/28/32/36/34/32** | **29/35/36/30/30/29** | **15.55e9** |
| A16W16 | 1177.5 | 125.3 | 32/32/32/32/36/34/32 | 33/35/36/34/34/33 | 19.28e9 |
| A20W16 | 1177.5 | 156.2 | 36/36/36/40/40/38/36 | 37/43/40/38/38/37 | 24.71e9 |

A16W12 的常数宽度（实测最大）：残差系数 `QB=32`、LN 倒数 `isq_frac=32`（31-bit 寄存器）、
GELU `QG=32`、exp2 表 256×Q20（4-bit 线性插值）、GELU LUT 4097×Q12、指数系数 `KC` 21 bit；
实测系数位宽 ma1 34、mb1 33、ma2 34、md2 21、M_A 17、mg 21、mgn 19。

### 3.1 冻结独立复跑（无标定）

`frozen/` 内含全部整数张量与系数，`MANIFEST.json` 记录 SHA256。`model/run_frozen.py`
只读冻结包 + checkpoint（FP32 参考）+ 240 条 validation，**不调用标定、不加载 training 行**：

| 项 | 值 |
| --- | --- |
| calibration_calls | 0 |
| training_rows_loaded | 0 |
| 240 条 pooled 最差层 | L11 **0.1517%** |
| 与搜索测量差 | `max|delta| = 0.0`（`comparison_identical=true`） |
| config sha256 | `6b273d895c716dcbc5a9652596a1756fd80d41e9f79dad60c8052b199d3b148f` |
| calib pack sha256 | `2f7f25b9ec6b2839e49b4a6ef866d0a4b3fd6d7ea0d74e8bf594867b1b8908be` |
| 整数 `y` 哈希 | 每层一个 SHA256，见 `frozen_results.json` |

开发期诚实说明：位宽阶梯（A12/16/20 × W10/12/16 + 定点升级）由 layer-0 单层对照与
24 条 validation 子集筛选确定；最终所有数字来自全部 240 条，并附 train-fit（前 96 条
training 样本、同一训练标定）作为与 validation 无关的交叉检查。validation 未参与任何
尺度/移位/LUT/系数拟合。


## 4. 成本代理口径

- `weight_bytes`：每层注意力 4×64×64、FFN1 64×256、FFN2 256×64 权重 + bias + LN 参数，
  按各自位宽计；12 层合计。
- `activation_bytes`：每层存储的 `x/a/r1/out1/g/r2/y`（g 为 99×256），按阶段位宽计。
- `multiplier_bits`：激活×权重、QK、PV、requant、LN 乘数位宽。
- `accumulator_bits`：解析界 + **实测最大累加值所需位宽**（`measured_accumulator_bits`）。
- `mac_bit_products`：Σ MAC 数 × 两个操作数位宽，作为面积/能耗的粗代理（非综合结果）。
- LUT：exp2 表项数/小数位、GELU 节点数/小数位、isq 小数位。

## 5. 饱和与范围检查

每个候选记录 `saturation`（`a/r1/g/r2/y` 触界元素数，240×12 层累计）与
`measured_accumulator_bits`（QKV、score、ctx、attn_out、ffn1、ffn2）。
触界主要来自 validation 样本超过 training 最大值；`headroom=1.05` 候选用于验证
“加 5% 训练最大值余量”可消除残差饱和且不损失精度。

## 6. 复现

```bash
bash graduate/ds_tasks/p027_precision/repro.sh
```

分段：

```bash
OPENBLAS_NUM_THREADS=4 python3 graduate/ds_tasks/p027_precision/model/run_search.py
OPENBLAS_NUM_THREADS=4 python3 graduate/ds_tasks/p027_precision/model/selftest.py
python3 graduate/ds_tasks/p027_precision/model/freeze_config.py --name <selected>
python3 graduate/ds_tasks/p027_precision/model/run_frozen.py --pack graduate/ds_tasks/p027_precision/frozen
python3 graduate/ds_tasks/p027_precision/model/summarize.py
```

真实退出码（本次）：`repro.sh` = 0；`run_search.py` = 0；`freeze_config.py` = 0；
`run_frozen.py` = 0；`selftest.py` = 0。`logs/exit_codes.txt` 记录前 3 段。

## 7. 证据层级

| 层级 | 状态 |
| --- | --- |
| L1 软件整数参考（参数化、训练标定、validation 评估） | 本任务已证 |
| 冻结包独立复跑（无标定、SHA256 校验、整数输出逐位哈希） | 见 `frozen_results.json` |
| L2 RTL 逐位 | 未做（本任务不改 RTL） |
| L3 E203 / L4 综合 / L5 官方 600 条 | 未证 |

## 8. 未证事项

1. 不是官方 600 条评测；本地 240 条 validation 只是同口径类比。
2. 未改 RTL，未做 Verilator 逐位对照；本报告的整数参考与基线 RTL 的位精确关系
   继承自 `encoder_12numeric`（本任务未复跑）。
3. 未做综合/时序/面积/功耗；位宽与 MAC bit-product 只是成本代理。
4. 高精度方案需要更宽的常数乘法器、更宽 LN 倒数寄存器与更大的 LUT；这些在
   `measured_coefficient_bits` / `cost` 中如实报告，未做面积换算。
