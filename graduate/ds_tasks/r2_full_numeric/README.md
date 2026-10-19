# r2_full_numeric — 用第一轮冻结资产做 12 层整链多样本数值检验

- 分支 / worktree：`codex/kwt-ds-grad-r2_full_numeric` @ `/home/yang/kwt-worktrees/ds-grad-r2_full_numeric`
- 基线：`codex/kwt-grad-encoder @ 7b6abee`
- 范围：只写 `graduate/ds_tasks/r2_full_numeric/**` 与根 `WORKER_REPORT.md`；其余 worktree / 主工程
  `/home/yang/Desktop/ic设计竞赛` 只读。只读使用了主工程 gitignored 数据
  `software/.build/quant_samples_12labels_20.npz` 与第一轮 worktree 的冻结资产；未提交数据数组。
- 术语：**第一轮冻结资产** = `ds-grad-*` 第一轮 worktree 里已冻结、不再重算的标定/整数包，本任务
  直接消费它们，**不做任何再标定**（训练划分可无梯度标定，本任务连标定都没有做；验证划分只评估）。

## 1. 结果

### 1.1 结论

**第一轮冻结资产不能原样拼成“一套”可执行的 12 层 RTL 配置；但在唯一覆盖 12 层的冻结包上，
可以独立跑完 246 条验证样本的 12 层整链数值检验。** 两件事分别给出可复现证据：

| 编号 | 判定 | 实跑退出码 |
| --- | --- | ---: |
| A. 层 0 输入尺度冲突（`attn_scale` × `postnorm_scale`） | **冲突**，最早阻塞点 = `layer0.x_scale_contract` / `layer0.r1` | `2` |
| B. 组装 RTL 常数头 × 第一轮冻结包（层 0） | **冲突**，16 个比较常数中 **13 个不一致** | `2` |
| C. 12 层整链多样本数值（仅用覆盖 12 层的冻结包） | **池化 NRMSE 全 12 层 < 3%**；保守逐样本尾部第 11 层 **3.7839% > 3%** | `0` |

### 1.2 A：层 0 输入尺度冲突（软件见证）

第一轮 `attn_scale` 与 `postnorm_scale` 对**同一个 encoder 输入张量 X** 冻结了两个不同尺度：

| 资产 | 标定样本 | `s_x` | 相对差 |
| --- | ---: | ---: | ---: |
| `ds-grad-attn_scale/attn_calib_train.json` | 训练 3 条 | `1.961475e-3` | — |
| `ds-grad-postnorm_scale/frozen_postnorm_calib.json` | 训练 240 条 | `6.737881e-3` | **70.89%** |

组装 RTL 把按 `scale_x` 量化的整数 `X_int` 直接送进 PostNorm 残差，而 `MA1/MB1` 由
`postnorm.s_x` 推出，二者必须相等。见证（3 条真实验证样本，注意力用第一轮 `attn_scale` 常数、
残差用第一轮 `postnorm_scale` 常数；未做任何再标定）：

| 样本 | 朴素合并层 0 `r1` NRMSE | 残差饱和 | 因果对照：把 X 重量化到 postnorm 尺度 |
| --- | ---: | ---: | ---: |
| `sample_00_encoder.npy` | **237.63%** | 0/6336 | 0.4716% |
| `sample_02_encoder.npy` | **251.30%** | 1/6336 | 0.3365% |
| `sample_05_encoder.npy` | **237.29%** | 2/6336 | 0.3112% |

对照列表明差异来自输入尺度契约，而不是注意力输出本身。第二个静态不变量也不成立：
`attn_scale × 2047 = 4.015` 覆盖不了训练集 `max|X| = 13.792`（57/240 条训练样本、58011 个元素会越界）。
故**实际运行到最早阻塞点 `layer0.r1` 即停止**，不再往后跑层 1–11。证据：`conflict.json`、
`logs/run_conflict_witness.console.log`。

### 1.3 B：组装 RTL 常数头 × 冻结包冲突（静态常数比较）

`graduate/rtl/encoder/` 单层 TB 仍带**旧的逐样本标定**常数头，与第一轮训练冻结包（层 0）不一致：

| 常数 | RTL 头 / TB | 冻结包层 0 | 常数 | RTL 头 | 冻结包层 0 |
| --- | ---: | ---: | --- | ---: | ---: |
| `KC` | 6680 | 5152 | `M_A` | 5981 | 5247 |
| `SH_KC` | 38 | 34 | `MA1` | 61302 | 65960 |
| `RQ` | 6 | 5 | `MB1` | 12260 | 15037 |
| `RV` | 7 | 6 | `EPS_Q1` | 31173 | 1462 |
| `RC` | 14 | 15 | `MGN` | 267644 | 129003 |
| `MC2` | 77111 | 62159 | `EPS_Q2` | 15272 | 9924 |
| `MO2` | 546 | 467 | （`RK/RO/MO1` 一致） | | |

因此组装 RTL **不能作为**该冻结数值链的可执行形态，除非按冻结包重生成常数头
（第一轮 `encoder_12numeric` 正是为每层单独重生成才跑通 12 层 RTL）。证据：`rtl_config.json`、
`logs/check_rtl_config.console.log`。本项是**静态比较**，不宣称仿真。

### 1.4 C：12 层整链多样本数值（冻结包本身自洽）

唯一覆盖 12 层的冻结包是 `encoder_12numeric/frozen_12layer_calib.json` +
`frozen_12layer_int.npz`（12 组不同权重/尺度，训练划分标定，验证未参与）。本任务
**直接加载该冻结包**，不动任何尺度/移位/LUT，独立实现驱动在 **246 条验证样本**
（240 条数据集验证 + 6 条真实 WAV 编码器输入）上跑 12 层整数链，对 FP32 golden 计 NRMSE。

口径：`NRMSE_layer = sqrt(Σ(int_dequant − fp32_ref)² / Σ(fp32_ref)²)`，分母下限 `1e-8`，
**逐层汇总全部 246 条样本**（题目“分层汇总全部评测样本”的本地类比，**不是**官方 600 条评测器）。

| 层 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 池化 y (%) | 0.79 | 1.06 | 1.30 | 1.44 | 1.56 | 1.69 | 1.78 | 1.86 | 1.95 | 2.05 | 2.24 | **2.75** |
| 逐样本最大 y (%) | 0.97 | 1.43 | 1.81 | 2.03 | 2.24 | 2.17 | 2.34 | 2.32 | 2.39 | 2.52 | 2.70 | **3.78** |

- **池化口径：12/12 层 < 3%**，最差第 11 层 2.7527% → `all_layers_below_limit=true`（`state=PASS_POOLED_TAIL_ABOVE_LIMIT`）。
- **保守逐样本最大口径：第 11 层 3.7839%（数据集）/ 3.1279%（真实）> 3%**，与第一轮
  `encoder_12numeric/results.json` 的 3.78%/3.13% **逐位吻合**，说明本任务是同一冻结包、不同统计口径，
  不是另一套结果。
- 饱和计数（246 样本 × 12 层累计）：`r1: 16`、`r2: 16`、`y: 5`（总元素约 7.3e7），输出有限。
- 阶段池化 NRMSE 最差在第 11 层 `r2` 2.7260%、`y` 2.7527%；`a` 阶段池化最大出现在第 9 层 2.3317%。
- 运行时 42.7 s（单进程）。证据：`results.json`、`logs/run_full_numeric.console.log`。

### 1.5 证据层级（逐层标注）

| 层级 | 状态 | 证据 / 说明 |
| --- | --- | --- |
| L1 软件参考 | **已证（本任务）** | `run_full_numeric.py`，246 验证样本 × 12 层，冻结包零再标定；`results.json` |
| L2 RTL 逐位 | **本任务未复跑** | 第一轮 `encoder_12numeric` 自报 6×12=72/72 层 `a/r1/g/f/r2/y` 与整数参考逐位一致；本任务**没有**独立复跑该 L2，且没有用本任务产物宣称 L2 |
| L3 E203 SoC（指令流/总线/NICE） | **未证** | 未在官方 E203 上运行；无 SoC/总线驱动证据 |
| L4 综合（150 MHz/面积/功耗） | **未证** | 无目标 PDK / SRAM 宏 / SDC，未做 STA/面积/功耗 |
| L5 官方验收（600 条评测集） | **未证** | 官方评测集/脚本不可得；本地 246 条验证样本 ≠ 官方 600 条 |

## 2. 复现命令与退出码

仓库根执行（一键）：

```bash
bash graduate/ds_tasks/r2_full_numeric/repro.sh    # REPRO_EXIT=0
```

分段（真实退出码见 `logs/exit_codes.txt`）：

| # | 命令 | 退出码 | 结果 |
| ---: | --- | ---: | --- |
| 1 | `python3 graduate/ds_tasks/r2_full_numeric/model/run_conflict_witness.py` | **2** | `CONFIG_CONFLICT`；最早阻塞 `layer0.x_scale_contract`/`r1`；3 样本朴素 `r1` 237.63/251.30/237.29% |
| 2 | `python3 graduate/ds_tasks/r2_full_numeric/model/check_rtl_config.py` | **2** | `RTL_CONFIG_CONFLICT`；16 比较中 13 失配 |
| 3 | `OPENBLAS_NUM_THREADS=4 python3 graduate/ds_tasks/r2_full_numeric/model/run_full_numeric.py` | **0** | 246 样本 × 12 层；池化 12/12 < 3%（最差 2.75%）；逐样本尾部 3.78% |

关键资产 SHA-256（只读引用，不复制大数据）：

| 资产 | SHA-256 |
| --- | --- |
| `frozen_12layer_calib.json` | `5f039ec745e2b4dda1e6de5abb6fdf97316c3e98a0265c31212b53eee669d444` |
| `frozen_12layer_int.npz` | `5cf836e0e5108d9a20d7d817ef24423d9cd9f196dd353872f0de5e82da317e5b` |
| `attn_calib_train.json` | `9a535f04b99f0cebce941e88fc0ef8d6fcc9e30f29656d32433dff8d07fa267f` |
| `frozen_postnorm_calib.json` | `3a1cf51b2f00716b0dc6a182d80e6d4cdf5886fa196affad268a47167bbf53fb` |
| `quant_samples_12labels_20.npz`（主工程 gitignored，只读） | `ee04e849bc407149618e4ecd5ff99575687586d4e57c0ade6a0b3f7b66c8c149` |
| `kwt1_fp32.npz` | `8f359abf7adbe287b8f89fca98af46194f98f90f2146784731bff551a155825a` |

## 3. 未证事项（禁止夸大）

1. **没有一套可执行的第一轮整链 RTL 配置**：A 与 B 两个冲突已证；把候选配置写回 RTL 常数头并重跑
   12 层 RTL 属集成步骤，**不在本任务范围**。本任务不宣称“12 层整链 E203/SoC 通过”。
2. **L2 RTL 逐位未由本任务独立复跑**：第一轮 72/72 逐位是**来源报告的自报**，本任务只读引用，未复现。
3. **第 11 层保守尾部超限**：逐样本最大 3.7839%（数据集）/ 3.1279%（真实）> 3%；池化口径 2.7527% < 3%。
   两个口径都保留；**不宣称“全部样本 < 3%”**。
4. **仅本地 246 条验证样本**：不是官方隐藏 600 条评测集，未核对官方评测脚本的张量展平/RMS 口径；
   `PASS_POOLED_TAIL_ABOVE_LIMIT` 不是官方精度结论。
5. **无综合/时序/面积/功耗**：无目标 PDK、SRAM 宏、SDC；无 150 MHz STA。
6. **无官方 SoC**：未跑 E203 指令流 / System Bus / NICE；组装 RTL 本身也未在本任务运行。
7. **未再标定**：训练划分无梯度标定本可做，但本任务的目的是检验第一轮冻结资产，故未做任何重标定；
   若允许重算，冲突 A 可按共享训练划分唯一化（见 `ds-grad-r2_scale_contract`），但那是新配置而非第一轮冻结资产。
