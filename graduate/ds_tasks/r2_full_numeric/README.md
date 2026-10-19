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
可以在独立 validation 划分的 240 条 npz 样本上跑完 12 层整链数值检验。** 两件事分别给出可复现证据：

| 编号 | 判定 | 实跑退出码 |
| --- | --- | ---: |
| A. 层 0 输入尺度冲突（`attn_scale` × `postnorm_scale`） | **冲突**，最早阻塞点 = `layer0.x_scale_contract` / `layer0.r1` | `2` |
| B. 组装 RTL 常数头 × 第一轮冻结包（层 0） | **冲突**，16 个比较常数中 **13 个不一致** | `2` |
| C. 12 层整链数值（仅用覆盖 12 层的冻结包） | **正式口径：240 条 npz validation 逐层池化，12/12 层 < 3%**；逐样本尾部第 11 层 **3.7839%**（仅诊断） | `0` |

**数据选择按真实 manifest.split 修正（本次修正的核心）**：`real_input_chain_20261017/manifest.json`
的 6 条真实 WAV 是 **validation 3 条 + training 3 条**，不是 6 条验证；并且这 3 条 validation
与 npz 的 240 条 validation 是**同一批 archive member**（去重后 +0 条）。因此：

- **正式本地指标只在独立 validation 的 240 条 npz 样本上逐层汇总**；
- 3 条真实 validation 样本（`sample_00/02/05`）**单独报告**，不并入 240 条计数（否则会把同样本算两遍）；
- 3 条真实 training 样本（`sample_01/03/04`）只作**非验证参考**，**不得写成验证样本**；
- 不再存在“246 条验证”这一口径。

### 1.2 A：层 0 输入尺度冲突（软件见证）

第一轮 `attn_scale` 与 `postnorm_scale` 对**同一个 encoder 输入张量 X** 冻结了两个不同尺度：

| 资产 | 标定样本 | `s_x` | 相对差 |
| --- | ---: | ---: | ---: |
| `ds-grad-attn_scale/attn_calib_train.json` | 训练 3 条 | `1.961475e-3` | — |
| `ds-grad-postnorm_scale/frozen_postnorm_calib.json` | 训练 240 条 | `6.737881e-3` | **70.89%** |

组装 RTL 把按 `scale_x` 量化的整数 `X_int` 直接送进 PostNorm 残差，而 `MA1/MB1` 由
`postnorm.s_x` 推出，二者必须相等。见证（3 条真实验证样本，注意力用第一轮 `attn_scale` 常数、
残差用第一轮 `postnorm_scale` 常数；未做任何再标定）：

| 样本 | manifest split | 朴素合并层 0 `r1` NRMSE | 残差饱和 | 因果对照：把 X 重量化到 postnorm 尺度 |
| --- | --- | ---: | ---: | ---: |
| `sample_00_encoder.npy` | validation | **237.63%** | 0/6336 | 0.4716% |
| `sample_02_encoder.npy` | validation | **251.30%** | 1/6336 | 0.3365% |
| `sample_05_encoder.npy` | validation | **237.29%** | 2/6336 | 0.3112% |

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

### 1.4 C：12 层整链数值（冻结包本身自洽）

唯一覆盖 12 层的冻结包是 `encoder_12numeric/frozen_12layer_calib.json` +
`frozen_12layer_int.npz`（12 组不同权重/尺度，训练划分标定，验证未参与）。本任务
**直接加载该冻结包**，不动任何尺度/移位/LUT，独立实现驱动在 **240 条 npz validation 样本**
上跑 12 层整数链，对 FP32 golden 计 NRMSE。

口径：`NRMSE_layer = sqrt(Σ(int_dequant − fp32_ref)² / Σ(fp32_ref)²)`，分母下限 `1e-8`，
**逐层汇总全部 240 条独立 validation 样本**。这是题目“分层汇总全部评测样本”的本地类比
（官方评测集为 600 条，本地不是官方评测器）。

| 层 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 正式：240 条逐层池化 y (%) | 0.79 | 1.06 | 1.30 | 1.44 | 1.56 | 1.69 | 1.79 | 1.86 | 1.96 | 2.05 | 2.24 | **2.75** |
| 诊断：240 条逐样本最大 y (%) | 0.97 | 1.43 | 1.81 | 2.03 | 2.24 | 2.17 | 2.34 | 2.32 | 2.39 | 2.52 | 2.70 | **3.78** |

- **正式口径：12/12 层 < 3%**，最差第 11 层 2.7528% → `all_layers_below_limit=true`
  （`state=PASS_FORMAL_POOLED_TAIL_ABOVE_LIMIT`，退出码 `0`）。
- **尾部诊断保留但不再是门槛**：240 条中逐样本最大第 11 层 3.7839% > 3%。题目门槛是
  “**600 条逐层汇总**”的池化 NRMSE，**不是每条样本都必须 < 3%**；因此该尾部超限不改变正式判定，
  本报告也不宣称“全部样本 < 3%”。
- 饱和计数（仅正式 240 条 × 12 层累计，`r1`/`r2`/`y` 各约 1.82e7 个元素）：`r1: 16`、`r2: 16`、`y: 5`，输出有限。
- 阶段池化 NRMSE 最差在第 11 层 `r2` 2.7260%、`y` 2.7528%；`a` 阶段池化最大出现在第 9 层 2.3329%。
- 运行时约 30–52 s（本机单进程，含 240 + 3 + 3 条；随机器负载波动）。证据：`results.json`、
  `logs/run_full_numeric.console.log`。

#### 1.4.1 分区与去重证据（路径级）

npz 路径形如 `./<label>/<file>.wav`，manifest 用 `<label>/<file>.wav`；去掉前导 `./` 后逐条比对
`archive_member`，并核对 npz `partitions` 字段与 manifest `split` 完全一致（脚本内断言，不一致即退出 `2`）：

| 真实文件 | manifest split | archive member | npz 索引 | npz partition | 与 240 条重复 | 计入正式指标 |
| --- | --- | --- | ---: | --- | --- | --- |
| `sample_00_encoder.npy` | validation | `four/750e3e75_nohash_0.wav` | 240 | validation | 是 | 否（已在 240 内） |
| `sample_01_encoder.npy` | training | `go/0ba018fc_nohash_4.wav` | 40 | training | 否 | 否（训练，仅参考） |
| `sample_02_encoder.npy` | validation | `no/45692b02_nohash_3.wav` | 320 | validation | 是 | 否（已在 240 内） |
| `sample_03_encoder.npy` | training | `no/7e7ca854_nohash_1.wav` | 80 | training | 否 | 否（训练，仅参考） |
| `sample_04_encoder.npy` | training | `yes/676f8138_nohash_0.wav` | 200 | training | 否 | 否（训练，仅参考） |
| `sample_05_encoder.npy` | validation | `yes/ec21c46b_nohash_1.wav` | 440 | validation | 是 | 否（已在 240 内） |

去重结论：`overlap_count=3`、`real_validation_counted_additionally=0`、
`unique_validation_total=240`、`combined_count_used=240`。此外用本任务同一 MFCC→encoder 投影
重算这 6 条的 encoder 输入，与文件最大绝对差 3.9e-06 ~ 2.3e-05（与 README 记录的 MFCC 重提取误差同量级），
说明“同一 archive member = 同一验证样本”，不是另一批数据。

真实 3 条 validation 单独结果（不并入 240）：

| 层 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 条池化 y (%) | 0.74 | 0.98 | 1.20 | 1.38 | 1.53 | 1.65 | 1.70 | 1.78 | 1.84 | 1.95 | 2.20 | 2.69 |
| 3 条逐样本最大 y (%) | 0.77 | 1.03 | 1.30 | 1.47 | 1.66 | 1.80 | 1.81 | 1.85 | 1.88 | 1.98 | 2.23 | 2.80 |

真实 3 条 training 仅作参考（**不是验证样本**）：池化最差第 11 层 2.8099%，逐样本最大第 11 层
3.1279%（来自 `sample_04`）。旧版把这一 3.1279% 写成“真实验证尾部”，实际来自训练划分样本，
本次已纠正；真实 validation 的尾部是 2.7963%。

### 1.5 证据层级（逐层标注）

| 层级 | 状态 | 证据 / 说明 |
| --- | --- | --- |
| L1 软件参考 | **已证（本任务）** | `run_full_numeric.py`，240 条 npz validation × 12 层正式指标 + 3 条真实验证单独报告，冻结包零再标定；`results.json` |
| L2 RTL 逐位 | **本任务未复跑** | 第一轮 `encoder_12numeric` 自报 6×12=72/72 层 `a/r1/g/f/r2/y` 与整数参考逐位一致；本任务**没有**独立复跑该 L2，且没有用本任务产物宣称 L2 |
| L3 E203 SoC（指令流/总线/NICE） | **未证** | 未在官方 E203 上运行；无 SoC/总线驱动证据 |
| L4 综合（150 MHz/面积/功耗） | **未证** | 无目标 PDK / SRAM 宏 / SDC，未做 STA/面积/功耗 |
| L5 官方验收（600 条评测集） | **未证** | 官方评测集/脚本不可得；本地 240 条独立 validation ≠ 官方 600 条 |

## 2. 复现命令与退出码

仓库根执行（一键）：

```bash
bash graduate/ds_tasks/r2_full_numeric/repro.sh    # REPRO_EXIT=0
```

`repro.sh` 的退出码语义：`0` = 三个阶段都返回**各自预期**的退出码（阶段 1/2 预期为 `2`，表示
**复现了冲突**，不是工具失败；阶段 3 预期为 `0`）；`1` = 任一阶段退出码不符预期；`3` = 正式数值门槛
本身失败（原样透传，不会被伪装成 `0`）。阶段 1/2 的 `2` 不会被当成脚本失败。

分段（真实退出码见 `logs/exit_codes.txt`）：

| # | 命令 | 退出码 | 结果 |
| ---: | --- | ---: | --- |
| 1 | `python3 graduate/ds_tasks/r2_full_numeric/model/run_conflict_witness.py` | **2** | `CONFIG_CONFLICT`；最早阻塞 `layer0.x_scale_contract`/`r1`；3 样本朴素 `r1` 237.63/251.30/237.29% |
| 2 | `python3 graduate/ds_tasks/r2_full_numeric/model/check_rtl_config.py` | **2** | `RTL_CONFIG_CONFLICT`；16 比较中 13 失配 |
| 3 | `OPENBLAS_NUM_THREADS=4 python3 graduate/ds_tasks/r2_full_numeric/model/run_full_numeric.py` | **0** | 240 条 npz validation 正式池化 12/12 < 3%（最差 2.75%）；3 条真实验证单独；尾部 3.78% 仅诊断 |
| — | `bash graduate/ds_tasks/r2_full_numeric/repro.sh` | **0** | 串联上面三项；`REPRO_OK conflict_exit=2 rtl_exit=2 numeric_exit=0` |

关键资产 SHA-256（只读引用，不复制大数据）：

| 资产 | SHA-256 |
| --- | --- |
| `frozen_12layer_calib.json` | `5f039ec745e2b4dda1e6de5abb6fdf97316c3e98a0265c31212b53eee669d444` |
| `frozen_12layer_int.npz` | `5cf836e0e5108d9a20d7d817ef24423d9cd9f196dd353872f0de5e82da317e5b` |
| `attn_calib_train.json` | `9a535f04b99f0cebce941e88fc0ef8d6fcc9e30f29656d32433dff8d07fa267f` |
| `frozen_postnorm_calib.json` | `3a1cf51b2f00716b0dc6a182d80e6d4cdf5886fa196affad268a47167bbf53fb` |
| `quant_samples_12labels_20.npz`（主工程 gitignored，只读） | `ee04e849bc407149618e4ecd5ff99575687586d4e57c0ade6a0b3f7b66c8c149` |
| `real_input_chain_20261017/manifest.json` | `d6f33945d3f416e48da5135ba73acfd2db90dbe5f630770b0babccb2186e268f` |
| `kwt1_fp32.npz` | `8f359abf7adbe287b8f89fca98af46194f98f90f2146784731bff551a155825a` |

## 3. 未证事项（禁止夸大）

1. **没有一套可执行的第一轮整链 RTL 配置**：A 与 B 两个冲突已证；把候选配置写回 RTL 常数头并重跑
   12 层 RTL 属集成步骤，**不在本任务范围**。本任务不宣称“12 层整链 E203/SoC 通过”。
2. **L2 RTL 逐位未由本任务独立复跑**：第一轮 72/72 逐位是**来源报告的自报**，本任务只读引用，未复现。
3. **第 11 层保守尾部超限**：240 条逐样本最大 3.7839% > 3%；正式池化口径 2.7528% < 3%。
   两个口径都保留；**不宣称“全部样本 < 3%”**，也不把尾部诊断当作门槛。
4. **仅本地 240 条独立 validation 样本**：不是官方隐藏 600 条评测集，未核对官方评测脚本的张量展平/RMS 口径；
   `PASS_FORMAL_POOLED_TAIL_ABOVE_LIMIT` 不是官方精度结论。
5. **真实 6 条不是 6 条验证**：3 条 validation（与 npz 重复，+0 条）+ 3 条 training（仅参考）；
   训练样本不得计入验证。
6. **无综合/时序/面积/功耗**：无目标 PDK、SRAM 宏、SDC；无 150 MHz STA。
7. **无官方 SoC**：未跑 E203 指令流 / System Bus / NICE；组装 RTL 本身也未在本任务运行。
8. **未再标定**：训练划分无梯度标定本可做，但本任务的目的是检验第一轮冻结资产，故未做任何重标定；
   若允许重算，冲突 A 可按共享训练划分唯一化（见 `ds-grad-r2_scale_contract`），但那是新配置而非第一轮冻结资产。
