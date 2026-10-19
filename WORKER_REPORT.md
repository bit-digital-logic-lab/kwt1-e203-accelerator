# WORKER_REPORT — codex/kwt-grad-encoder（研究生 5–6 项：单层 PostNorm 编码器组装 + 12 层调度）

- 分支 / worktree：`codex/kwt-grad-encoder` @ `/home/yang/kwt-worktrees/grad-encoder`
- 基线：`main@3d9ae98`（grad-attention `405c876` / grad-nonlinear `3084222` / grad-golden `c9a4307`；
  grad-nonlinear 拼接期间由自身 worker 从 `10aa202` 追加 ANSI 端口表修复提交，见 `PROVENANCE.md`）
- 范围：把三项成果**组装**为单层 PostNorm 编码器（注意力 + 两处残差 + 两处 LayerNorm +
  `64→256→64` 前馈 + GELU），并实现 **12 层参数更新/搬运控制**；冻结 E203 软件边界与分类头
  软件边界。对应 [`题目/640.jpg 9`](题目/640.jpg%209) 第 5、6 项。
- 边界：只在本 worktree 内操作；未开子智能体；未改主目录/他人分支；未
  `merge`/`rebase`/`cherry-pick`/`push`；未改 `hardware/`、`software/` 既有本科核心。
- 交付契约：根目录 `ACCEPTANCE.json`（编排者 `git archive HEAD` 干净副本独立复跑）。

详细边界与来源见 `graduate/encoder/README.md`、`SOFTWARE_BOUNDARY.md`、`PROVENANCE.md`。

## 1. 本次新增/组装的文件

| 文件 | 作用 |
| --- | --- |
| `graduate/rtl/encoder/kwt_encoder_layer.sv` | 单层 PostNorm 编码器数据通路（注意力→requant→残差1→LN1→FFN1→GELU→FFN2→残差2→LN2），统一装载端口 |
| `graduate/rtl/encoder/kwt_ffn1_row.sv` | FFN1 行引擎 `64→256`，INT48 累加，输出 per-channel scale `s_h[c]`（接 postnorm `h`） |
| `graduate/rtl/encoder/kwt_encoder_layers_ctrl.sv` | 12 层参数更新/层间搬运/分类头边界控制器（纯事务序列，无算术） |
| `graduate/rtl/encoder/kwt_encoder_top.sv` | layer + ctrl 顶层；E203-like 事务端口 `o_ld_*`/`p_rdata` |
| `graduate/rtl/encoder/third_party/*` | 只读复制：注意力 RTL（grad-attention）、postnorm FFN/GELU/LN RTL（grad-nonlinear） |
| `graduate/encoder/model/{attn_int_model,kwt_postnorm_ref}.py` | 只读复制的独立整数参考（仅路径适配） |
| `graduate/encoder/reference/kwt1_fp32_golden.py` | 只读复制的 FP32 golden（仅路径适配） |
| `graduate/encoder/model/verify_encoder.py` | 组合独立整数模型 + FP32 误差逐级对照（新增） |
| `graduate/encoder/vectors/gen_vectors.py` | 生成全部 RTL 激励/参数 hex（新增） |
| `graduate/encoder/tb/tb_encoder.sv`、`tb/run.sh` | 单层逐级对照 TB + 一键脚本（新增） |
| `graduate/encoder/tb/tb_layers_smoke.sv`、`tb/run_smoke.sh` | 12 层调度 smoke TB + 一键脚本（新增） |

## 2. 实跑命令与退出码（本次，2026-10-19）

| 命令（仓库根执行） | 退出码 | 结果 |
| --- | ---: | --- |
| `bash graduate/encoder/tb/run.sh` | **0** | `TB_PASS`（`cycles=4,700,560`）+ `VERIFY_PASS`；6 阶段 99 行**全逐位一致** |
| `bash graduate/encoder/tb/run_smoke.sh` | **0** | `SMOKE_PASS`；`n_start=12 max_layer=11 yrows=1188 n_ld_par=595200 head=1` |
| `verilator --lint-only … --top-module kwt_encoder_top …` | **0** | 0 error（仅第三方文件 WIDTH 警告） |

复现细节：`run.sh` 先生成向量（`gen_vectors.py` = 运行复制的 `attn_int_model.py` +
`kwt_postnorm_ref.calibrate()`），再 `verilator --binary` 建 `tb_encoder` 并运行，
最后 `verify_encoder.py` 做逐位 + FP32 校验。`TB_DONE … sat=0/0` 无 X。

## 3. 样本、层

- **样本**：真实验证输入 `software/validation/real_input_chain_20261017/sample_00_encoder.npy`
  （`X[99,64]`，SHA 与 manifest 一致）。
- **层**：`enc_layers/0`（第 0 层），尺度与 FFN1 冻结包（`ffn1_a12w10_static_sym.npz`
  的 `layer01_*`）一致。
- **单层逐级对照**：99 行全量，6 个阶段（`a,r1,g,f,r2,y`）。
- **12 层调度 smoke**：12 层顺序执行，逐层重载 49,600 项参数，共 1,188 行输出。

## 4. 数值误差

单层逐级（RTL vs 组合独立整数模型逐位；RTL 反量化 vs 研究生 FP32 golden）：

| 阶段 | 逐位 | FP32 NRMSE | max_abs |
| --- | --- | ---: | ---: |
| `a`（注意力，含 requant） | 99×64 exact | 0.1591% | 1.32e-3 |
| `r1` | 99×64 exact | 0.1082% | 2.89e-3 |
| `g`（GELU） | 99×256 exact | **1.3987%** | 1.29e-2 |
| `f`（FFN2） | 99×64 exact | 1.3968% | 1.54e-2 |
| `r2` | 99×64 exact | 0.7000% | 5.00e-2 |
| `y`（编码器输出） | 99×64 exact | 0.7289% | 4.23e-2 |

最差阶段 `g` = **1.3987% < 3% 门槛**；`y` = 0.7289%。逐位一致（0 失配）覆盖
`a`（含注意力 + requant）到 `y` 的整条链。

## 5. 未证事项（不得宣称完整研究生组验收）

1. **无官方 SoC**：未在真实 E203 指令流/System Bus/NICE 上运行；`o_ld_*`/`p_rdata` 为
   **E203-like** 事务模型（已在 `SOFTWARE_BOUNDARY.md` 逐条标注）。
2. **仅第 0 层标定**：第 1–11 层尺度/EPS/GELU LUT 未逐层重算；12 层 smoke 只证**调度**，
   不证 12 层数值。
3. 无目标 PDK / SRAM 宏 / SDC：未做 150 MHz STA、面积、功耗签核；仅 Verilator 功能仿真
   （未跑 Icarus 复核）。
4. FFN2/注意力的大数组与组合读端口仍是行为存储，未映射题目规定 SRAM 宏。
5. 分类头为**软件**阶段（题目规定），硬件只给 `head_token`/`head_valid`；未做分类头 RTL。
6. 未测官方 600 条评测集 NRMSE / 准确率；FP32 参考为官方源码语义的 NumPy golden
   （grad-golden 已与官方 TF 原文 ≤9.3e-6 吻合），本目录不再重跑 TF。
7. 本报告不声明“12 层整层 E203 通过”或“研究生 5–6 项全部通过”。


---

# WORKER_REPORT — p027 精度搜索（12 层编码器输出 NRMSE ≤ 0.27%）

- 分支 / worktree：`ds-p027-precision`（本 worktree，未改主工程与其他分支）
- 范围：只新增 `graduate/ds_tasks/p027_precision/**` 与本文件；未修改
  `encoder_12numeric`、`r2_full_numeric`、`graduate/rtl/**`、`hardware/**`、`software/**`
- 基线：A12W10 冻结 12 层整数链，240 条本地 validation 逐层池化最差 **2.7528%**
  （`graduate/ds_tasks/r2_full_numeric/results.json`）。本报告**不**改写该数字。
- 目标：240 条独立 validation、12 层、逐层池化编码器输出 NRMSE ≤ **0.27%**
- 边界：training 划分只做无梯度标定；validation 只评估；本地 split 无 test 划分；
  不声明 RTL / E203 / 综合 / 官方 600 条结论。

## 1. 交付物

| 文件 | 作用 |
| --- | --- |
| `model/prec_ref.py` | 参数化整数参考（激活/权重位宽、softmax/GELU/LN 定点宽度、`QB`、`isq`、`QG`、注意力内部寄存器） |
| `model/run_search.py` | 有界精度搜索；每候选记录逐层池化 NRMSE、饱和、累加器位宽、成本代理、配置 SHA256 |
| `model/selftest.py` | 标量 vs 向量整数原语逐位自检、尺度连续性、累加器上界 |
| `model/freeze_config.py` | 冻结选中方案（全部整数张量 + 系数 + SHA256） |
| `model/run_frozen.py` | 冻结包独立复跑：不标定、不读 training、校验 SHA256、记录整数输出哈希 |
| `model/summarize.py` | 由 `results.json` 生成汇总表 |
| `repro.sh` | 搜索 → 冻结 → 独立复跑，记录真实退出码 |
| `results.json` / `frozen_results.json` | 机器可读结果 |
| `logs/` | 控制台日志、退出码、汇总表、自检日志 |

## 2. 实测结果

| 方案 | 激活位宽 | 权重位宽 | 最差层 | 240 条池化 y NRMSE | train-fit | 权重 KiB/12L | MAC bit-product |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A12W10（基线契约） | 12 | 10 | 11 | 2.7538% | 2.6968% | 745.5 | 10.74e9 |
| A16W10 | 16 | 10 | 11 | 0.3301% | 0.3218% | 745.5 | 13.68e9 |
| **A16W12（选中）** | **16** | **12** | **11** | **0.1517%** | **0.1286%** | **889.5** | **15.55e9** |
| A16W16 | 16 | 16 | 11 | 0.1301% | 0.1043% | 1177.5 | 19.28e9 |
| A20W16 | 20 | 16 | 11 | 0.0977% | 0.0562% | 1177.5 | 24.71e9 |
| A16W12_lut10 | 16 | 12 | 11 | 0.2524% | 0.2377% | 889.5 | 15.55e9 |
| SEL_A20xy_W12 | 20/16 | 12 | 11 | 0.1287% | 0.0981% | 889.5 | 18.17e9 |
| SEL_A20xy_W16 | 20/16 | 16 | 11 | 0.1023% | 0.0640% | 1177.5 | 22.22e9 |
| A16W16_hr105 | 16 | 16 | 11 | 0.1078% | 0.1066% | 1177.5 | 19.28e9 |
| SEL_A20xy_W12_hr105 | 20/16 | 12 | 11 | 0.1047% | 0.0983% | 889.5 | 18.17e9 |

- **结论**：8/10 候选达到 ≤0.27%；最好 A20W16 = 0.0977%，但按“≥25% 余量 + 最小 MAC
  bit-product”规则选中 **A16W12**：240 条 validation 最差层（L11）**0.1517%**，
  train-fit 0.1286%，相对基线 2.7538% 改善 **18.1×**，距目标余量 **1.78×**。
- 选中方案逐层（240 条，池化 y %）：0.0480 / 0.0591 / 0.1039 / 0.1017 / 0.1055 /
  0.1108 / 0.1091 / 0.1115 / 0.1150 / 0.1201 / 0.1300 / **0.1517**。
- 饱和（240×12 层累计）：`a:0, r1:16, g:6, r2:18, y:5`；实测累加器位宽
  `qkv 29 / score 35 / ctx 36 / attn_out 30 / ffn1 30 / ffn2 29`。
- 成本：权重 889.5 KiB/12 层（74.1 KiB/层）；激活 125.3 KiB/层；乘法器
  attn/ffn1/ffn2/QK/PV/requant/LN = 28/28/28/32/36/34/32 bit；实测常数位宽
  ma1 34、mb1 33、ma2 34、md2 21、M_A 17、mg 21、mgn 19；`isq_frac=32`（31-bit 寄存器）；
  exp2 表 256×Q20（4-bit 插值）、GELU LUT 4097×Q12、KC 21 bit。
- **冻结独立复跑**：`frozen_results.json`，`calibration_calls=0`，training 行加载数 0，
  240 条 validation 最差 0.1517%，与搜索测量 `max|delta|=0.0`，整数 `y` 逐层哈希已记录；
  `config_sha256=6b273d895c716dcbc5a9652596a1756fd80d41e9f79dad60c8052b199d3b148f`，
  `calib_sha256=2f7f25b9ec6b2839e49b4a6ef866d0a4b3fd6d7ea0d74e8bf594867b1b8908be`。


## 3. 关键定点问题（本次实测发现）

1. **残差系数 `QB`**：A16 时 `md2 = sh2/s_r2·2^16` 被舍入为 0，FFN2 残差支路丢失，
   `r2` 误差 20.5% → `QB = A+16`。
2. **LayerNorm 倒数 `isq`**：`isq = 2^16/√var` 在 A16 下只剩约 2 bit 有效位 → 
   `isq_frac = 16+2(A−12)+8`，寄存器上限 31 bit。
3. **GELU 输入 rescale `QG`**：固定 28 时 A20 的 `mg` 仅 134（0.7% 乘数噪声）→ `QG = 28+(A−12)`。
4. **GELU LUT `LF`**：LF=8 的输出量化在 A16 下成为主误差 → LF=12（节点间距 1/256）。
5. **softmax 概率位宽**：15-bit 概率使注意力输出误差约 0.14% → `prob_bits=20`。
6. **注意力 requant 双截断**：`a_int = O_int·M_A/2^16` 在 validation 超出训练最大值时
   会在 16-bit 处二次截断（半数据标定的 holdout 曾放大到 2%）→ `a` 寄存器加 2 bit 余量。

## 4. 命令与退出码

| # | 命令（仓库根执行） | 退出码 | 结果 |
| ---: | --- | ---: | --- |
| 1 | `bash graduate/ds_tasks/p027_precision/repro.sh` | **0** | 串联下面 3 段，`REPRO_OK selected=A16W12` |
| 2 | `python3 graduate/ds_tasks/p027_precision/model/run_search.py` | **0** | 10 个候选，548.6 s；`state=TARGET_MET` |
| 3 | `python3 graduate/ds_tasks/p027_precision/model/selftest.py` | **0** | 标量==向量逐位、尺度连续、acc≤64；基线行 2.7538% |
| 4 | `python3 .../freeze_config.py --name A16W12` | **0** | 冻结包 + MANIFEST SHA256 |
| 5 | `python3 .../run_frozen.py --pack .../frozen` | **0** | 独立复跑 0.1517%，`comparison_identical=true` |

日志：`logs/run_search.console.log`、`logs/run_frozen.console.log`、
`logs/freeze_config.console.log`、`logs/selftest.log`、`logs/exit_codes.txt`、
`logs/summary.md`。


## 5. 证据层级

| 层级 | 状态 |
| --- | --- |
| L1 参数化整数参考（训练标定、validation 评估） | 已证（本任务） |
| 冻结包独立复跑（无标定、无 training、SHA256 校验） | 见 `frozen_results.json` |
| L2 RTL 逐位 | 未做（不改 RTL） |
| L3 E203 / L4 综合 / L5 官方 600 条 | 未证 |

## 6. 未证事项

1. 240 条本地 validation 不是官方 600 条评测；未核对官方脚本的展平/RMS 口径。
2. 未跑 RTL/Verilator；整数参考与 RTL 的位精确关系沿用 `encoder_12numeric` 的既有结论，
   本任务未复跑。
3. 未做综合、时序、面积、功耗；`cost` 中的位宽与 MAC bit-product 只是代理指标。
4. 高精度方案需要更宽常数乘法器/倒数寄存器/LUT，面积影响未换算。
5. 搜索的方案阶梯是硬件位宽选择；层 0 诊断与 24 样本筛选阶段使用过 validation 的前几个
   样本（已在 `README.md` 记录），最终数字为全部 240 条并附 train-fit 交叉检查。
