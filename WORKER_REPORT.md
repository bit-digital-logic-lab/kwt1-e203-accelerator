# WORKER_REPORT — codex/kwt-grad-encoder（研究生 5–6 项：单层 PostNorm 编码器组装 + 12 层调度）

- 分支 / worktree：`codex/kwt-grad-encoder` @ `/home/yang/kwt-worktrees/grad-encoder`
- 基线：`main@3d9ae98`（grad-attention `405c876` / grad-nonlinear `10aa202` / grad-golden `c9a4307` 均为终态）
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
