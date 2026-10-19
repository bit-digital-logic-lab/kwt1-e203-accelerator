# WORKER_REPORT — r2_scale_contract（交叉核查 attn_scale × postnorm_scale 能否合成一套冻结配置）

- 分支 / worktree：`codex/kwt-ds-grad-r2_scale_contract` @ `/home/yang/kwt-worktrees/ds-grad-r2_scale_contract`
- 范围：**只写** `graduate/ds_tasks/r2_scale_contract/` 与根 `WORKER_REPORT.md`；第一轮
  worktree `ds-grad-attn_scale`、`ds-grad-postnorm_scale` **只读**。
- 边界：只在本 worktree 操作；未开子智能体；未改其他 worktree；未 `merge`/`rebase`/`push`；
  只读使用主工程 `/home/yang/Desktop/ic设计竞赛` 的 gitignored 数据；未提交大数据/密钥/缓存；
  Verilator 构建目录固定在 `/tmp/opencode/r2_scale_contract/`。

## 1. 结果

**判定：两项第一轮成果不能原样合成一套冻结配置；在共享训练划分上可以合成，并已给出候选配置与验证证据。**

- **尺度冲突（核心）**：attn 与 postnorm 对**同一个 encoder 输入张量 X** 冻结了两个不同尺度——
  `attn.scale_x = 1.961475e-3`（3 条训练样本，max|X|=4.015）vs
  `postnorm.s_x = 6.737881e-3`（240 条训练样本，max|X|=13.792），相差 **3.435×**。
  组装 RTL 把注意力按 `scale_x` 量化的 X 直接送入 PostNorm 残差（`kwt_encoder_layer` `x_buf <- attn rd_sel 0`），
  而残差系数 `MA1` 由 `postnorm.s_x` 推出，**两者必须相等**。朴素合并的残余 `r1` NRMSE：
  `sample_00/02/05 = 247.19% / 243.70% / 245.76%`，**远超 3% 门槛（INCOMPATIBLE）**。
- **训练样本**：两批属**同一官方 split**——attn 训练 3 条 wav 全部落在 `quant_samples_12labels_20.npz`
  的 training（240 条），attn 验证 3 条落在 verification（240 条）；npz train/val 互斥。
  attn 只是**3 条子集标定**。跨源 X 一致（npz 投影 vs `real_input_chain` npy，max|Δ|=5.8e-6）；
  两文件权重 sha256 同为 `8f359abf…`（同一 checkpoint）。
- **量化边界不一致**：attn 标称 A12 但 `quantize_activation` 只夹到 int16 容器 ±32767（`|Xi|>2047` 合法）；
  postnorm `CLAMP_A=2047` 硬夹；RTL `o→a` requant 饱和在 int16 ±32768。把 attn 3 样本尺度套到 240 条训练集，
  **57/240** 条样本出现 `|Xi|>2047`（58011 个元素）→ 与 postnorm 边界不兼容。
- **RTL 常数不一致**：`third_party/attn_params.svh` 的 `KC=6680/SH_KC=38` 落后于冻结模型 `6986/37`；
  `third_party/kwt_postnorm_cfg.svh` 8 个模型常数**仅 `MO1` 1 项**与训练冻结模型一致（L1 `[5]` 段逐项打印，
  如 `MA1` 61302 vs 65960、`MGN` 267644 vs 129003）；
  `tb_encoder.sv` 的 `M_A=5981` 仍是旧逐样本标定值。另有命名陷阱：`gen_vectors.py` 读
  `cal["scales"]["sx"]`，该键实为 **FFN1 包激活尺度**（≈2.437e-3），不是 encoder 输入尺度。
- **可行的一套冻结配置（候选，已跑通）**：在共享训练划分（240 条）上唯一化输入尺度
  `s_x = max_train|X|/2047 = 6.737881e-3`（== postnorm 的 `s_x`；attn 3 条 max 4.015 为其子集），
  无梯度重拟合注意力常数（`RQ=5 RK=7 RV=6 RC=15 RO=11 KC=5152 SH_KC=34`,
  `scale_o=1.229728e-4`），postnorm 尺度不变，`M_A=round(scale_o/s_a*2^16)=5247`。
  该候选在**验证划分（仅评估，未用于任何标定）**：

  | 指标 | 值 |
  | --- | ---: |
  | 硬件忠实链（含 o→a requant）最差 NRMSE | **1.6742% < 3%** |
  | 软件参考链最差 NRMSE | 1.6898% |
  | 平均分级 NRMSE r1/g/r2/y | 0.3648% / 1.1723% / 0.8694% / 0.7828% |
  | 验证样本数 | 240（npz validation）+ 3（real `sample_00/02/05`） |
  | 验证 X 量化越界/夹断（±2047） | 0 / 0（max\|X_int\|=2047，恰在边界） |
  | 注意力 RTL 逐位 | 3 条 × **41481/41481**，NRMSE 0.1923%/0.2497%/0.2171% |

  候选产物：`candidate/attn_params.svh`、`candidate/coherent_frozen_calib.json`（**仅提案**）。
- **证据层级**：L1 软件参考 = **已证**（`run_cross_check.py`，7/7 gate PASS）；L2 注意力 RTL 逐位
  = **已证**（`rtl_check_coherent.sh`，Verilator）；L2 整链/12 层 RTL 逐位、E203、综合/时序/面积、
  官方 600 条验收 = **未证**（见 §3）。

## 2. 复现命令与退出码（本 worktree 根目录，2026-10-19 实跑）

| 命令 | 退出码 | 结果 / 样本数 |
| --- | ---: | --- |
| `bash graduate/ds_tasks/r2_scale_contract/repro.sh` | **0** | `REPRO_EXIT=0`（L1+L2 串联） |
| `python3 graduate/ds_tasks/r2_scale_contract/run_cross_check.py` | **0** | `SCALE_CONTRACT_PASS verdict=COMPOSABLE_ON_SHARED_TRAINING_SPLIT`；gate 7/7 PASS；训练 240（+3 attn 子集）、验证 240+3；`results.json`、`logs/cross_check*.log` |
| `bash graduate/ds_tasks/r2_scale_contract/rtl_check_coherent.sh` | **0** | `RTL_CHECK_EXIT=0`；3 条验证各 `41481/41481`，`RESULT: PASS`；`logs/rtl_coh_*.log` |

只读引用（只读，未修改）：`ds-grad-attn_scale/…/attn_calib_train.json`（sha `9a535f04…`）、
`ds-grad-postnorm_scale/…/frozen_postnorm_calib.json`（sha `3a1cf51b…`）、两者各自的 `WORKER_REPORT.md`
与 `results.json`。关键产物 sha256：`run_cross_check.py=e7537371…`、
`gen_coherent_vectors.py=a1c715d8…`、`rtl_check_coherent.sh=259e20ca…`、`repro.sh=3e54fe10…`、
`candidate/attn_params.svh=aaa340af…`、`candidate/coherent_frozen_calib.json=e08bf3f0…`。
（`results.json` 内含 wall-clock 运行时长，其哈希随运行变化，不作为校验基准。）

收口说明：本轮把 `[5]` 段从“只打印 attn 的 `KC/SH_KC` 与 `tb M_A`”补齐为**逐项打印
postnorm 8 个常数**（header vs 训练冻结模型，结果 `1/8` 仅 `MO1` 一致）以及
`gen_vectors.py` 的 `cal["scales"]["sx"]` 命名陷阱（该键 `0.00243741778` = FFN1 包
`layer01_activation_scale`，≠ encoder 输入尺度 `0.00673788148`）。改后 `repro.sh` 重跑仍
`REPRO_EXIT=0`，且 §1 全部数值与首次完整运行逐位一致（`logs/cross_check_console.log` 已含
`[1]`–`[gate]` 全段，不再截断在 `[7]` 的候选生成处）。

## 3. 未证事项

1. **整链/12 层 RTL 逐位未恢复**：本任务只证注意力级 RTL 逐位；postnorm 常数头
   `third_party/kwt_postnorm_cfg.svh` 与整链 TB `tb_encoder.sv`（含 `M_A`）**未按候选重生**，
   12 层 RTL 逐位未重跑。候选配置写入 `third_party/` 并重跑整链 TB 属集成步骤，不在本任务范围。
2. **候选仅为提案**：`candidate/` 未并入 RTL/第三方常数头；注意力常数一旦训练划分变化须重拟合。
3. **无官方 SoC**：未在真实 E203 指令流 / System Bus / NICE 上运行，全部为软件模型 + Verilator
   功能仿真；E203 集成未证。
4. **无综合/时序/面积/功耗签核**：无目标 PDK、SRAM 宏、SDC，未做 150 MHz STA。
5. **无官方验收**：未跑官方 600 条评测集的 NRMSE / 准确率；本任务验证集为 240+3 条自建划分。
6. **仅第 0 层**：第 1–11 层的尺度/EPS/LUT 未逐层重算。
7. **边界余量**：候选尺度下验证集 max\|X_int\|=2047、恰在 A12 边界（无夹断无饱和，但零余量）；
   若评测分布超出训练 max 会被夹断。
8. 不宣称“整链 E203 通过”“研究生 4–6 项全部通过”或“官方精度达标”。
