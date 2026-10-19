# WORKER_REPORT — r2_full_numeric（用第一轮冻结资产做 12 层整链多样本数值检验）

- 分支 / worktree：`codex/kwt-ds-grad-r2_full_numeric` @ `/home/yang/kwt-worktrees/ds-grad-r2_full_numeric`
- 基线：`codex/kwt-grad-encoder @ 7b6abee`；唯一产物目录 `graduate/ds_tasks/r2_full_numeric/`
- 边界：只在本 worktree 操作；未开子智能体；未改其他 worktree；未 `merge`/`rebase`/`push`；
  只读使用主工程 `/home/yang/Desktop/ic设计竞赛` 的 gitignored 数据与第一轮 worktree；未提交数据数组/密钥/缓存。
- 详细结果、表格与资产哈希见 `graduate/ds_tasks/r2_full_numeric/README.md`。

## 1. 结果

**第一轮冻结资产不能原样拼成一套可执行的 12 层 RTL 配置；但覆盖 12 层的冻结整数包本身自洽，
可独立跑完 246 条验证样本的 12 层整链数值检验。** 三项实跑：

1. **层 0 输入尺度冲突（A，冲突）**：第一轮 `attn_scale.scale_x=1.961475e-3`（训练 3 条）
   与 `postnorm_scale.s_x=6.737881e-3`（训练 240 条）相差 **70.89%**，而残差系数由 postnorm 尺度推出。
   最早阻塞点 = `layer0.x_scale_contract`；见证运行到 `layer0.r1` 即停：`sample_00/02/05` 朴素合并
   `r1` NRMSE **237.63% / 251.30% / 237.29%**（因果对照 0.31–0.47%）。另有静态不变量不成立：
   `attn_scale×2047=4.015 < max_train|X|=13.792`。
2. **组装 RTL 常数头冲突（B，冲突）**：`graduate/rtl/encoder/` 常数头/TB 与冻结包层 0 比较，
   16 个常数 **13 个失配**（`KC` 6680 vs 5152、`SH_KC` 38 vs 34、`M_A` 5981 vs 5247、`MA1` 61302 vs 65960、
   `EPS_Q1` 31173 vs 1462、`MGN` 267644 vs 129003 等）。组装 RTL 不是该冻结数值链的可执行形态。
3. **12 层整链多样本数值（C，运行）**：直接加载第一轮 12 层冻结包（零再标定），在 **246 条验证样本**
   （240 数据集 + 6 真实）上跑 12 层整数链对 FP32 golden。池化 NRMSE **12/12 层 < 3%**，最差第 11 层
   **2.7527%**；保守逐样本最大口径第 11 层 **3.7839%（数据集）/ 3.1279%（真实）> 3%**，与第一轮
   `encoder_12numeric` 的 3.78%/3.13% 逐位吻合。饱和计数 `r1:16 r2:16 y:5`，输出全有限。
   `state=PASS_POOLED_TAIL_ABOVE_LIMIT`。

**证据层级**：L1 软件参考 = **本任务已证**（246 样本 × 12 层，冻结包零再标定）；
L2 RTL 逐位 = **本任务未复跑**（第一轮自报 72/72 逐位仅只读引用）；L3 E203/SoC = **未证**；
L4 综合/时序/面积/功耗 = **未证**；L5 官方 600 条验收 = **未证**。

## 2. 复现命令与退出码

仓库根执行（2026-10-19，真实退出码见 `graduate/ds_tasks/r2_full_numeric/logs/exit_codes.txt`）：

| # | 命令 | 退出码 | 结果 |
| ---: | --- | ---: | --- |
| 1 | `python3 graduate/ds_tasks/r2_full_numeric/model/run_conflict_witness.py` | **2** | `CONFIG_CONFLICT`；最早阻塞 `layer0.x_scale_contract`→`layer0.r1`；3 样本 `r1` 237.63/251.30/237.29% |
| 2 | `python3 graduate/ds_tasks/r2_full_numeric/model/check_rtl_config.py` | **2** | `RTL_CONFIG_CONFLICT`；16 比较中 13 失配 |
| 3 | `OPENBLAS_NUM_THREADS=4 python3 graduate/ds_tasks/r2_full_numeric/model/run_full_numeric.py` | **0** | 246 样本 × 12 层；池化 12/12 < 3%（最差 2.75%）；逐样本尾部 3.78% |
| — | `bash graduate/ds_tasks/r2_full_numeric/repro.sh` | **0** | 串联上面三项；`REPRO_OK conflict_exit=2 rtl_exit=2 numeric_exit=0` |

冻结资产（只读引用）：`frozen_12layer_calib.json` sha256 `5f039ec7…d444`、
`frozen_12layer_int.npz` sha256 `5cf836e0…7e5b`；主工程 gitignored
`quant_samples_12labels_20.npz` sha256 `ee04e849…c149`。

## 3. 未证事项

1. **没有一套可执行的第一轮整链 RTL 配置**：冲突 A/B 已证；把配置写回常数头并重跑 12 层 RTL 属集成步骤，不在本任务范围。
2. **L2 RTL 逐位未由本任务独立复跑**：第一轮 72/72 逐位为来源报告自报，本任务只读引用。
3. **第 11 层保守尾部超限**：逐样本最大 3.7839%/3.1279% > 3%；池化口径 2.7527% < 3%；不宣称“全部样本 < 3%”。
4. **仅本地 246 条验证样本**，非官方隐藏 600 条评测集；未核对官方评测脚本展平/RMS 口径；不是官方精度结论。
5. **无综合/时序/面积/功耗签核**：无目标 PDK、SRAM 宏、SDC，未做 150 MHz STA。
6. **无官方 SoC**：未跑 E203 指令流 / System Bus / NICE；组装 RTL 也未在本任务运行。
7. **未再标定**：本任务刻意不重算尺度/移位/LUT；冲突 A 若要消除需按共享训练划分唯一化，那会得到新配置而非第一轮冻结资产。
