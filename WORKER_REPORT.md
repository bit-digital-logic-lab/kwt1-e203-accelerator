# WORKER_REPORT — r2_full_numeric（用第一轮冻结资产做 12 层整链多样本数值检验）

- 分支 / worktree：`codex/kwt-ds-grad-r2_full_numeric` @ `/home/yang/kwt-worktrees/ds-grad-r2_full_numeric`
- 基线：`codex/kwt-grad-encoder @ 7b6abee`；唯一产物目录 `graduate/ds_tasks/r2_full_numeric/`
- 边界：只在本 worktree 操作；未开子智能体；未改其他 worktree；未 `merge`/`rebase`/`push`；
  只读使用主工程 `/home/yang/Desktop/ic设计竞赛` 的 gitignored 数据与第一轮 worktree；未提交数据数组/密钥/缓存。
- 详细结果、表格与资产哈希见 `graduate/ds_tasks/r2_full_numeric/README.md`。
- 本次为**验收修正**：按真实 `manifest.split` 重做数据选择与结果/报告，保留冻结包、整数运算、
  误差公式与冲突证据，**未对验证集重新标定**。

## 1. 结果

**第一轮冻结资产不能原样拼成一套可执行的 12 层 RTL 配置；但覆盖 12 层的冻结整数包本身自洽，
可在独立 validation 划分的 240 条 npz 样本上跑完 12 层整链数值检验。** 三项实跑：

1. **层 0 输入尺度冲突（A，冲突）**：第一轮 `attn_scale.scale_x=1.961475e-3`（训练 3 条）
   与 `postnorm_scale.s_x=6.737881e-3`（训练 240 条）相差 **70.89%**，而残差系数由 postnorm 尺度推出。
   最早阻塞点 = `layer0.x_scale_contract`；见证运行到 `layer0.r1` 即停：`sample_00/02/05` 朴素合并
   `r1` NRMSE **237.63% / 251.30% / 237.29%**（因果对照 0.31–0.47%）。另有静态不变量不成立：
   `attn_scale×2047=4.015 < max_train|X|=13.792`。
2. **组装 RTL 常数头冲突（B，冲突）**：`graduate/rtl/encoder/` 常数头/TB 与冻结包层 0 比较，
   16 个常数 **13 个失配**（`KC` 6680 vs 5152、`SH_KC` 38 vs 34、`M_A` 5981 vs 5247、`MA1` 61302 vs 65960、
   `EPS_Q1` 31173 vs 1462、`MGN` 267644 vs 129003 等）。组装 RTL 不是该冻结数值链的可执行形态。
3. **12 层整链数值（C，运行）**：直接加载第一轮 12 层冻结包（零再标定），正式指标只在
   **240 条 npz validation 样本**上逐层池化，对 FP32 golden 计 NRMSE。**12/12 层 < 3%**，最差第 11 层
   **2.7528%**。逐样本最大尾部第 11 层 3.7839% 仅作诊断保留（题目门槛是 600 条逐层汇总，不是每条 < 3%），
   不改变退出码。饱和计数 `r1:16 r2:16 y:5`（仅正式 240 条），输出全有限。
   `state=PASS_FORMAL_POOLED_TAIL_ABOVE_LIMIT`。

### 1.1 数据选择修正（按真实 manifest.split）

`software/validation/real_input_chain_20261017/manifest.json` 实际为 **validation 3 条 + training 3 条**，
旧版把 sample_00..05 全称 real_validation 并池化成“246 条验证”是错的。修正后：

| 真实文件 | manifest split | archive member | npz 索引 / partition | 与 240 条重复 | 计入正式指标 |
| --- | --- | --- | --- | --- | --- |
| `sample_00_encoder.npy` | validation | `four/750e3e75_nohash_0.wav` | 240 / validation | 是 | 否 |
| `sample_01_encoder.npy` | training | `go/0ba018fc_nohash_4.wav` | 40 / training | 否 | 否（训练参考） |
| `sample_02_encoder.npy` | validation | `no/45692b02_nohash_3.wav` | 320 / validation | 是 | 否 |
| `sample_03_encoder.npy` | training | `no/7e7ca854_nohash_1.wav` | 80 / training | 否 | 否（训练参考） |
| `sample_04_encoder.npy` | training | `yes/676f8138_nohash_0.wav` | 200 / training | 否 | 否（训练参考） |
| `sample_05_encoder.npy` | validation | `yes/ec21c46b_nohash_1.wav` | 440 / validation | 是 | 否 |

去重证据：npz 路径去掉前导 `./` 后与 manifest `archive_member` 逐条相等，且 npz `partitions`
与 manifest `split` 完全一致（脚本断言，不一致退出 `2`）。3 条真实验证 = 240 条中的 3 条，
`overlap_count=3`、`real_validation_counted_additionally=0`、`unique_validation_total=240`。
另用同一 MFCC→encoder 投影重算，最大绝对差 3.9e-06–2.3e-05，与已记录的 MFCC 重提取误差同量级。
真实 3 条 validation 单独报告（第 11 层池化 2.6879%、逐样本最大 2.7963%）；真实 3 条 training
仅作参考（第 11 层池化 2.8099%、逐样本最大 3.1279%，来自 `sample_04`），**不写成验证样本**。
旧版“真实验证尾部 3.1279%”实为训练划分样本，已纠正。

**证据层级**：L1 软件参考 = **本任务已证**（240 条正式 + 3 条真实验证单独，冻结包零再标定）；
L2 RTL 逐位 = **本任务未复跑**（第一轮自报 72/72 逐位仅只读引用）；L3 E203/SoC = **未证**；
L4 综合/时序/面积/功耗 = **未证**；L5 官方 600 条验收 = **未证**。

## 2. 复现命令与退出码

仓库根执行（2026-10-19，真实退出码见 `graduate/ds_tasks/r2_full_numeric/logs/exit_codes.txt`）：

| # | 命令 | 退出码 | 结果 |
| ---: | --- | ---: | --- |
| 1 | `python3 graduate/ds_tasks/r2_full_numeric/model/run_conflict_witness.py` | **2** | `CONFIG_CONFLICT`；最早阻塞 `layer0.x_scale_contract`→`layer0.r1`；3 样本 `r1` 237.63/251.30/237.29% |
| 2 | `python3 graduate/ds_tasks/r2_full_numeric/model/check_rtl_config.py` | **2** | `RTL_CONFIG_CONFLICT`；16 比较中 13 失配 |
| 3 | `OPENBLAS_NUM_THREADS=4 python3 graduate/ds_tasks/r2_full_numeric/model/run_full_numeric.py` | **0** | 240 条 npz validation 正式池化 12/12 < 3%（最差 2.75%）；3 条真实验证单独；尾部 3.78% 仅诊断 |
| — | `bash graduate/ds_tasks/r2_full_numeric/repro.sh` | **0** | 串联上面三项；`REPRO_OK conflict_exit=2 rtl_exit=2 numeric_exit=0` |

`repro.sh` 语义已修正：阶段 1/2 的 `2` 是**复现冲突的预期码**，不再被当成失败；脚本 `0` 表示
三阶段都符合预期，`1` 表示退出码不符，`3` 表示正式数值门槛失败并原样透传（不伪装成 `0`）。
指标语义也在 `exit_codes.txt` 与 `results.json` 中写明：正式 = 240 条 npz validation 逐层池化，
真实 validation 3 条单独（+0 唯一），真实 training 3 条仅参考，逐样本尾部仅诊断。

冻结资产（只读引用）：`frozen_12layer_calib.json` sha256 `5f039ec7…d444`、
`frozen_12layer_int.npz` sha256 `5cf836e0…7e5b`；主工程 gitignored
`quant_samples_12labels_20.npz` sha256 `ee04e849…c149`；
`real_input_chain_20261017/manifest.json` sha256 `d6f33945…268f`。

## 3. 未证事项

1. **没有一套可执行的第一轮整链 RTL 配置**：冲突 A/B 已证；把配置写回常数头并重跑 12 层 RTL 属集成步骤，不在本任务范围。
2. **L2 RTL 逐位未由本任务独立复跑**：第一轮 72/72 逐位为来源报告自报，本任务只读引用。
3. **第 11 层保守尾部超限**：240 条逐样本最大 3.7839% > 3%；正式池化 2.7528% < 3%；
   不宣称“全部样本 < 3%”，也不把尾部诊断当门槛。
4. **仅本地 240 条独立 validation 样本**，非官方隐藏 600 条评测集；未核对官方评测脚本展平/RMS 口径；不是官方精度结论。
5. **真实 6 条不是 6 条验证**：3 条 validation（与 npz 重复，+0 条）+ 3 条 training（仅参考）；训练样本不计入验证。
6. **无综合/时序/面积/功耗签核**：无目标 PDK、SRAM 宏、SDC，未做 150 MHz STA。
7. **无官方 SoC**：未跑 E203 指令流 / System Bus / NICE；组装 RTL 也未在本任务运行。
8. **未再标定**：本任务刻意不重算尺度/移位/LUT；冲突 A 若要消除需按共享训练划分唯一化，那会得到新配置而非第一轮冻结资产。
