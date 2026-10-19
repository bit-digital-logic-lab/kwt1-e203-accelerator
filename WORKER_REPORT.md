# WORKER_REPORT — encoder_gate（研究生 5–6 项：NRMSE 3% 门禁修复 + 负例注入）

- 分支 / worktree：`codex/kwt-ds-grad-encoder_gate` @ `/home/yang/kwt-worktrees/ds-grad-encoder_gate`
- 基线：`5136ca1`（grad-encoder 组装终态）
- 允许改动范围：`graduate/encoder/model/verify_encoder.py` 与 `graduate/ds_tasks/encoder_gate/`
  （外加本根报告）。未改 RTL、未改 `gen_vectors.py`/`kwt_postnorm_ref.py`/`run.sh`、
  未动 `hardware/`、`software/`、`docs/`；未开子智能体；未 `merge`/`rebase`/`push`。
- 上一版 grad-encoder 组装报告见 git 提交 `5136ca1`（本文件按任务要求改为三段式）。

## 1. 结果

**缺陷**：`verify_encoder.py` 第 3 项检查算了 `worst` 并打印，却从未用它判定；
只有“逐位一致 (`ok`)”会失败。于是阶段逐位一致、但最差阶段 NRMSE 超 3% 时，
脚本仍打印 `VERIFY_PASS` 且 `exit 0` —— 3% 门禁是空门禁。

**修复**（仅 `verify_encoder.py`，+63/−7）：

- 新增规格常数 `NRMSE_THRESHOLD = 0.03` 与谓词 `nrmse_ok(worst)`（`<= 3%` 过，**严格超 3% 失败**）。
- 最差阶段超阈值 → 打印 `GATE ... nrmse=FAIL`、`VERIFY_FAIL reason=nrmse`，`exit 1`。
- exit 0 当且仅当**逐位全一致 且 最差阶段 NRMSE ≤ 3%**。
- 负例注入 `--negative-inject {none,bitexact,nrmse}`：`bitexact` 污染 `y[0,0]`；
  `nrmse` 只乘反量化路径（整数阵列不动），使两条门禁可分别被证伪。
- `--nrmse-threshold` 可覆盖阈值，用于证明门禁随实测值动作。

**证据层级表**（本任务只到 E0/E1）：

| 层级 | 内容 | 状态 | 证据 |
| --- | --- | --- | --- |
| **E0 软件参考** | NumPy 独立整数模型 + 研究生 FP32 golden | ✅ 已用 | `verify_encoder.py` oracle / NRMSE 参考 |
| **E1 RTL 逐位** | Verilator 5.030 功能仿真，6 阶段 × 99 行逐位一致 | ✅ 已跑 | `logs/01_run_all.log`、`logs/02_verify_positive.log` |
| **E2 E203** | 真实 E203 指令流 / System Bus / NICE / SRAM 宏 | ❌ 未覆盖 | 仅 E203-like 事务模型（`SOFTWARE_BOUNDARY.md`） |
| **E3 综合** | 逻辑综合 / 150 MHz STA / 面积 / 功耗 | ❌ 未覆盖 | 本任务未跑 |
| **E4 官方验收** | 官方 600 条评测集 NRMSE / 准确率 | ❌ 未覆盖 | 无官方评测集/脚本 |

**实测数值**（真实样本，`sample_00`，第 0 层，99 行）：

- 正向：`GATE bit_exact=PASS nrmse=PASS rows=99 stages=6 worst=1.3987% threshold=3.00%` → `VERIFY_PASS`。
- 逐位：6 阶段（`a,r1,g,f,r2,y`）× 99 行 = **57,024** 个整数项全部一致。
- NRMSE 最差阶段 `g`(GELU) = **1.3987%** < 3%；`y`(编码器输出) = 0.7289%。
- 负例 `bitexact`：`reason=bit_exact`，`exit 1`。
- 负例 `nrmse`：`y` NRMSE **9.9819%**，`reason=nrmse`，`exit 1`。
- 阈值灵敏度：干净数据 + 0.10% 阈值 → `reason=nrmse`，`exit 1`（证明门禁随实测值动作）。
- 门禁谓词边界：12 项软件单测通过（3.00% 过、3.01% 失败）。

**划分策略**：训练划分 `sample_01/03/04` 只做无梯度标定；验证划分
`sample_00/02/05` 仅评估。`NRMSE_THRESHOLD` 为常数 3%，**不由任何样本拟合**。

## 2. 复现命令与退出码

在仓库根 `/home/yang/kwt-worktrees/ds-grad-encoder_gate` 执行（2026-10-19，Verilator 5.030，Python 3.10 + NumPy 1.21.5）：

| 命令 | 退出码 | 结果 / 日志 |
| --- | ---: | --- |
| `bash graduate/encoder/tb/run.sh` | **0** | `TB_PASS` + `VERIFY_PASS`；6 阶段 99 行逐位一致（`logs/01_run_all.log`） |
| `python3 graduate/encoder/model/verify_encoder.py` | **0** | `GATE ... nrmse=PASS worst=1.3987%`（`logs/02_verify_positive.log`） |
| `python3 .../verify_encoder.py --nrmse-threshold 0.001` | **1** | `VERIFY_FAIL reason=nrmse`（`logs/02b_clean_tight_threshold.log`） |
| `python3 .../verify_encoder.py --negative-inject bitexact` | **1** | `VERIFY_FAIL reason=bit_exact`（`logs/03_neg_bitexact.log`） |
| `python3 .../verify_encoder.py --negative-inject nrmse` | **1** | `VERIFY_FAIL reason=nrmse`，`y` 9.9819%（`logs/04_neg_nrmse.log`） |
| `python3 graduate/ds_tasks/encoder_gate/gate_selfcheck.py` | **0** | `SELFCHECK_PASS checks=12`（`logs/05_gate_selfcheck.log`） |
| `python3 graduate/ds_tasks/encoder_gate/split_calibration.py` | **0** | `SPLIT_OK n_train=3 n_val=3 threshold_source=constant`（`logs/06_split_calibration.log`） |
| `bash graduate/ds_tasks/encoder_gate/run_gate.sh` | **0** | 7 用例全过：`PASS=7 FAIL=0` / `GATE_DRIVER_PASS`；汇总 `summary.json` |
| `bash graduate/encoder/tb/run_smoke.sh` | **0** | `SMOKE_PASS`（12 层调度；未改 RTL，回归确认） |

样本数：正向 99 行 × 6 阶段 = 57,024 项逐位；标定 3 个训练样本；评估 3 个验证样本。
墙上时间：`run.sh` ≈ 22 s，`run_gate.sh` ≈ 46 s，`run_smoke.sh` ≈ 259 s。

## 3. 未证事项

1. **未修先前尺度泄漏（超出本任务文件范围）**：`vectors/meta.json` 与 RTL 激励的尺度来自
   `kwt_postnorm_ref.calibrate()`，其硬编码 `SAMPLE = sample_00`（**验证样本**）。训练划分重算显示
   `s_x` 偏差 **+44.6%**、`s_a` **+19.7%**、`s_out` **+4.8%**。该文件本任务无权修改，
   故尺度仍未遵守“训练标定/验证评估”分离；需在后续任务把 `calibrate()` 改为读取训练划分。
   门禁阈值不受影响（是规格常数）。
2. **只到 E1**：本门禁只证“Verilator 功能仿真逐位一致 + 反量化 NRMSE ≤ 3%”，
   不证 E2（真实 E203/总线/SRAM 宏）、E3（综合/150 MHz STA/面积/功耗）、
   E4（官方 600 条评测集 NRMSE/准确率）。
3. **仍是单层单样本**：只评估第 0 层、验证样本 `sample_00`；12 层数值正确性、
   其余验证样本（`sample_02/05`）的端到端 NRMSE 未测。负例注入为**构造故障**，
   证明门禁会响，不代表真实设计必然触发。
4. **FP32 参考为 NumPy golden**：非本次从官方 TF 重跑（grad-golden 已与官方 TF ≤9.3e-6 吻合）。
5. **ACCEPTANCE.json 的 `verify.log` 哈希将失效**：正向日志新增 `GATE` 行，
   旧 `ACCEPTANCE.json` 记录的 `graduate/encoder/tb/build/verify.log` sha256 不再匹配；
   `ACCEPTANCE.json` 不在本任务允许改动范围，未更新。
