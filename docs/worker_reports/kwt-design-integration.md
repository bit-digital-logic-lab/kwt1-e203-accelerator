# WORKER_REPORT — design-integration（本科组整体设计决策与证据核对）

- 分支：`codex/kwt-design-integration`（worktree `/home/yang/kwt-worktrees/design-integration`）
- 任务：围绕 KWT-1 本科必做 1-3 项，独立核对源代码与真实产物（不照抄报告），产出整体设计决策与分阶段整合方案；新增并实跑证据校验脚本。未改核心 RTL/量化代码，未调用子智能体，未 push/merge/rebase，只写 `docs/`、本报告。

## 改动

新增：

- `docs/KWT1_整体设计决策_20261018.md`：① PC/E203/DMA/PE/软件数值模型的职责、格式、地址与所有权、接口假设；② 保守可交付 vs 性能优化两路径（布局、SRAM、带宽、周期公式、量化格式、INT32 边界、需改的 RTL/驱动）并区分已测/模型推算/正式平台未知；③ Claude 审查的**正确/过期/未证**逐项裁定；④ 按验收门槛排序的剩余工作、可复现命令、缺失资源与停止条件；⑤ 6 个分支的分阶段整合顺序、冲突点、验收门槛（本次不实际 merge）。
- `docs/check_design_evidence.py`：只读证据核对脚本，`--results-dir` 指定主目录结果路径，不修改共享数据。
- 本文件 `WORKER_REPORT.md`。

未改动：任何 RTL、量化、驱动源码；主目录、其他 worktree、官方资料只读。

## 实际执行的命令与退出结果

| 命令 | 退出码 | 结果 |
| --- | ---: | --- |
| `python3 docs/check_design_evidence.py --results-dir "/home/yang/Desktop/ic设计竞赛/hardware/architecture/results"` | **0** | `SUMMARY: PASS=72 WARN=0 FAIL=0`；含 12 层 exit/PASS/SHA、摘要计数、向量格式边界、INT32 边界、关键文件存在性，并对照旧 0/12 失败运行 |
| `python3 - <<PY`（独立核算 12 层日志 SHA-256 与 summary.json） | 0 | 12/12 逐字一致（见下） |
| `python3 - <<PY`（读 manifest 字段/形状/边界/int32） | 0 | 形状 X[99,64]/W[64,256]/bias[256]/Y[99,256]；最大保守界 67,780,536 |

## 关键实测（本次独立核验，非转述）

- 修正运行 `real_e203_validation0_ascii_*`：`finalizer.status`=`SUMMARIZE_EXIT 0`；12 个 `layerNN.exit` 全 `0`；每层恰好一条全层 `PASS`（`start=0 count=208 outputs=25344 padding=1280 task_cycles=838028 total_cycles=838041`）；无 `FATAL/mismatch/error/xxxx`。
- 12 份日志 SHA-256 与 `*_ascii_summary.json` 的 `log_sha256` **逐字一致**；摘要合计 `checked_outputs=304128`、`checked_padding=15360`、`total_task_cycles=10056336`（=12×838028）。
- 向量 manifest：hex 行数精确为 a16=6656、w16=16384、bias32=256、expected32=25344，且 SHA-256 全部匹配；A12W10 保守上界 `max=67,780,536`（INT32 的 3.16%），实测最大 `|expected INT32|`=4,593,325。
- 旧运行 `real_e203_validation0_*`：12 个 `.exit` 全 `1`、日志含 `got=xxxxxxxx`、`SUMMARIZE_EXIT 1` —— 属**过期**状态（中文路径 `$readmemh`），不是当前状态。

## 结论要点

- 当前通过范围：**一条真实验证样本的 A12W10 冻结向量在 E203 + 仿真 NICE-like 系统 + 8×16 老引擎**上，12 层各 25,344（合计 304,128）输出逐位一致。Claude 把旧 0/12 当成当前失败属**过期**。
- `154,240`（16 行，L1 测试台）与 `838,028`（8×16，L2 E203）**不可相减当加速比**；`1.63×` 是 8 行/16 行同为 L1 的模型比值。
- A14W12/A12W12/A16W10 **仅软件证据**（无 RTL 逐位、无综合）；A14W12 保守上界已占 INT32 的 50.59%。
- 样本数不得混用：480 软件、240 RTL、1 条 E203、官方 600（未测）为四个不同统计。

## 未证事项

- 正式组委会 SoC/System Bus/NICE 规范、PDK、SRAM 宏、SDC 全部缺失；无 150 MHz、≤2.25 mm²、功耗/能耗的正式签核。
- C 驱动未在 E203 仿真中执行（仅宿主测试 + RV32 交叉编译）。
- 官方 600 条评测集未发布，官方 NRMSE/分类准确率未测。
- 本次未重跑 E203 仿真与 240 样本 RTL（`docs/check_design_evidence.py` 只核验既有真实产物与哈希）；E203 通过路径是 8×16 老引擎，不是 16 行核。
- 本次未实际执行任何 merge；§5 的整合顺序与验收门槛为方案，须由人工在受控分支执行并复核。
