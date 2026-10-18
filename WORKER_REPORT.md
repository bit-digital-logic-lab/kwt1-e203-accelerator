# WORKER_REPORT — next-analysis（下一阶段设计审查）

- 分支：`codex/kwt-next-analysis`（worktree `/home/yang/kwt-worktrees/next-analysis`，基线 `d94fb6f`）
- 任务：围绕本科必做 1–3 项做**下一轮工程拆解**而非综述；独立核对关键实现；产出设计审查、并行工作包、Git 整合顺序。
- 边界：只修改本 worktree；未开子智能体；未改主目录/其他 worktree/官方资料；未 push/merge/rebase；未改任何 RTL/量化/驱动源码。

## 改动（仅新增，本 worktree）

- `docs/KWT1_下一阶段设计审查.md`：① 最短关键路径（功能/数值/驱动/总线/综合/交付门槛）；② 4 处值得质疑的具体点（Q1–Q4，带文件/行/证据）；③ 5 个工作包（文件所有权、输入、产物、验收命令、最长运行、停止条件、证明边界）；④ 专项评估（C 驱动真执行缺口 / 多样本防爆炸 / 16 行核前后 / iSTA abort / A14W12）；⑤ Git 整合顺序（含 `WORKER_REPORT.md` 真冲突与 synth 漏合）。
- `docs/check_next_stage_facts.py`：只读事实核对脚本（本审查附带的验收工具）。
- 本文件 `WORKER_REPORT.md`。

## 实际执行的只读命令与结果

| # | 命令（只读） | 结果 |
| --- | --- | --- |
| 1 | 读题目 `640.jpg`（封面）与 `640.jpg 8/9/11/12/13`、`题目/README.md` | 图像可读：必做 1–3、GELU/第二线性层为选做、3% 门槛、12 任务口径、System Bus 或 NICE 二选一、600 条、2.25 mm²、150 MHz |
| 2 | `git worktree list` / `git rev-parse --abbrev-ref HEAD` / `git log --oneline` | 9 个工作分支 + base + main；当前 worktree `d94fb6f` |
| 3 | 读 `design-integration/docs/KWT1_整体设计决策_20261018.md`（全 250 行） | 关键路径、周期数字、证据层级、§5 整合顺序 |
| 4 | 读 9 份 `WORKER_REPORT.md` | architecture/design-integration/driver/hook-fix/packaging/preprocess/quant/synth/verification |
| 5 | 读主目录 `run_e203.sh`、`tb_kwt_real_layer_e203.sv`、`e203_kwt_full_layer.S`、`e203_cpu_instance.svh`、`kwt_nice_system_model.sv`、`e203_kwt_nice_core.sv`、`kwt_tile_system_model.sv`、`kwt_dma.sv`、`kwt_tile_buffers.sv`、`kwt_nice_driver.c/h`、`test_kwt_nice_driver.c`、`driver/README.md`、`run_row_broadcast.py`、`run_quantized_ffn1_batch_rtl.py`、`config.v` | 见下方发现 |
| 6 | `git diff`（main 工作树）、`git diff --name-only 1494597 d94fb6f` | main 8 个未提交改动；`d94fb6f` 已含其内容 |
| 7 | `git merge-base <branch> d94fb6f`（9 分支）、`git rev-list --count` | 9/9 base = `d94fb6f`；领先 1–2 提交 |
| 8 | `git diff --name-only d94fb6f <branch>`（9 分支）、`git rev-parse <branch>:WORKER_REPORT.md` | **唯一重复路径 = 根 `WORKER_REPORT.md`**，9 个 blob 互不相同，基线无此文件 |
| 9 | `diff <(git show d94fb6f:...run_e203.sh) <main 工作副本>` + `sha256sum` | **逐字节相同**（`5fda3e15…`） |
| 10 | `ls`/`cat` `results/`：旧 vs ascii `finalizer.status`、12×`.exit`、`*_ascii_summary.json` | 旧 = `SUMMARIZE_EXIT 1` ×12；ascii = `SUMMARIZE_EXIT 0` ×12；每层 `838028` 拍 |
| 11 | `python3 docs/check_design_evidence.py --results-dir <main results>` | 退出 0，`PASS=72 WARN=0 FAIL=0` |
| 12 | `python3 docs/check_next_stage_facts.py --repo <main> --design-doc <design doc>` | 退出 0，`checks=21 FAIL=0`（FINDING 与 Q1–Q3 对应） |
| 13 | `grep` 全仓 A12W10 上界两值 | E203 manifest 逐层最大 **67,780,536**；软件标定另见 **67,826,716** |
| 14 | `grep` ITCM/`mem_icb` 相关 | TB 仅载 64 B ITCM；CPU `mem_icb_*` 全置零 |

## 主要发现（均带证据，非转述）

1. **候选路径已过、驱动未过**：E203 12 层通过是“测试台层次引用预置 208 描述符 + 16 条手写汇编”，C 驱动从未在 E203 执行（`driver/README.md` 第 5 行）。必做 3 的驱动门槛**未过**。
2. **C 驱动真执行缺三环**：链接/ITCM 按镜像装载（现在只 64 B）、CPU→共享 RAM 数据通路（`mem_icb` 置零）、输出回读。
3. **Q1**：设计报告第 24 行 `67,780,536` 与第 109 行 `67,826,716` 自相矛盾（同格式同标签）。
4. **Q2**：packaging 报告把过期（中文路径）0/12 当当前状态；实际 `*_ascii_*` 为 12/12。
5. **Q3**：§5 整合顺序既有假冲突（`run_e203.sh` 主目录副本 = 基线，无文本冲突），又漏真冲突（根 `WORKER_REPORT.md` add/add ×9）与 `codex/kwt-synth` 分支（第 215 行还写成“六个分支”）。
6. **Q4**：§2.4 A12W10 行的 bound 数字来源（软件标定）与行内证据标签（RTL/E203）不匹配。
7. **多样本爆炸**：L2 每样本 12 层 ≈ 1–2 分钟（mtime 实测），600 样本 ~10–20 h；建议数值走 L1、L2 层轮转（N+12）。
8. **16 行在关键路径后**：功能已由 8×16 可交付，16 行只在计算核边界，未接 E203。
9. **iSTA abort 不阻塞功能**：只阻塞 150 MHz 结论；脚本已改“失败即非 0”。
10. **A14W12 不切换**：只软件证据、上界占 50.59%，而 A12W10 门槛余量约 20×。

## 未证事项

- 本任务**未重跑**任何 E203 / Verilator / 综合仿真；`check_next_stage_facts.py` 只核对文件、哈希、退出码与分支元数据。
- 未验证 `d94fb6f` 与 main 工作树在所有非列名文件上逐字节一致（只对 `run_e203.sh` 做了哈希比对）。
- 未证明主目录未提交的 8 个文件与 `d94fb6f` 的对应文件**全部**一致（只抽查关键文件 `run_e203.sh`）。
- 工作包的“最长运行时间”是墙钟预算（基于实测），不是完成天数；未实测 WP-A/WP-B 的最终耗时。
- 未执行任何 merge；§5 流程为方案，须由人工在受控 worktree 执行并复核。
- 官方 SoC/PDK/宏/SDC/600 条评测集仍缺失，相关门槛未证。

## 结论

分支 `codex/kwt-next-analysis`；提交见 `git log -1`；结论：**必做 1–2 已可交付，必做 3 的 E203 驱动是真缺口，最短关键路径为“接口冻结 → C 驱动真执行 → 多样本功能”；综合/交付多数被外部资源阻塞；整合应在独立 worktree 从 `d94fb6f` 建分支，并专门处理根 `WORKER_REPORT.md` 的 add/add 冲突与 synth 漏合。**
