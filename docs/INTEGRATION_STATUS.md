# KWT-1 整合分支状态（codex/kwt-integration）

> 目的：把并行工作的 **源文件快照 `d94fb6f`** 与 **8 个已完成工作分支** 整合成一个可审查的
> Git 分支，保留每份 provenance，消除会互相覆盖的根 `WORKER_REPORT.md` 冲突，并如实记录
> 冲突处理、来源 commit、实跑测试退出码与保留缺口。
>
> 约束（本次严格遵守）：只在 `/home/yang/kwt-worktrees/integration` 内操作；未 `checkout`/清理
> 主目录 `/home/yang/Desktop/ic设计竞赛`；未 push、未碰 `origin`；未开子智能体；未改核心数值
> RTL；未重跑 12 层整层 E203，未做大面积综合。

- 基线：`main@1494597`（当前分支 `codex/kwt-integration` 建立于此）
- 最终 HEAD：见根 `WORKER_REPORT.md` 与本文件末尾
- 工作树状态：干净（`git status --porcelain` 为空）

---

## 1. 第一步：合入 `d94fb6f`（旧主工作树选定源文件快照）

`d94fb6f`（`codex/kwt-dispatch-base` / `codex/kwt-next-analysis`）是**主工作树部分未提交文件**
的选定快照，**不是**主目录全部未提交文件。它相对基线 `d6719eb` 新增/修改了：

- 新增：`software/driver/*`、`hardware/rtl/control/tb/prepare_real_e203_vectors.py`、
  `tb_kwt_real_layer_e203.sv`、`summarize_real_e203.py`、两个 `finalize_real_e203_validation0*.sh`、
  `AGENTS.md`、`opencode.json`、`doc/本科FFN1数据与硬件边界核对_20261017.md`、
  `本科组审查纠偏与下一步_20261017.md`、`.opencode/plugins/quarter-hour-check.js`、
  `.opencode/skills/quarter-hour-check/**`（旧版钩子）。
- 修改：`.gitignore`、`hardware/architecture/run_quantized_ffn1_batch_rtl.py`、
  `run_row_broadcast.py`、`tb_kwt_row_broadcast_staged_task.sv`、`run_e203.sh`、
  `software/README.md`、`compare_ffn1_calibration.py`、`export_ffn1_quantized.py`。

合并提交：`dc4fd2a`（`--no-ff`，merge-base = `d6719eb`）。

### 1.1 冲突与解决（3 处 add/add）

| 文件 | 处理 | 理由 |
| --- | --- | --- |
| `.opencode/skills/quarter-hour-check/SKILL.md` | 保留 `main@1494597`（ours） | main 的**已测**钩子实现 |
| `.opencode/skills/quarter-hour-check/scripts/check.py` | 保留 `main@1494597`（ours） | 同上 |
| `hardware/rtl/control/tb/finalize_real_e203_validation0_ascii.sh` | 保留 `main@1494597`（ours） | main 的 trap 终态登记实现 |

### 1.2 快照中不再保留的钩子残留

快照仍带有**旧版钩子**：`.opencode/plugins/quarter-hour-check.js` 与
`.opencode/skills/quarter-hour-check/assets/{quarter-hour-check.service,quarter-hour-check.timer}`。
`main@1494597` 已用 `check.py` + `finalize_hook.sh` + `tests/` 替换该实现。为“保留 main 的已测
钩子实现、不重复合入”，本次在合并提交中删除这三个残留文件（`git rm`）。验证：
`git diff 1494597 -- .opencode` 与 `git diff 1494597 -- hardware/rtl/control/tb/finalize_real_e203_validation0_ascii.sh`
均为空。

### 1.3 其它源码

除上述两项外，合并结果一律取快照内容（`AGENTS.md`、`opencode.json`、文档、驱动源码、
向量生成/汇总脚本、`run_e203.sh` 等）。`codex/kwt-hook-fix` 的提交 `c89095c` 未重复合入——
其逻辑已由 `main@1494597` 以同等实现引入（见 §3）。

---

## 2. 第二步：整合 8 个工作分支（保留 provenance）

每个分支都以 `d94fb6f` 为 merge-base（`codex/kwt-synth`、`codex/kwt-hook-fix` 亦同）。
所有分支均在仓库根新增 `WORKER_REPORT.md`，blob 互不相同，直接顺序 merge 会 **add/add 冲突**
并让后一份**覆盖**前一份。

处理方式：每合一个分支前，先把该分支的报告 **归档**到 `docs/worker_reports/kwt-<分支>.md`，
再在合并提交中删除根 `WORKER_REPORT.md`（根文件不进入整合树，最后统一重写为整合分支自己的报告）。

| 序 | 来源分支 | 来源 commit | 整合提交 | 归档报告 | 主要新增/修改 |
| ---: | --- | --- | --- | --- | --- |
| 1 | `codex/kwt-verification` | `1a3aa75`（含 `c12a0ea`） | `fb71ede` | `docs/worker_reports/kwt-verification.md` | tile 级边界用例、独立 golden 负向注入、`negative_injection_check.py` |
| 2 | `codex/kwt-preprocess` | `747d5fa` | `282017e` | `docs/worker_reports/kwt-preprocess.md` | `98×40→99×64` 输入边界独立审计 + unittest |
| 3 | `codex/kwt-quant` | `7afe9ff`（含 `b995458`） | `932db77` | `docs/worker_reports/kwt-quant.md` | 单容器位宽对照、INT32 边界地图、结果 CSV/JSON |
| 4 | `codex/kwt-architecture` | `df93c87` | `3e82ea9` | `docs/worker_reports/kwt-architecture.md` | FFN1 流量/容量/周期取舍脚本与结论 |
| 5 | `codex/kwt-driver` | `fcb4b72` | `e1aa166` | `docs/worker_reports/kwt-driver.md` | `run_e203.sh` **ASCII 路径修复**（本分支明确保留） |
| 6 | `codex/kwt-synth` | `cb23d04` | `d758eeb` | `docs/worker_reports/kwt-synth.md` | 修正后的 `run_ffn1_8x16_sta.sh`（失败即透传非 0）、综合状态说明 |
| 7 | `codex/kwt-design-integration` | `56b8a39` | `8f6ca75` | `docs/worker_reports/kwt-design-integration.md` | 整体设计决策、`docs/check_design_evidence.py` |
| 8 | `codex/kwt-packaging` | `0731cf3` | `29d0b00` | `docs/worker_reports/kwt-packaging.md` | `submission/` 证据与缺口清单、`evidence_check.py` |

`codex/kwt-synth` 分支**必须**合入：它带有唯一被修正的综合脚本 `hardware/synth/run_ffn1_8x16_sta.sh`
与 `hardware/synth/FFN1_8x16_synthesis_status.md`；早前审查文档 §5 曾漏列该分支。

### 2.1 hook-fix 报告单独归档

`codex/kwt-hook-fix`（`c89095c`）的**代码**不合入（`main@1494597` 已含同等逻辑），仅保留其
provenance：`docs/worker_reports/kwt-hook-fix.md`（提交 `3ba1412`）。

### 2.2 归档的九份报告

```
docs/worker_reports/kwt-architecture.md
docs/worker_reports/kwt-design-integration.md
docs/worker_reports/kwt-driver.md
docs/worker_reports/kwt-hook-fix.md
docs/worker_reports/kwt-packaging.md
docs/worker_reports/kwt-preprocess.md
docs/worker_reports/kwt-quant.md
docs/worker_reports/kwt-synth.md
docs/worker_reports/kwt-verification.md
```

根 `WORKER_REPORT.md` 在整合过程中始终不存在，避免任何一份报告覆盖另一份；整合完成后由
整合工作本身写入一份新的根 `WORKER_REPORT.md`。

---

## 3. 必须区分的事实：E203 12 层“0/12 FAIL”是**过期**结论

`codex/kwt-packaging` 的报告（`docs/worker_reports/kwt-packaging.md`）记录：
“主目录 `real_e203_validation0` 12 层**全 FAIL（0/12）**，疑中文路径 `$readmemh`”。
该描述对**旧运行**本身为真，但**不是当前结论**，不得当作当前状态：

- 旧运行 `real_e203_validation0_*`：`finalizer.status = SUMMARIZE_EXIT 1`，12 个 `layer*.exit` 全为 `1`，
  日志含 `got=xxxxxxxx`——**被替换过的中文路径运行**。
- 修正运行 `real_e203_validation0_ascii_*`：`finalizer.status = SUMMARIZE_EXIT 0`，12 个 `layer*.exit`
  全为 `0`；每层 `outputs=25344 padding=1280 task_cycles=838028`；合计 `checked_outputs=304128`、
  `checked_padding=15360`、`total_task_cycles=10056336`。
- 摘要：`/home/yang/Desktop/ic设计竞赛/hardware/architecture/results/real_e203_validation0_ascii_summary.json`
  → **单样本 12 层 12/12 PASS**。
- 独立核对脚本 `docs/check_design_evidence.py` 退出码 `0`，`SUMMARY: PASS=72 WARN=0 FAIL=0`
  （同时并置校验了“旧运行 0/12 失败”这一历史事实）。

**结论冻结**：`*_ascii_*` 为当前状态；`real_e203_validation0_*` 为历史/过期对照。

---

## 4. 需保留的 driver ASCII 路径修复

`codex/kwt-driver` 对 `hardware/rtl/control/tb/run_e203.sh` 的修复已随 `e1aa166` 合入并保留：
把 `--real-layers` 的 `+A/+W/+BIAS/+EXPECTED` 复制到 ASCII 临时目录后再交给 Icarus `$readmemh`，
避免中文工程路径被加载为 `x`。该修复未改动 C 驱动源码。

---

## 5. 实跑快速验收（实际退出码）

全部在整合 worktree 根目录、整合后的工作树上执行。未重跑 12 层整层，未做大面积综合。

| # | 命令 | 退出码 | 结果摘要 |
| ---: | --- | ---: | --- |
| 1 | `git diff --check 1494597 HEAD` | **2（非 0，见下）** | 仅两处**既有产物**的行尾问题：`software/results/ffn1_container_headroom_12labels_40.csv`、`..._bounds.csv`（quant 分支使用 CRLF，合并前后 rc 均为 2）与 `submission/evidence_report.md`（文件末尾空行）。**未修改**这些来源分支产物以保 provenance；非核心 RTL、不影响构建/测试。 |
| 2 | `python3 -m unittest discover -s .opencode/skills/quarter-hour-check/tests` | **0** | `Ran 20 tests ... OK` |
| 3 | `python3 docs/check_design_evidence.py --results-dir /home/yang/Desktop/ic设计竞赛/hardware/architecture/results` | **0** | `SUMMARY: PASS=72 WARN=0 FAIL=0`；含 ASCII 12 层、旧 0/12 对照、向量/哈希/INT32 边界 |
| 4 | `cc -std=c11 -O2 -Wall -Wextra -Werror software/driver/kwt_nice_driver.c software/driver/test_kwt_nice_driver.c -o /tmp/opencode/kwt-nice-driver-test && /tmp/opencode/kwt-nice-driver-test` | **0** | `PASS provisional NICE FFN1 descriptor layout and bounds` |
| 5 | `riscv64-linux-gnu-gcc -std=c11 -O2 -Wall -Wextra -Werror -ffreestanding -march=rv32imac -mabi=ilp32 -c software/driver/kwt_nice_driver.c -o /tmp/opencode/kwt_nice_driver_rv32.o` | **0** | 生成 `ELF 32-bit ... RISC-V ... soft-float ABI` 目标文件 |
| 6 | `PATH=/home/yang/yosys-sta/oss-cad-suite/bin:$PATH python3 hardware/testbenchplus/common/negative_injection_check.py` | **0** | 原 golden 通过；污染后 fp32/int16 均 `exit=1` 被检出 |
| 7 | `PATH=/home/yang/yosys-sta/oss-cad-suite/bin:$PATH bash hardware/rtl/control/tb/run_tile_edges.sh` | **0** | 正向 `PASS ... cases=21 ... task_cycles=82406`；负向 `MISMATCH ... [98,255] ... NEGATIVE_INJECTION_DETECTED`；子运行 exit=1 被判定检出 |
| 8 |（附加）`PATH=... bash hardware/rtl/control/tb/run_e203.sh --real-layers "/home/yang/Desktop/ic设计竞赛/hardware/architecture/results/real_e203_validation0_vectors" 0 1` | **0** | 真实**中文** CASE_ROOT 下，12 层各 tile 0 单块 `PASS ... task_cycles=4025`；证明 ASCII 路径修复生效（非整层、非吞吐结论） |

> 说明：第 8 项是**每层单块 tile**回归，用来证明 driver 的 ASCII 路径修复在真实非 ASCII 路径下
> 有效；它**不是** 12 层整层、不是吞吐/能耗/频率结论。设计文档 §5 的 `0 208` 整层运行本次**未**执行。

---

## 6. 保留缺口与未证事项（禁止夸大）

1. **`git diff --check` 非 0**：原因是 quant 分支产物 CRLF 与 packaging 报告末尾空行（见 §5#1）。
   为保 provenance 未改；如评审要求 0 退出，可另加纯格式提交，但需注明会改动来源产物字节。
2. 正式组委会 SoC / System Bus / NICE 规范、PDK、SRAM 宏、SDC **全部缺失**：无 150 MHz、
   无 ≤2.25 mm²、无功耗/能耗签核。iSTA 仍 abort（exit 134），**无时序结论**。
3. **C 驱动未在 E203 指令流中执行**：现有证据仅为宿主测试 + RV32 交叉编译；E203 全层运行仍是
   测试台预置描述符 + 汇编循环，未跑 C 驱动。
4. 官方隐藏 **600 条评测集不发布** → 官方 NRMSE / 分类准确率**未测**。本地统计不得混用：
   480 软件 / 240 RTL / 1 条 E203（单样本 12 层）/ 官方 600 是四个不同统计。
5. `154,240`（16 行，L1 测试台）与 `838,028`（8×16，L2 E203）**边界不同，不可相减当加速比/吞吐**。
6. 16 行核**未接入 E203**；A14W12/A12W12/A16W10 只有**软件证据**（无 RTL 逐位、无综合）。
7. 文档缺陷（**未在本整合分支修 RTL，也不改来源文档数值**）：A12W10 保守累加上界在两处为
   `67,780,536`（`ffn1_a12w10_static_sym.json` 与 E203 向量 manifest 逐层最大值）与
   `67,826,716`（`ffn1_multi_scale_12labels_20.json` / 多方案标定，quant 分支摘要为 `67,780,536`）。
   两者都 < 2³¹−1，属**不同标定批次**的口径差异，需在正式交付中唯一化，但本次不修改来源数据。

---

## 7. HEAD 与可复现

- 整合 HEAD：见根 `WORKER_REPORT.md`（`git rev-parse HEAD`）。
- 本分支所有提交均为本地提交；**未 push**，**未 merge 进 main**。
- `main` 的既有未提交改动保持原样。
