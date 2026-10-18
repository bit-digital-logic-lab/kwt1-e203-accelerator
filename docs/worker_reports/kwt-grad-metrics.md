# WORKER_REPORT — codex/kwt-integration（KWT-1 整合分支）

- 分支：`codex/kwt-integration`（worktree `/home/yang/kwt-worktrees/integration`）
- 基线：`main@1494597`
- 范围：把源文件快照 `d94fb6f` 与 8 个已完成工作分支整合为一个可审查的 Git 分支；
  归档 9 份 `WORKER_REPORT.md`；实跑快速验收并如实记录退出码；补 `docs/INTEGRATION_STATUS.md`。
- 边界：只在本 worktree 内操作；未 `checkout`/清理主目录；未 push / 未碰 `origin`；未开子智能体；
  未改核心数值 RTL；未重跑 12 层整层 E203；未做大面积综合。

详细冲突、来源 commit、逐项测试与缺口见 **[`docs/INTEGRATION_STATUS.md`](docs/INTEGRATION_STATUS.md)**。

## 提交

| 提交 | 内容 |
| --- | --- |
| `dc4fd2a` | 合入 `d94fb6f` 源文件快照（3 处冲突取 main@1494597 的已测钩子实现与 ascii finalizer；其余取快照；删除旧版钩子残留 plugin/assets） |
| `3ba1412` | 归档 hook-fix 报告（`c89095c` 逻辑已由 main 引入，不重复合入） |
| `fb71ede` | 整合 `codex/kwt-verification`（`1a3aa75`） |
| `282017e` | 整合 `codex/kwt-preprocess`（`747d5fa`） |
| `932db77` | 整合 `codex/kwt-quant`（`7afe9ff`） |
| `3e82ea9` | 整合 `codex/kwt-architecture`（`df93c87`） |
| `e1aa166` | 整合 `codex/kwt-driver`（`fcb4b72`，保留 `run_e203.sh` ASCII 路径修复） |
| `d758eeb` | 整合 `codex/kwt-synth`（`cb23d04`） |
| `8f6ca75` | 整合 `codex/kwt-design-integration`（`56b8a39`） |
| `29d0b00` | 整合 `codex/kwt-packaging`（`0731cf3`） |
| 本提交 | `docs/INTEGRATION_STATUS.md` + 本报告 |

9 份来源报告归档于 `docs/worker_reports/kwt-*.md`（verification / preprocess / quant /
architecture / driver / synth / design-integration / packaging / hook-fix）；合并过程中根
`WORKER_REPORT.md` 始终被移除并归档，**未发生后一份覆盖前一份**。

## 实跑验收（实际退出码）

| 命令 | 退出码 |
| --- | ---: |
| `git diff --check 1494597 HEAD` | **2**（仅 quant CSV 的 CRLF 与 packaging 报告末尾空行，既有产物；为保 provenance 未改，详见 STATUS §5#1） |
| `python3 -m unittest discover -s .opencode/skills/quarter-hour-check/tests` | **0**（20 tests OK） |
| `python3 docs/check_design_evidence.py --results-dir /home/yang/Desktop/ic设计竞赛/hardware/architecture/results` | **0**（`PASS=72 WARN=0 FAIL=0`） |
| 宿主 C 驱动测试（`cc ... test_kwt_nice_driver.c`） | **0**（`PASS provisional NICE FFN1 descriptor layout and bounds`） |
| RV32 交叉编译（`riscv64-linux-gnu-gcc ... -march=rv32imac -mabi=ilp32 -c`） | **0** |
| `negative_injection_check.py`（oss-cad-suite） | **0**（fp32/int16 污染均被检出 exit=1） |
| `run_tile_edges.sh`（oss-cad-suite） | **0**（正向 21 cases PASS；负向 MISMATCH 被检出） |
| （附加）`run_e203.sh --real-layers <中文 CASE_ROOT> 0 1` | **0**（12 层各 tile 0 单块 PASS，验证 ASCII 路径修复） |

## 关键口径裁定

- **E203 12 层状态以 `real_e203_validation0_ascii_*` 为准 = 单样本 12/12 PASS**；
  `real_e203_validation0_*` 的 `0/12 FAIL` 是**过期**的中文路径运行。packaging 报告中的
  “0/12” 不得当作当前结论。
- 保留 `codex/kwt-driver` 的 `run_e203.sh` ASCII 路径修复。
- `codex/kwt-synth` 已合入（含被修正的综合脚本），补上早前审查文档 §5 的遗漏。

## 未证事项

1. 官方 SoC / System Bus / NICE、PDK、SRAM 宏、SDC 缺失 → 无 150 MHz / ≤2.25 mm² / 功耗签核；
   iSTA 仍 abort（exit 134），无时序结论。
2. C 驱动**未**在 E203 指令流中执行（仅宿主 + RV32 编译）。
3. 官方 600 条评测集未发布 → 官方 NRMSE / 准确率未测；480 软件 / 240 RTL / 1 条 E203 / 官方 600
   是四个不同统计，不可混用。
4. `154,240`（L1）与 `838,028`（L2 E203）不可相减当加速比/吞吐；16 行核未接入 E203。
5. A14W12/A12W12/A16W10 只有软件证据。
6. `git diff --check` 非 0（纯格式/CRLF，见上）。
7. 本次未实际 merge 进 `main`、未 push；`main` 的既有未提交改动保持原样。
