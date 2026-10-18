# WORKER_REPORT（codex/kwt-driver）

## 任务与边界
修复 `hardware/rtl/control/tb/run_e203.sh` 在含中文的工程绝对路径下，把 real-layer 的 `+A/+W/+BIAS/+EXPECTED` 交给 Icarus `$readmemh` 而加载成 X 的问题；用脚本自动在 ASCII 临时目录做安全别名并在结束时清理，不要求用户手工建立 `/tmp` 别名。先用主目录 `hardware/architecture/results/real_e203_validation0_vectors` 的 layer01 做 tile 0、tile 207 两个上机回归，再跑 software/driver 宿主测试与 RV32 交叉编译。

范围仍按本科必做 1–3 项：`99×64 @ 64×256 + bias → 99×256`、12 套独立层参数、E203 驱动与综合。GELU/第二线性层/注意力为选做，本报告不涉及。

## 结论与证据层级
- **修复后 layer01 tile 0 与 tile 207 通过**，日志无 X。证据层级：**NICE-like E203 行为仿真**（Icarus 14.0 / oss-cad-suite + 本仓库仿真总线/SRAM 模型 vvp），**不是**组委会正式 SoC/PDK，**不是**综合网表，**不是** 150 MHz 时序或面积/功耗结论。
- **driver 宿主测试与 RV32 交叉编译通过**。证据层级：宿主/交叉编译单元级；**C 驱动未在 E203 仿真中执行**。
- `task_cycles=4025` 为**单 tile 仿真**记数，属仿真观察值，不能当吞吐/能耗/频率结论。

## 改动（仅 1 个文件）
`hardware/rtl/control/tb/run_e203.sh`
1. 保证 `build_dir` 落在 ASCII 临时根：`tmp_base` 依次取 `KWT_E203_TMP_BASE`、`TMPDIR`，非绝对路径或含非 ASCII 字节则回退 `/tmp`；`build_dir="$(mktemp -d "$tmp_base/kwt_e203.XXXXXX")"`，仍由 `trap ... EXIT` 删除。
2. `--real-layers` 分支：对每个 `layer??` 把 `a16/w16/bias32/expected32` 复制到 `"$build_dir/real_cases"`（ASCII 路径），再用别名路径传 `+A/+W/+BIAS/+EXPECTED`；不再把中文 `CASE_ROOT` 路径直接交给 `$readmemh`。
   - 因此运行结束自动清理，**不需要**手工 `ln -s` 建立 `/tmp/kwt-e203-ascii-cases` 之类的别名。

未改 C 驱动源码（宿主测试与交叉编译在改动前后均通过）。

## 实际执行的命令与退出结果
工具链：必须用 oss-cad-suite 的 `iverilog`/`vvp` 14.0（`PATH=/home/yang/yosys-sta/oss-cad-suite/bin:$PATH`）。`/usr/bin/iverilog` 11.0 无法解析 E203 的 SV 立即断言（`sirv_gnrl_dffs.v:266`），与本修复无关；主目录运行也使用 oss-cad-suite 的 vvp。

回归 `CASE_ROOT=/tmp/opencode/driver_e203_回归`（路径含中文），内含指向主目录 vectors 的 `layer01_*` 符号链接（仅 layer01），以在非 ASCII 路径下复现并验证。

| # | 命令 | 退出码 | 结果 |
| --- | --- | --- | --- |
| 1 | 最小复现：`$readmemh` 中文路径 | — | `WARNING ... non-printable characters`，`w0=xxxx`；ASCII 拷贝后 `w0=fbf2` |
| 2 | 修复前 runner（`git show HEAD:...`）`--real-layers <中文CASE_ROOT> 0 1` | **1** | 4 条 `non-printable characters` + `FATAL ... layer output [0,0] got=xxxxxxxx` |
| 3 | 修复后 `--real-layers <中文CASE_ROOT> 0 1` | **0** | `PASS ... start=0 count=1 outputs=128 padding=0 task_cycles=4025` |
| 4 | 修复后 `--real-layers <中文CASE_ROOT> 207 1` | **0** | `PASS ... start=207 count=1 outputs=48 padding=80 task_cycles=4025` |
| 5 | `cc -std=c11 -O2 -Wall -Wextra -Werror software/driver/kwt_nice_driver.c software/driver/test_kwt_nice_driver.c -o /tmp/opencode/kwt-nice-driver-test && /tmp/opencode/kwt-nice-driver-test` | **0** | `PASS provisional NICE FFN1 descriptor layout and bounds` |
| 6 | `riscv64-linux-gnu-gcc -std=c11 -O2 -Wall -Wextra -Werror -ffreestanding -march=rv32imac -mabi=ilp32 -c software/driver/kwt_nice_driver.c -o /tmp/opencode/kwt_nice_driver_rv32.o` | **0** | 生成 rv32 目标文件 |

清理核验：运行后无残留 `/tmp/kwt_e203.*`；临时脚本与临时 CASE_ROOT 已删除；`git status` 仅剩本次有意修改。

## 未证事项
- 未跑 12 层全量（主目录正在跑），未验证 layer02–12 与 tile 0/207 之外的 tile。
- 未在组委会正式 SoC、系统总线或 NICE 接口上联调；驱动地址与指令编码只与本仓库仿真模型一致。
- C 驱动未在 E203 仿真中运行（`software/driver/README.md` 同此声明）。
- 无综合、无 150 MHz 时序、无面积/功耗/吞吐/能耗结论。
- 本次未复测 NRMSE；manifest 中 layer01 `nrmse_pct≈0.1017%` 属**软件向量生成阶段**证据，未在本次 RTL 回归中复核。
- GELU、第二线性层、注意力、完整编码器未做。

---

# WORKER_REPORT（16 行核接入 NICE-like E203/DMA 系统模型）

## 任务与边界
把 16 行计算核（`kwt_row_broadcast_tile`）接入与旧 8×16 路径**同一**的 NICE-like 命令/
`128-bit` 仿真总线/16-bit SRAM/DMA 系统模型，闭合“高性能核只在 L1 测试台”的差距。
只改本 worktree `kwt-worktrees/row16-e203`；未改旧 8×16 核、未改 `run_e203.sh`、未 push/merge/rebase。

## 结论与证据层级
- **M1 完成**：`hardware/optimization/ROW16_E203_INTEGRATION.md` 冻结 16 行核端口、
  `cmd_row_mask` 由 8→16 位、A/W/bias/输出地址图、`208→112` tile 控制变化，并列 G1–G6 接口缺口。
- **M2 完成并上机通过**：新适配器 `hardware/rtl/row16_integration/kwt_row16_tile_adapter.sv`
  + 测试台 `hardware/rtl/row16_integration/tb/tb_kwt_row16_adapter.sv`。冻结 A12W10 **同一真实向量**
  （与旧 E203 路径同一 `layerNN_a16/w16/bias32/expected32.hex`），跑 **tile0** 与 **尾 tile
  （rt=6,ct=15,row_mask=16'h0007，行 96..98）**，对软件独立生成的 `expected32` **逐位一致**：
  `outputs=304 padding=208`（tile0 256 + 尾 tile 48 有效 + 208 填充行）。
- **证据层级**：**NICE-like E203/DMA 行为仿真**（Icarus 11 与 14、Verilator 5.047），
  **不是**组委会正式 SoC/System Bus/NICE 平台，**不是**综合网表，**不是** 150 MHz 时序/面积/功耗结论。
- A/W/bias 由 `kwt_dma` 经共享总线真实搬运；输出经 `kwt_dma` 写回；DUT 用 16 行核真实算 MAC。
  **未**在测试台预计算 `XW+b`，**未**用 golden 当 DUT 输出。

## 改动（全部新增，未改旧文件）
- `hardware/optimization/ROW16_E203_INTEGRATION.md`（M1 设计/接口缺口）
- `hardware/rtl/row16_integration/kwt_row16_tile_adapter.sv`（16 行单 tile 适配器）
- `hardware/rtl/row16_integration/tb/tb_kwt_row16_adapter.sv`（半上机测试台）
- `hardware/rtl/row16_integration/tb/run_row16.sh`（独立 runner；ASCII 临时目录暂存 hex）
- `hardware/optimization/row16_integration_logs/iverilog_regression.log`（运行记录）

## 实际执行的命令与退出结果
工具链：`/home/yang/yosys-sta/oss-cad-suite/bin` 的 `iverilog`/`vvp`；另用 `/usr/bin/iverilog`(v11)
与 `verilator` 5.047 复核。向量目录 `VECTORS=/home/yang/Desktop/ic设计竞赛/hardware/architecture/results/real_e203_validation0_vectors`。

| # | 命令 | 退出码 | 结果 |
| --- | --- | --- | --- |
| 1 | `run_row16.sh $VECTORS 01`（iverilog v14） | **0** | `PASS ... outputs=304 padding=208 total_cycles=9084` |
| 2 | `run_row16.sh $VECTORS 12`（iverilog v14） | **0** | 同上（layer12） |
| 3 | `run_row16.sh $VECTORS 01`（/usr/bin iverilog v11） | **0** | 同上 |
| 4 | `run_row16.sh $VECTORS 01 --verilator` | **0** | 同上（Verilator 5.047，build+run） |
| 5 | 修复前中间版本（a_words 范围 3584，实为 6656） | **1** | `$readmemh ... Too many words` + 尾 tile `got=xxxxxxxx` 的**可复现失败用例**，据此定位 8 行 tiling 索引越界 |

`total_cycles=9084` 为两 tile 的仿真观察值，不能当吞吐/能耗/频率结论。

## 未证事项（诚实边界）
- 只跑 **单 tile**（tile0 + 尾 tile），**未**跑全层 112 tile、**未**跑 12 层（M3 未做）。
- **未**把 16 行适配器接进真正由 E203 CPU 发 NICE 指令的路径；本适配器只暴露等价命令口，
  描述符 `row_mask` 16 位重排（G1）、控制模型硬编码（G2/G3）、A 16 路并行读 vs 1R1W 宏（G4）、
  relu（G5）均为待改造缺口，详见 M1 文档。
- **未**改 `run_e203.sh` 与旧 8×16 核；**未**在组委会正式平台/总线/NICE 上联调。
- **无**综合、**无** 150 MHz STA、**无**面积/功耗/能耗结论（需目标库与指定 16-bit 宏）。
- 未复测 NRMSE；`expected32` 逐位一致不等于官方评测集 NRMSE。
