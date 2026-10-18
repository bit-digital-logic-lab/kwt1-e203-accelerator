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
