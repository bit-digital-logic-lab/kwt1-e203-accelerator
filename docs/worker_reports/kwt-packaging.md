# WORKER_REPORT — packaging 分支（本科必做 1-3 项证据/缺口清单）

## 结论

依据题目截图 `题目/640.jpg 8` 与现有工程，产出**本科必做 1-3 项**的证据/缺口清单与可复现交付入口。核心纠偏：截图原文写明“**GELU 和第二线性层列为本科选做内容**”，因此 **GELU 不是本科必做，也不是 3% NRMSE 的验收对象**；本科必做止于第一线性层 `X[99,64]@W[64,256]+b[256]→[99,256]`，12 层为依次换参的独立任务。

按证据层级区分：**L1** 已上机数值核（Verilator）、**L2** NICE-like E203 仿真、**L3** C 驱动宿主测试 / RV32 编译、**L4** 示例库综合（Yosys+Nangate45）；组委会 System Bus/PDK/宏/SDC/600 条评测集记为 **GAP**。所有数值/周期/面积结论均在文档中标注层级，缺失证据一律不写 PASS。

## 改动（仅新增 `submission/`，未动 RTL/软件核心）

- `submission/evidence_check.py`：证据核对脚本。核对证据文件存在性与哈希、官方来源 git blob SHA、工具存在性；扫描官方材料是否缺失；`--run-fast` 实跑 C 驱动宿主测试与 RV32 编译；输出 `evidence_report.json/.md`。PASS 仅给已上机且产物齐备条目。
- `submission/evidence_report.json` / `evidence_report.md`：脚本实跑产物（本次提交）。
- `submission/本科必做1-3项证据与缺口清单_20261018.md`：人读清单，逐项带证据层级、状态与未证事项。
- `submission/README.md`：可复现交付入口（分子层级命令、环境依赖、已知坑）。

## 实际执行的命令与退出结果

| 命令 | 退出/结果 |
| --- | --- |
| `python3 submission/evidence_check.py --run-fast` | 退出码 0；汇总 `PASS 7 / PARTIAL 2 / ENTRY 2 / FAIL 1 / GAP 6` |
| `cc -std=c11 -O2 -Wall -Wextra -Werror software/driver/kwt_nice_driver.c software/driver/test_kwt_nice_driver.c -o /tmp/kwt-nice-driver-evidence && /tmp/kwt-nice-driver-evidence` | 退出码 0，stdout `PASS provisional NICE FFN1 descriptor layout and bounds` |
| `riscv64-linux-gnu-gcc … -march=rv32imac -mabi=ilp32 -c software/driver/kwt_nice_driver.c -o /tmp/kwt_nice_driver_evidence_rv32.o` | 退出码 0 |
| `git hash-object`（逐条核对 `资料/来源校验.json` 的 5 个官方文件） | 5/5 与记录一致 |
| `sha256sum`（核对冻结参数/权重） | `kwt1_fp32.npz`、`ffn1_blackbox_a11w10.npz`、`ffn1_a12w10_static_sym.npz` 均与摘录 SHA-256 一致 |

关键层级化事实：240×12 A11W10 最差层 NRMSE **0.18216320%**、A12W10 **0.14739360%**（均为 **L1**，本地 240 条）；16 行 staged **154,240 拍/层**（**L1-cycle**，含测试台定时模型）；16 行核示例库面积 **120,434.692 库单位**（**L4**，不可与 2.25 mm² 比较）。

## 未证事项（不计通过）

1. 官方 600 条精度结果（现仅本地 240/480 条）。
2. 真实量化样本经 E203+DMA+RTL 逐位一致：主目录 `real_e203_validation0` 12 层**全 FAIL（0/12）**，疑中文路径 `$readmemh`；此为**只读外部证据**，本报告记为 FAIL。
3. C 驱动在 E203/SoC 仿真中实际执行（现有仅宿主测试 + RV32 编译）。
4. 目标 PDK/SRAM 宏/SDC 下的综合、STA、150 MHz、≤2.25 mm²、功耗与单任务能耗。
5. `--full-layer` 合成整数整层 PASS 的运行日志（现仅文档转述，记为 PARTIAL）。
6. `software/.build` 样本归档不在提交树（`.gitignore` 排除），复现依赖主目录同哈希副本，记为 **GAP**。
7. 154,240 / 838,028 拍不得当作 SoC 吞吐量或加速比结论。

## 边界声明

本分支仅新增 `submission/` 内容；未修改 `/home/yang/Desktop/ic设计竞赛` 主目录、他人 worktree、官方资料与共享数据（仅只读引用）；未 push/merge/rebase。
