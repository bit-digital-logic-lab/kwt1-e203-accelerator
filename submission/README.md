# 可复现交付入口（packaging 分支）

本目录只提供**证据核对入口**与**复现说明**，不改变任何 RTL 或软件核心。

- 证据/缺口清单（人读）：[`本科必做1-3项证据与缺口清单_20261018.md`](本科必做1-3项证据与缺口清单_20261018.md)
- 自动核对脚本：[`evidence_check.py`](evidence_check.py)
- 最近一次核对产物：[`evidence_report.json`](evidence_report.json) / [`evidence_report.md`](evidence_report.md)

## 1. 一键核对

```bash
cd <本 worktree 根目录>
python3 submission/evidence_check.py --run-fast
```

`--run-fast` 会实际执行两件轻量测试：

1. C 驱动宿主测试（`cc … && /tmp/kwt-nice-driver-evidence`）；
2. C 驱动 RV32 交叉编译（`riscv64-linux-gnu-gcc … -c`）。

不带 `--run-fast` 时只核对文件/哈希/工具存在性，这两项记为 `NOT_RUN`。

退出码：`0` 表示核对流程正常完成；结论以 `evidence_report.json` 的
`items[].status` 为准。**PASS 只给已上机且产物齐备的条目**，
`ENTRY`（仅入口）、`PARTIAL`（声称无日志）、`FAIL`、`GAP` 一律不算通过。

## 2. 环境依赖

| 用途 | 命令 | 本次核对 |
| --- | --- | --- |
| 证据核对 | `python3` | 3.10.12 |
| L1 数值核 | `verilator` | 5.030 |
| L2 仿真 | `iverilog` + `vvp` | 已安装 |
| L4 综合 | `yosys` + 示例 Liberty | 已安装 |
| L3 驱动 | `cc` / `riscv64-linux-gnu-gcc` | 11.4.0 / 11.4.0 |

## 3. 分层复现命令

### L3 C 驱动（本机可独立复现）

```bash
cc -std=c11 -O2 -Wall -Wextra -Werror \
  software/driver/kwt_nice_driver.c software/driver/test_kwt_nice_driver.c \
  -o /tmp/kwt-nice-driver && /tmp/kwt-nice-driver
# 期望 stdout: PASS provisional NICE FFN1 descriptor layout and bounds

riscv64-linux-gnu-gcc -std=c11 -O2 -Wall -Wextra -Werror -ffreestanding \
  -march=rv32imac -mabi=ilp32 -c software/driver/kwt_nice_driver.c \
  -o /tmp/kwt_nice_driver_rv32.o
```

宿主测试只证明描述符生成与边界检查；**驱动尚未在 E203 仿真中执行**，
不等于 SoC 联调通过。

### L1 数值核（Verilator）

```bash
# 需要样本归档 software/.build/quant_samples_12labels_20.npz
# 记录 SHA-256 = ee04e849bc407149618e4ecd5ff99575687586d4e57c0ade6a0b3f7b66c8c149
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python3 hardware/architecture/run_quantized_ffn1_batch_rtl.py \
  --scheme A12W10 --activation-bits 12 --per-label 0 --rows 16 --staged \
  --out /tmp/kwt-a12w10-rtl
```

该归档被 `.gitignore` 排除、**不在提交树内**；主目录有同哈希副本可只读引用。
缺归档时本步不可复现，`evidence_check.py` 会将其记为 `GAP`，不得当成 PASS。

### L2 NICE-like E203 仿真（非正式 SoC）

```bash
bash hardware/rtl/control/tb/run_e203.sh --full-layer          # 合成整数整层
bash hardware/rtl/control/tb/run_e203.sh --real-layers CASE_ROOT 0 4  # 真实向量
```

- 需要只读的上游 E203：`资料/e203_hbirdv2_full -> /home/yang/Desktop/ic设计竞赛/资料/e203_hbirdv2_full`。
- 已知坑：工程绝对路径含中文时，`$readmemh` 可能把真实层向量载入为 X。
  主目录 `real_e203_validation0` 的 12 层因此全 FAIL（0/12）；
  新脚本 `finalize_real_e203_validation0_ascii.sh` 在 ASCII 临时目录做安全别名。
- 这是 **NICE-like** 系统模型，不是组委会正式 SoC。

### L4 示例库综合

```bash
LIBERTY=/home/yang/yosys-sta/pdk/nangate45/lib/Nangate45_typ.lib \
  bash hardware/synth/run_engine.sh
```

结果只有示例库相对值，**不能**与题目 `2.25 mm²` / `150 MHz` 比较。

## 4. 官方来源校验

`资料/来源校验.json` 用 git blob SHA 锁定指定官方文件；`evidence_check.py`
会逐条 `git hash-object` 复核。本次全部一致，见报告
`official_source_integrity` 段。
