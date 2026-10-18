# WORKER_REPORT — 本科必做验证缺口

分支：`codex/kwt-verification`
范围：只新增验证文件，不改计算核、量化算法、主目录或共享资料。

## 改动

新增（均在 `hardware/rtl/control/tb/`）：

- `tb_kwt_tile_edge_cases.sv`：在 `kwt_tile_system_model`（DMA + 8×16 引擎 +
  行为级外部存储/总线）上运行的 tile 级自检。按 `cases.hex` 顺序执行，逐项与
  外部 golden（`golden.hex`）比较，填充行必须为 0。
- `run_tile_edge_cases.py`：只读加载主目录
  `hardware/architecture/results/real_e203_validation0_vectors/layer01_*`，校验
  manifest 中 4 个文件的 SHA256；用整数点积**独立重算** golden 并与
  `expected32` 交叉核对；把算子与 golden 暂存为 ASCII 文件再给 Icarus（规避中文
  路径）；先跑正向，再复制 golden、把最后一个被检查的有效元素 +1 跑负向注入。
- `run_tile_edges.sh`：入口脚本，`python3 -u` 调用上面的驱动。
- `TILE_EDGE_TESTS.md`：覆盖项、判据与证据层级说明。

新增（`hardware/testbenchplus/common/negative_injection_check.py`）：对两个示例
向量各跑两次未改动的 `run_vectors.py`——原 golden 必须通过，单个 `expected`
元素污染后必须失败——以证明 testbenchplus 的独立 golden 真的能拦住错值。

覆盖：独立 golden 负向注入、末 tile（行 96..98 有效 / 99..103 填充，16 个列
tile）、无复位连续任务、计算中注入复位并在复位后恢复核对。

## 实际执行的命令与退出结果

均在当前 worktree 根目录执行；Icarus 使用 oss-cad-suite 的 14.0（Ubuntu 自带
11.0 无法展开冻结 RTL 的未打包数组，属既有环境问题）。

| 命令 | 退出码 | 结果 |
| --- | ---: | --- |
| `PATH=/home/yang/yosys-sta/oss-cad-suite/bin:$PATH bash hardware/rtl/control/tb/run_tile_edges.sh` | 0 | 正向 `PASS`：21 个 case，末 tile 768 有效 + 1280 填充，连续 19 次，复位 1 次；负向注入 `MISMATCH case=20 row=98 col=255`，子运行退出 1，被脚本判定为“已检出” |
| `PATH=/home/yang/yosys-sta/oss-cad-suite/bin:$PATH bash hardware/testbenchplus/run_all_tests.sh --vectors-only` | 0 | 既有 testbenchplus 向量 2/2 通过（int16 逐位、fp32 NRMSE 0） |
| `PATH=/home/yang/yosys-sta/oss-cad-suite/bin:$PATH python3 hardware/testbenchplus/common/negative_injection_check.py` | 0 | 原 golden 通过；污染后 fp32/int16 均以 exit=1 检出 |
| `PATH=/home/yang/yosys-sta/oss-cad-suite/bin:$PATH bash hardware/rtl/control/tb/run.sh` | 0 | 既有 tile system model 通过（128 个结果） |
| `bash hardware/rtl/control/tb/run.sh`（默认 `/usr/bin/iverilog` 11.0） | 1 | 既有 RTL 的 `unresolved wire array`；非本次改动引入 |

关键自检输出（正向）：

```
independent_golden: valid=1328 padding=1360 expected32_cross_checks=1328
cases=21 reset_index=19 last_tile_valid=768 last_tile_padding=1280
PASS tile edge cases: cases=21 checked_tiles=20 valid=1200 padding=1360
  reset=1 continuous=19 task_cycles=82406 total_cycles=82405
```

负向注入输出：`MISMATCH ... golden mismatch case=20 [98,255]`，
`NEGATIVE_INJECTION_DETECTED golden_index=2607 exit=1`。

## 证据层级

- 以上均为 **行为级/Icarus tile 级** 结果：外部存储、总线、SRAM、时序是简化
  模型，不是组委会 SoC 或综合后网表。
- `task_cycles/total_cycles` 只是 harness 观察值，**不构成** 150 MHz、吞吐、
  面积或功耗结论；本套件不做综合与时序。
- golden 独立于 DUT：由冻结 `a16/w16/bias32` 重算，且与给定 `expected32`
  1 328 项逐项一致；SHA256 由脚本按 manifest 校验。

## 未证事项

- 未跑 208 块/12 层整层，未跑真实 E203 SoC 或竞赛平台，未做综合/时序/面积/
  功耗；GELU 与第二线性层未涉及。
- 负向注入只验证“独立 golden 被污染时测试能报错”，不是对 DUT 的故障注入。
- 只覆盖 layer01、validation_index 0 的单个真实样本；12 层换参由既有
  `run_e203.sh` 系列覆盖，本套件不重复。
- Icarus 11 与既有控制类测试脚本的兼容性问题仍存在，未在本次修复。
