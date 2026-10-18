# WORKER_REPORT — 数据搬运 / 容量 / 周期模型修正

分支：`codex/kwt-bus-budget`（worktree：`/home/yang/kwt-worktrees/bus-budget`，基点 = `codex/kwt-architecture`）
本提交只新增文件，未改任何 RTL、驱动或既有架构报告（仅按任务要求更新本 WORKER_REPORT）。

## 任务与结论

修正整个 FFN1 数据搬运核算的口径错误：既有报告写 `147,840 B`，把**未补零 A** 与
**补零 tile 的物理读/写**混在一起。物理一致的两种口径为
**有效 `148,480 B` / 填充 `153,600 B`**（`147,840 = 148,480 − 640`，缺 A 的 640 B 补零读）。

## 改动（仅新增）

- 新增 `hardware/optimization/kwt_bus_capacity_model.py`：可运行算术/容量/周期检查脚本。
- 新增 `hardware/optimization/kwt_bus_capacity_model.md`：修正口径、RTL 证据、容量与端口分析、同口径公式、证据边界。

## 实际执行的命令与退出结果

```bash
python3 hardware/optimization/kwt_bus_capacity_model.py \
  --json /tmp/kwt-bus-capacity.json --markdown /tmp/kwt-bus-capacity.md
# 退出码 0；自检 PASS，11 项断言

python3 hardware/optimization/run_hook.py bus_capacity_model --timeout 60 -- \
  python3 hardware/optimization/kwt_bus_capacity_model.py \
  --json /tmp/kwt-bus-capacity.json --markdown /tmp/kwt-bus-capacity.md
# PASS bus_capacity_model rc=0 elapsed=0.064s
# 状态文件 /tmp/kwt-optimization-hooks/bus_capacity_model.json：exit_code=0, timed_out=false
```

## 关键数值（均带证据层）

- 唯一数据量：有效 `148,480 B`、填充 `153,600 B`（L0；A padded 13,312、W 32,768、bias 1,024、
  有效 Y 101,376、补零 Y 106,496）。
- 逐 tile 重装：`3,648 B/tile × 208 = 758,784 B`（L1，来自 `kwt_tile_system_model.sv:70-72` 的
  A 512 + W 1024 + bias 32 + out 256 字）；16 行候选 `5,184 × 112 = 580,608 B`。
- 512 KiB 共享 SRAM：12×W16 + padded A + bias + 有效 Y + desc = `515,584 B ≤ 524,288 B`（余 8,704）；
  再加 12 层 bias 则超出 2,560 B。**能放下，但不能简单说“绝对放不下”**。
- 端口瓶颈：乒乓操作数存储每拍 2 字（1A+1B）、`compute_page=load_page=0` 致装载/计算串行；
  8×16 引擎 `7.11` 乘积/拍、16×16 `14.22` 乘积/拍。
- 同口径周期（8×16 / 16×16）：compute 212,992 / 114,688；dump 26,624 / 28,672；
  load@128-bit 47,424 / 36,288；测试台 staged 实测 251,328 / 154,240（L1）。

## 证据边界（本提交未证）

1. `758,784 B` 是 RTL 在**仿真 128-bit 总线**上的 payload（L1），不是赛题总线实测活动，字节数 ≠ 周期数。
2. 仅描述符取指另有 `6,656 B`（208×2 beat）未计入。
3. 当前 RTL 写回补零 Y 行，故现状是填充口径；有效口径依赖尚未实现的尾行屏蔽。
4. 官方 16-bit 宏的延迟/端口语义/字节使能/面积/功耗、赛题 System Bus 参数、150 MHz SDC 全部缺失，无法签核。
5. L1/L2 周期（251,328 / 154,240 / 838,028）来自测试台计时模型，不是平台吞吐。
