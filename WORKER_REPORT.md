# WORKER_REPORT — codex/kwt-ds-streaming-mac（KWT-1 矩阵单元：是否等待全部数据 / 双 CACC 流式调度）

- 分支 / worktree：`codex/kwt-ds-streaming-mac` @ `/home/yang/kwt-worktrees/ds-p027-streaming`
- 范围：只在本 worktree 内操作；未开子智能体；未改 `hardware/rtl/**` 既有 RTL、未改任何既有 baseline TB；新增只落在
  `hardware/experiments/streaming_mac/` 与本报告。
- 交付物：该目录下 `README.md`、`results.json`、`run.sh`、两个 TB、`model/`，以及本报告。

## 1. 结论（先回答问题）

**MAC 核不等待全部数据。** 16 行广播核在理想供数下 **II=1**（[compute_only_pipeline.md](../../hardware/architecture/compute_only_pipeline.md)
已跑 1024 拍连续供数）。**加深乘法流水不会提高任务吞吐。** 真正“等”的是 **任务/累加器调度**：

- 旧 8×16 tile system（`hardware/rtl/control/kwt_tile_system_model.sv`）逐 tile
  `LOAD_A→LOAD_B→LOAD_BIAS→RUN_TILE→STORE_OUT` 完全串行（`compute_page/load_page` 固定为 0），装载/计算/写回不重叠；
- 16 行 staged 任务模型已把**输出写回**与下一 tile 计算重叠，但只有一份 CACC，仍付**串行 CACC 导出** 29 008 拍
  = 单层 FFN1 的 **18.8%**。

**改动（本实验）**：双 CACC（两页累加器）任务调度——第 *T* 个 tile 的 CACC 导出落在第 *T+1* 个 tile 的计算窗口内。
算术核与输出暂存 RTL **原封不动复用**。同核、同总线、同钟：

| 单层 FFN1（99×64 × 64×256 + bias，16 行） | baseline（单 CACC） | dual-CACC | 差值 |
| --- | ---: | ---: | ---: |
| 总拍数 | **154 240** | **125 603** | **−28 637（−18.6%），1.228×** |
| 初始装载 | 7 872 | 7 872 | — |
| 计算 | 117 040 | 117 040 | — |
| CACC 导出/排空 | 29 008 | 356（仅尾部） | 被计算掩盖 |
| stall | — | 0 | — |
| 共享字节 | 147 840 | 147 840 | — |
| INT32 逐位输出 | 25 344 | 25 344 | — |

baseline 与已提交的 `hardware/architecture/staged_task_row16_summary.json` 的 `full`=**154 240 完全一致**，证明比较是同一
时钟/总线口径。16 行纯 MAC 下界 114 688 拍：baseline 比算术下界高 34.5%，dual 高 9.5%。增益只在
`K·COLS > ROWS·COLS`（16 行即 K>16）时出现；tiny/ragged/two-tile/overflow（K≤8）增益≈0（有时多 1 拍），已如实记录。

## 2. 实跑命令与退出码（本 worktree，2026-10-19）

| 命令（仓库根执行） | 退出码 | 结果 |
| --- | ---: | --- |
| `bash hardware/experiments/streaming_mac/run.sh` | **0** | `PASS streaming-mac experiment: 5 cases, unit=PASS`；full baseline=154 240 / dual=125 603，25 344 输出 INT32 逐位一致 |
| `python3 hardware/experiments/streaming_mac/model/event_schedule.py --baseline-rtl 154240 --dual-rtl 125603` | **0** | **仅模型**：单层 dual 预测 125 824 vs RTL 125 603（误差 +0.18%）；12 层 1 509 888 拍 |

`run.sh` 一次编译 baseline（既有未改 TB）+ dual TB + dual-CACC 微测，跑 tiny/ragged/two-tile/overflow/full 五个 FFN1 形状，
逐 `(row,col)` 与独立 numpy INT32 参考比对，并核对访存计数；日志在 `hardware/experiments/streaming_mac/logs/`。

## 3. 测试覆盖

- **独立 INT32 参考**：`(a@w + bias) & 0xFFFFFFFF`，含 INT16 极值 / INT32 环绕（overflow）、ragged（k=7,n=19,m=10）、
  两 tile 连续（twotile 32×8×16）、完整 99×64×256。
- **复位/反压/有效序**：`tb_kwt_dualacc_unit.sv`（exit 0）证复位清空两页、导出与累加并行、运行中复位清空、dump_req 间隙
  反压不丢值与保序；任务 TB 另跑 `+BACKPRESSURE=4`（tiny）/ `=3`（ragged），均 INT32 正确（ragged 1/3 占空比产生 597 stall 拍）。

## 4. 字节与宏/端口

单层必需共享流量 = A 6 336 字 + W 16 384 字 + bias 512 槽 + 输出 50 688 槽 = **147 840 B**。内部缓冲：
A 常驻 12 672 B（2×`16b×4096`）；W 双页 4 096 B（2×`16b×1024`）；CACC 双页 2 048 B（寄存器，或 4×`16b×1024`）；
输出暂存 256×32=1 024 B（2×`16b×1024` 拼宽）。以上为**容量口径**，bank/端口仲裁与 150 MHz 闭合**未证**。
本实验为复用算术核，实例化了**两份** `kwt_row_broadcast_tile`（多 16 个 16×16 乘法器）；面积最优映射是只加一页 CACC
（寄存器 / 每 32 bit 两个字宏），不需要第二套乘法器。

## 5. 主要来源与可迁移性

- NVDLA unit description：<https://nvdla.org/hw/v1/ias/unit_description.html>
- NVDLA hardware architecture：<https://nvdla.org/hw/v1/hwarch.html>
- Gemmini README：<https://github.com/ucb-bar/gemmini/blob/master/README.md>
- Gemmini 论文（DAC'21）：<https://people.eecs.berkeley.edu/~alonamid/papers/gemmini-arxiv-1911.09925.pdf> ｜ 摘要 <https://arxiv.org/abs/1911.09925>

**可迁移（本次已 RTL 证明）**：NVDLA CACC 把 assembly SRAM（累加）与 delivery SRAM（导出）分开、用 valid-credit 反压
→ 正是本实验的“独立输出缓冲 / 双 CACC”；NVDLA ping-pong 寄存器编程与 Gemmini `LoopMatmul/LoopConv` 双缓冲 → W 页与
12 层换参双缓冲；Gemmini decoupled access/execute（load/store/execute 分离 + ROB）与 NVDLA CDMA→CBUF→CMAC→CACC 流水
→ 装载/写回与计算重叠；NVDLA multi-batch 用激活复用隐藏 FC 权重重载 → 本层 99 行天然共享一份 W，故 A 常驻 + W 广播正确。

**超出本项目端口/面积预算**：NVDLA CBUF = 512 KB（16×32 KB bank，2 写 3 读），CDMA 512-bit、burst≤4——与
`2.25 mm²` + `16b×128/1024/4096` 宏预算矛盾，且赛题未给宽总线；“更长 burst”只作用于装载段（7 872/154 240 = 5.1%），
不是吞吐杠杆。Gemmini 全脉动 + banked scratchpad/accumulator + TLB/ROB + 双 DMA 面积远超预算，只能借其**调度思想**。
NVDLA 的 Winograd/稀疏压缩对稠密 `64→256` FC 无意义且需额外端口。

## 6. 未证事项（不得宣称）

1. 非 E203 / System Bus / NICE 实测；装载与写回总线是**定时模型**。
2. 非 150 MHz STA、非面积/功耗签核；文中 ms 数为对仿真拍数的**条件换算**。
3. 内部 SRAM 宏映射、端口/bank 争用未落实。
4. 12 层整任务为**软件事件模型**，非 RTL；换层参数、总线仲裁未证。
5. 只测 16 行配置；增益来自调度，不是 MAC 流水深度或 Fmax 结论。
6. 研究生单行 FFN1 引擎（`graduate/rtl/encoder/kwt_ffn1_row.sv`）为逐行单乘法器，非吞吐引擎，本实验未改其数值。
