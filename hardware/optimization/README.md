# DMA / FSM / MAC 优化工作台

本目录记录三条独立工作线的命令和证据。RTL 修改分别限于
`hardware/rtl/dma/`、`hardware/rtl/control/`、`hardware/rtl/pulse_array/`；
整体验证在三个结果稳定后运行。代理之间只通过模块接口、文件和状态 JSON
交接，避免反复发送长日志。

## 静默回归钩子

```bash
python3 hardware/optimization/run_hook.py mac_continuous --timeout 120 -- \
  bash hardware/architecture/run_mac_continuous.sh
cat /tmp/kwt-optimization-hooks/mac_continuous.json
```

钩子把完整输出写入 `/tmp/kwt-optimization-hooks/<name>.log`，原子写入
`<name>.json`。成功只打印一行；失败打印末尾 20 行以便定位。每条工作线使用
自己的名称前缀，避免并行任务覆盖状态。钩子只是执行与汇报机制，不替代测试。

当前并发额度是含主代理在内的 4 个，因此先让 3 个 Sol 各自实现与测试；
待有空位后，由各 Sol 顺序启动自己的 Luna 子代理做独立复核。Luna 只读主实现，
跑已经写好的回归命令并检查日志；发现失败时再反馈给 Sol。这样不会把“看护”
误写成额外算力或未经测量的性能收益。

## 统一比较口径

| 指标 | 记录什么 | 可作的结论 |
| --- | --- | --- |
| 正确性 | 独立整数参考与 RTL 逐项比较，含边界输入和握手停顿 | 所测输入及所测接口模型的功能 |
| 吞吐 | 连续 `in_valid` 的接受间隔、有效输出个数、排空周期 | 计算核内部 II 与延迟 |
| 资源 | 同一综合脚本、参数及库下的乘法器、加法器、寄存器、存储器数 | 结构差异；无 SRAM 宏时非最终面积 |
| 逻辑深度 | 同一综合脚本下的关键组合路径层数 | 流水切分的方向性证据；非 150 MHz STA |
| 整层周期 | 同一抽象内存模型下的 208 tile 回归 | 功能和模型内的周期数；非比赛总线性能 |

最终集成必须检查 DMA `done/error`、FSM 请求/响应保持、MAC 清空与排空时序。
真实 SRAM 端口和总线时序未定，暂不把乒乓预取或总线并发作为已证实的收益。

完整 E203 整层回归是单进程连续执行；需要较快覆盖全部 tile 的数值时可用
`run_e203_chunks.sh` 分成 13 个独立的 16-tile 区间。分块检查覆盖每个输出，
但不检查区间边界连续提交，不能替代单进程整层回归。

## 当前架构研究的小实验

用户本轮要求只跑模块和分块，以下命令都不启动 E203 或整层仿真：

```bash
python3 hardware/optimization/run_small_architecture.py --jobs 3
```

这个入口并行运行 7 项检查，逐项通过 `run_hook.py` 留下日志和 JSON 状态；
一项失败即总体返回非零。也可单独运行其中一项：

```bash
python3 hardware/optimization/run_hook.py tiling_experiment --timeout 20 -- \
  python3 hardware/optimization/tiling_experiment.py \
  --out /tmp/kwt-optimization-hooks/tiling_results.json
python3 hardware/optimization/run_hook.py matrix_stride --timeout 20 -- \
  python3 hardware/optimization/matrix_stride_experiment.py
python3 hardware/optimization/run_hook.py kernel_collision --timeout 120 -- \
  bash hardware/optimization/kernel_collision.sh
python3 hardware/optimization/run_hook.py kernel_os --timeout 120 -- \
  bash hardware/optimization/kernel_os.sh
```

总体建议见 `architecture_decision_2026-10-17.md`；依据见
`tiling_findings.md`、`matrix_transport.md`、`kernel_tradeoff.md` 与
`research_dataflow.md`。前两项检查行主序分块、二维 stride 和独立黄金值；
后两项用真实 RTL 检查当前 CACC 冲突及输出驻留小核的连续发射。
