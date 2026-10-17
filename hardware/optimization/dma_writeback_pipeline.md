# DMA 写回单口 SRAM 流水优化

## 改动

`kwt_dma.sv` 写回原先对每个 16 位字交替执行 `STORE_READ` 和 `STORE_CAPTURE`。现在 `STORE_READ` 发出首字地址，`STORE_CAPTURE` 收取同步 SRAM 上一拍的数据，同时发出同一 128 位总线拍内的下一字地址。拍边界及最后一字停止预读，下一拍仍由 `STORE_REQ` 的 `bus_valid && bus_ready` 握手决定。接口、尾拍 `bus_wstrb`、描述符检查和装载路径不变。优化仍只使用单口 SRAM 每拍一次读。

## 同条件验证

`tb_kwt_dma.sv` 的总线 ready 每隔一拍节流；相同测试台、相同外存模型下，改动前后的完成周期如下。周期从启动脉冲撤销后的下降沿计到 `done`，包含测试模型的总线等待。

| 搬运 | 原实现 | 当前实现 | 减少 |
| --- | ---: | ---: | ---: |
| 输出写回 10 字，含 2 字尾拍 | 24 | 14 | 10（41.7%） |
| 输出写回 16 字，两个满拍 | 36 | 20 | 16（44.4%） |
| 输出写回 1 字 | 3 | 3 | 0 |
| 装载 16 字 | 20 | 20 | 0 |

测试逐字检查写回结果，并检查 10 字和 1 字尾拍未覆盖的外存字节保持原值；三 SRAM bank 的装载、非法 bank、未对齐、零长度和越界描述符仍通过。`bash hardware/rtl/control/tb/run.sh` 的 DMA—控制集成测试通过；Verilator `--lint-only -Wall` 通过。

同一 Yosys 0.63 / Nangate45_typ 示例库综合：原实现 1615 个映射单元、面积 3406.396；当前实现 1651 个单元、面积 3466.246，增加 59.850（1.76%）。`ltp -noff` 最长拓扑路径均为 32 级，不能据此推断目标频率。此相对面积不包括 SRAM 宏和正式总线桥。

## 复跑

在仓库根目录运行：

```bash
python3 hardware/optimization/run_hook.py --timeout 120 dma_regression -- bash hardware/rtl/dma/tb/run.sh
python3 hardware/optimization/run_hook.py --timeout 120 dma_lint -- verilator --lint-only --timing -Wall -Wno-fatal -Ihardware/rtl/sram --top-module kwt_dma hardware/rtl/dma/kwt_dma.sv
python3 hardware/optimization/run_hook.py --timeout 120 dma_control_integration -- bash hardware/rtl/control/tb/run.sh
```

钩子仅输出 PASS/FAIL 摘要；完整日志及 JSON 状态保存在 `/tmp/kwt-optimization-hooks/dma_*.{log,json}`。上述周期只对应本仓库抽象 128 位总线和单口 SRAM 模型，不是组委会 System Bus、目标工艺时序或整层性能实测。
