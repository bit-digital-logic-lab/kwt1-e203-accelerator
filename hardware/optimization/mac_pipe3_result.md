# MAC 计算路径实验：16 行流水乘法

`kwt_row_broadcast_tile_pipe3.sv` 是独立实验模块，基线 `kwt_row_broadcast_tile.sv` 未修改。输入 A 的高 8 位按有符号数、低 8 位按无符号数分别与有符号 16 位 B 相乘；第一级保存两个部分积，第二级将高部分积左移 8 位并相加。每拍仍可接收一组 16 行输入，乘积到累加存储体增加 1 拍延迟。此实现仅适用于 `DATA_W=16, ACC_W=32`。

验证使用独立整数参考值，四组各 1024 个输入周期：轮转列、同一列、两列交替，以及行掩码和输入气泡。输入含 `-32768`、`32767`、`-1` 等符号边界；所有 16×16 输出在排空流水后逐位比对，包含 bias。连续三组均接受 1024/1024 拍，间歇组接受 877/1024 拍。命令：

```bash
python3 hardware/optimization/run_hook.py --timeout 120 mac_pipe3_verify -- bash hardware/optimization/mac_verify.sh
```

上述命令的结果存于 `/tmp/kwt-optimization-hooks/mac_pipe3_verify.json`，完整日志存于同名 `.log`；最近一次结果为 PASS。

| ROWS=16，COLS=16 | 基线 | pipe3 | 变化 |
| --- | ---: | ---: | ---: |
| Yosys 通用门级 cells | 57,699 | 56,858 | -1.46% |
| Yosys `ltp -noff` 最长逻辑路径 | 33 级 | 27 级 | -18.2% |
| Nangate45 映射面积，平方微米 | 120,434.692 | 123,911.844 | +2.89% |

门级路径是 Yosys 逻辑级数，不是频率或时序签核。Nangate45 统计将部分和数组映射成触发器，不能解释为片上 SRAM 面积。测试未包含 A/B 读取、DMA、总线或输出阶段的吞吐。pipe3 尚未接入顶层，是否采用应取决于目标工艺时序与存储接口验证。
