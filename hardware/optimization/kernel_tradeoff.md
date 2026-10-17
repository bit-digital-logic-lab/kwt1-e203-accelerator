# 计算核小仿真：真正的发射间隔由部分和端口决定

本轮只仿真 `2×3` 乘积阵列/CACC、`2×3` 输出驻留小核和现有 16 行广播核；
没有跑整层或整芯片。`run_hook.py` 的状态和完整日志位于
`/tmp/kwt-optimization-hooks/kernel_*.{json,log}`。

## 发现

当前 `kwt_product_array` 对一次 `launch`，同一行的不同列乘积依次出阵列。
`kwt_cacc` 每行只有一个读改写通道，若下一波紧接着发射，上一波的列 1 与
下一波的列 0 会在同拍到达该行。`kernel_collision_tb.sv` 的真实 RTL 结果：

- `2×3` 发射间隔为 3 拍时，所有 6 个 INT32 输出逐项正确，碰撞标志为 0。
- 连续两拍发射时，碰撞标志为 1；`row0,col0` 得到 3，独立期望为 51。

因此现有单通道行 CACC 要保持精确，`COLS=16` 时发射间隔至少要
16 拍。现有 tile 控制器每个 K 向量串行读取 16 列 B，随后捕获/发射；
它没有发挥 128 个乘法器的每拍接收潜力。仅给乘法器增加流水寄存器无法
解除这个部分和端口约束。

`kernel_os_accum.sv` 则给每个输出一个 32-bit 寄存器与加法器。与相同
乘积阵列连接的 `2×3` 小核连续发射 16 拍，6 个结果与独立点积逐项相等。
Yosys 结构统计将其扩为 8×16：输出驻留原型有 128 个 `$add`、128 个
`$adffe`，现有行 CACC 有 8 个 `$add`、8 个 `$mem_v2`。这只是局部结构，
不含乘法器、供数网络或存储宏面积，不能按单元数判断最终面积。

| 路线 | MAC 发射/部分和限制 | 供数条件 | 初步用途 |
| --- | --- | --- | --- |
| 现有 8×16 脉冲阵列 + 8 行 CACC | 至少隔 16 拍再发 K 向量；128 乘法器平均最多约 8 个有用 MAC/拍 | 每次发射的 8 A + 16 B 可在间隔中串行预取 | 保留作低风险功能基线 |
| 8×16 输出驻留 | 局部小核证实连续发射；需 128 条 32-bit 累加状态/加法器 | 稳态每拍 8 A + 16 B 或等价宽缓存/移位供给 | 只有端口和目标面积支持时才有价值 |
| 16 行广播 + 行 CACC | 原核小仿真连续接受一组 16 行和一个 B/拍；16 个加法通道 | 16 个 A 值在寄存器中保留 16 列拍，下一 K 的 A 可从另一寄存器页逐拍预取；B 每拍 1 字 | 当前单口 A/B 宏条件下优先做 tile 级供数原型 |

“约 8 个有用 MAC/拍”是 `8×16/16` 的计算核上界，未包含装载、排空和
输出导出。行广播并不等于 Eyeriss 卷积的 row-stationary 数据流；名称只描述
本项目中一个 B 标量向多行广播。

## 复跑

```bash
python3 hardware/optimization/run_hook.py kernel_collision --timeout 120 -- \
  bash hardware/optimization/kernel_collision.sh
python3 hardware/optimization/run_hook.py kernel_os --timeout 120 -- \
  bash hardware/optimization/kernel_os.sh
python3 hardware/optimization/run_hook.py kernel_structural --timeout 30 -- \
  bash hardware/optimization/kernel_structural.sh
python3 hardware/optimization/run_hook.py kernel_row16 --timeout 90 -- \
  bash hardware/architecture/run_mac_continuous.sh
```

需要下一步验证的是**供数**：A 的两组 16×16-bit 寄存器、B 的单字/拍读取、
同步 SRAM 延迟、切换 K 时无气泡或可计数气泡、部分和 dump 与下一 tile
是否相互阻塞。上述小核验证不证明这些系统条件。
