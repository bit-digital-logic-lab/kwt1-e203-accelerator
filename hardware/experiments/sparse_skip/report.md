# KWT-1 行广播核零值门控实验

## 结论

当前 `kwt_row_broadcast_tile` 每拍广播一个 B，8 行各有一个乘法器和 CACC bank。实验版仅当 `in_valid && row_mask[r] && A[r]!=0 && B!=0` 时更新乘积寄存器和该行 CACC；稀疏位置不读改写部分和。输入仍按全部 64×16 个 `(K,C)` 时隙发送，地址和外部存储格式不变。因此这是**活动率优化，不是周期加速**。没有对模型做剪枝或近零舍入。

## 复现和实测

`bash hardware/experiments/sparse_skip/run.sh` 重新生成实验 RTL，编译 Verilator，运行基准种子。之后可执行 `hardware/experiments/sparse_skip/.build/Vtb_zero_gate +SEED=7`。`make_variant.py` 由现有 RTL 生成独立变体，原 RTL 不改。

Verilator 5.047：种子 12345、1、7、101、20261019 均 PASS；每种子覆盖 4 个 8×16 tile、K=64，总计 512 个输出，逐位比较原核、门控核和 testbench 中独立的有符号 INT16×INT16、INT32 回绕参考（含 bias）。场景含近乎全密、A/B 混合零值、全零、5/8 有效行；所有输出核对通过。每种子均为 5,148 仿真拍，原核 29,696 个有效乘积更新，门控核 9,731 个，减少 19,965（67.23%）；`sim.log`、`multi_seed.log` 保留实测输出。更新次数不等于实测功耗。

Yosys `read_verilog -sv; hierarchy; proc; opt; memory_collect; stat` 对 8 行默认配置的粗结构：原核 368 cells，门控核 394 cells；两者均为 8 `$mul`、9 `$add`、8 `$mem_v2`。增加的 26 个粗单元主要是零检测与使能组合逻辑。此统计既非标准单元面积，也非时序或能耗报告；日志分别为 `yosys_dense.log` 和 `yosys_gate.log`。

## 架构界限

现有核的 `in_col` 连续接收 16 个列拍，A 保持、B 一字/拍；CACC 仍是一行一个按列寻址的 1R1W bank，输出导出和系统总线时间均不变。要缩短拍数，必须在上游有非零索引/压缩编码和变长调度，允许不发送零 B 列及跳过所有行 A 为零的时隙，并处理控制、索引解码、DMA 对齐和存储带宽。逐行 A 稀疏只能关掉该行乘法/CACC，不能跳过整拍；按 K 全零块跳过还要保证权重和 A 的位置一致。当前项目缺正式 SRAM 宏、平台总线和完整控制路径，不能声称这里的活动率减幅可转化为功耗、频率或完整任务周期收益。
