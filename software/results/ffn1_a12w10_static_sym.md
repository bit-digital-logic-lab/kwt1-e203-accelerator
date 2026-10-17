# A12W10 静态对称量化及 Verilator 验证

## 选择理由

在当前 240 条训练/240 条验证样本的对照中，A12W10 静态对称方案达到
**0.14739% 最差层 NRMSE**，优于 A11W10 的 0.18216%。两者权重都是
10 位，紧密存储容量同为 240 KiB；A12W10 的激活多 1 位。它比已验证的
A11W10 精度更高，且保守累加范围仍可使用 INT32，因此本轮选它作为高精度候选。

## 编码规则

- 激活：每层一个训练集最大绝对值尺度，12 位有符号静态对称，零点为 0，范围 `[-2047,2047]`。
- 权重：每个输出通道独立最大绝对值尺度，10 位有符号对称，范围 `[-511,511]`。
- bias：按 `activation_scale × weight_scale` 量化成 INT32。
- 舍入：nearest, ties-to-even；整数乘加后才做浮点解码计算误差。
- 参数只用训练分区的 240 条样本标定；验证集未参与尺度确定。

每层上界按 `64×2047×511 + max(abs(qbias))` 保守估算，最大为
67,826,716，小于 INT32 正上限。实际验证样本中观测到的最大累加值为
4,912,389。

参数由下面命令生成：

```bash
python3 software/export_ffn1_quantized.py \
  --activation-bits 12 --weight-bits 10 --device cpu
```

该文件把 W10 权重放在 INT16 容器中以供当前 RTL 测试台读取。240 KiB
是紧密 10 位打包后的理论权重容量；物理 10 位打包/解包逻辑还未实现。

## Verilator 精度结果

对全部 240 条验证样本的 12 层输入各跑一次 RTL，共 72,990,720 个输出。
全部输出都与独立 INT32 整数 GEMM 逐位相等。按赛题口径逐层汇总后，最差
是第 11 层，NRMSE **0.14739360%**；软件 A12W10 baseline 为
0.14739372%，差异约 `1.24×10⁻⁷` 个百分点。结果及源文件哈希见
[Verilator 逐层摘要](../../hardware/architecture/quantized_validation240_a12w10_rtl_summary.json)。

复跑 RTL：

```bash
python3 hardware/architecture/run_quantized_ffn1_batch_rtl.py \
  --scheme A12W10 --activation-bits 12 --per-label 0 \
  --rows 16 --staged --out /tmp/kwt-ffn1-a12w10-rtl
```

当前证据覆盖计算核和 RTL 输出暂存；输入/权重装载与共享总线由测试台建模。
它不是正式 600 条评测，也没有验证 10 位物理存储格式、真实 DMA/E203、SRAM
宏时序、面积或功耗。
