# 240 条验证样本的 Verilator 精度复测（2026-10-17）

运行的是本科必做第一线性层 `99×64 × 64×256 + bias`，使用冻结的
A11W10 参数和本地 240 条验证样本。每条样本的 12 层输入来自 FP32 KWT-1
前向模型；每层先在 Verilator 中计算有符号 INT16 乘积、INT32 累加与
整数偏置，再逐输出对比独立 NumPy INT64 GEMM 的低 32 位。

```bash
python3 hardware/architecture/run_quantized_ffn1_batch_rtl.py \
  --per-label 0 --rows 16 --staged --out /tmp/kwt-ffn1-all240-rtl
```

结果：**240×12 个任务、72,990,720 个输出全部逐位一致**。按每层全部
240 条样本的平方误差汇总，再取 12 层最大值，最差为第 11 层
**0.18216320% NRMSE**，小于 3% 门槛。和原软件标定报告的
0.18216321% 仅相差 1.46×10⁻⁸ 个百分点；各层最大差异不超过
2.61×10⁻⁸ 个百分点。逐层数据、参数及数据 SHA-256 见
[`quantized_validation240_12layers_rtl_summary.json`](quantized_validation240_12layers_rtl_summary.json)。

输出级硬件为 `kwt_row_broadcast_tile` 加 RTL 输出暂存；A/W/bias 装载和
共享总线仍是测试台的定时模型。记录的 444,211,200 拍是 2,880 次独立
任务仿真的周期和，**不能**作为真实 E203 SoC 的吞吐量。验证集来自本地
训练/验证分割，并非组委会冻结后选取的 600 条官方样本；该结果证明当前
量化参数和计算核在所选 240 条样本上的数值一致性，不证明正式赛题通过。
