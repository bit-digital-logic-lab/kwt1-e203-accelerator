# 本科组 FFN1 整体量化初测

## 范围与数据

- 指定 KWT-1 checkpoint：12 层、每层 `99×64 × 64×256 + bias → 99×256`。只改第一前馈线性计算的数值表示；其输入由未量化的完整 FP32 模型计算。没有重训练或修改权重。
- 从官方 Speech Commands V2 压缩包抽取 10 个目标词，每词 10 条训练样本用于标定、10 条验证样本用于比较，共 200 条，训练与验证的文件及说话人哈希分离。文件名完整记录在 `ffn1_ptq.json`。测试划分没有用于正式结果或尺度选择。
- 特征按指定 `flags.json` 和官方 `speech_features`、`windowing`、`magnitude_rdft_mel`、`mel_table`、`dct` 的非流式推理公式生成。前处理为 1 秒 16 kHz PCM、`98×40` MFCC，再由 FP32 模型形成每层 FFN1 的 `99×64` 输入。

## 数值规则

签名对称量化范围为 INT8 `[-127,127]`、INT16 `[-32767,32767]`；舍入为 nearest ties-to-even。每层的 256 个输出通道共用一个权重尺度，每个 `99×64` 输入张量共用一个输入尺度。`global_max` 跨 12 层共用输入尺度和权重尺度；`layer_max` 逐层静态最大值；`layer_p999` 逐层静态 99.9 分位并截断；`layer_pow2` 将逐层最大值尺度上取整到 2 的幂；`dynamic_max` 每条输入现场计算最大值。静态输入尺度只用训练样本求得。

实际计算为 `qx = clip(round(x/sx))`、`qw = clip(round(w/sw))`、`qb = round(b/(sx·sw))`、`acc = int_matmul(qx,qw)+qb`、`yq = float(acc)·sx·sw`。INT8 用 int32 累加，INT16 用 int64 累加；没有把操作数反量化后再做浮点乘加。最终的浮点反量化只用于计算误差。

每层误差按所有 100 条验证语音的 `99×256` 输出汇总：`NRMSE = sqrt(Σ(yq−yfp32)² / Σ yfp32²)`，最终取 12 层中的最大值。题目原图要求每层汇总全部评测样本后取最大层，但未给出归一化分母的具体公式；这里明确采用 FP32 输出的 RMS 作为分母。

## 100 条验证语音结果

| 位宽 | 方案 | 最大层 NRMSE | 最差层 | 初测 3% 门槛 |
| --- | --- | ---: | ---: | --- |
| INT8 | 跨层全局最大值 | 2.13919% | 7 | 低于 |
| INT8 | 逐层最大值 | 1.81822% | 8 | 低于 |
| INT8 | 逐层 99.9 分位 | 3.46252% | 6 | 超过 |
| INT8 | 逐层 2 的幂尺度 | 2.61823% | 7 | 低于 |
| INT8 | 每条输入动态最大值 | 1.77413% | 8 | 低于 |
| INT16 | 跨层全局最大值 | 0.01031% | 11 | 低于 |
| INT16 | 逐层最大值 | 0.03429% | 1 | 低于 |
| INT16 | 逐层 99.9 分位 | 3.21099% | 6 | 超过 |
| INT16 | 逐层 2 的幂尺度 | 0.01020% | 7 | 低于 |
| INT16 | 每条输入动态最大值 | 0.00694% | 8 | 低于 |

逐层 99.9 分位截断的最大输入截断比例为 0.118%，最大权重截断比例为 0.104%；它在 INT16 下仍超过 3%，说明截断误差主导。逐层最大值 INT8 比全局最大值降低最大层误差，但需要 12 组输入和权重尺度。动态最大值误差更低，代价是每条输入扫描最大值、运行时改尺度及 bias 编码。2 的幂尺度可用移位实现尺度变换，但 INT8 最大层误差增至 2.618%。

INT8 所有观测整数累加绝对值不超过 100,378；INT16 观测最大值为 6,666,676,192，超过 int32 范围，因此 INT16 方案不能直接使用 32 位累加器。上述是观测值；RTL 位宽还需按输入范围和累加长度给出最坏情况证明。

这个初测只覆盖 100 条验证语音，未覆盖组委会的 600 条评测样本，也未包含 RTL 的编码、舍入、饱和和逐位一致性验证。建议先保留 INT8 逐层最大值与 INT16 跨层全局最大值作为下一轮 RTL 对照候选；正式选择应在更广的训练/验证样本上固定数值规则后再评估。

开发过程中曾用少量测试划分音频试跑程序；发现赛题的测试集限制后，正式结果已全部改用训练标定和验证比较。那些已看过的测试音频不能再作为完全未触碰的最终保留集。

## 单 16 位容器精度余量与 INT32 边界

[`FFN1单容器精度余量与INT32边界.md`](FFN1单容器精度余量与INT32边界.md)（脚本
[`compare_ffn1_container.py`](../compare_ffn1_container.py)）在同一 480/480 缓存上补测三个新方案：
A12W12、A16W10、A14W12（A11W10/A12W10 作控制点，逐位复核既有报告）。结论是**权重位宽是瓶颈、
激活位宽在 12 位后饱和**：权重 10→12 位把最差层从 0.147047% 降到 0.070279%，激活 12→16 位只降到
0.133404%；同样约 50.5% 的 INT32 余量下，A14W12（0.036708%）比 A16W10（0.133404%）低 3.6 倍。
现有 16 位容器和 16×16 物理乘数都不需要更换，只有实现 W12 打包时权重才会从 240 KiB 增到 288 KiB。

## A12W10 标定方法对照

[`../quant_experiments/README.md`](../quant_experiments/README.md)（脚本
[`calibrate_ffn1_methods.py`](../quant_experiments/calibrate_ffn1_methods.py)）在 480/480 缓存上、冻结 A12W10 容器内比较
三种只用训练划分的无梯度标定：最大值 `max`、分位 `percentile`、误差最小化 `err_min`。结论是
`max` 并列最优（最差层 0.147047%，第 11 层，与既有 480/480 A12W10 逐位一致），误差最小化不改善最差层，
分位截断单调变差（p99.9 已 0.993163%，p99.5/p99 超过 3%）。未使用梯度、未重训、验证仅评估。

## 复现与来源

在 `software` 目录运行 README 中的两条命令。逐层误差、尺度、截断率、累加器峰值见 `ffn1_ptq.csv`；样本路径和规则见 `ffn1_ptq.json`。

官方实现参考：[KWT-1 Transformer](https://github.com/ARM-software/keyword-transformer/blob/237e4097afd4f6851e0b82f0a8d65c2dfe8eae30/kws_streaming/models/transformer_utils.py)、[MFCC 主流程](https://github.com/ARM-software/keyword-transformer/blob/237e4097afd4f6851e0b82f0a8d65c2dfe8eae30/kws_streaming/layers/speech_features.py)、[mel 矩阵](https://github.com/ARM-software/keyword-transformer/blob/237e4097afd4f6851e0b82f0a8d65c2dfe8eae30/kws_streaming/layers/mel_table.py)、[DCT](https://github.com/ARM-software/keyword-transformer/blob/237e4097afd4f6851e0b82f0a8d65c2dfe8eae30/kws_streaming/layers/dct.py)。
