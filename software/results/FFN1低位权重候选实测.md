# FFN1 低位权重候选实测

## 方法

继续使用每类 40 条训练、40 条验证的 Speech Commands V2 缓存，12 类合计各 480 条；只量化本科必做的 12 层第一前馈线性运算 `99×64 @ 64×256 + bias`，其他模型计算仍为冻结 FP32 离线参考。尺度只取自训练集，七组预设倍率仅用每类 10 条、共 120 条训练样本选择；验证集用于冻结后的逐层汇总 NRMSE。没有梯度、重训或测试集调参。规则及完整数值协议见 [扩大样本报告](FFN1扩大样本静态量化取舍.md)。

## 480 条验证样本、12 层结果

| 方案 | 标定 | 最差层 NRMSE | 层 | 12 层 W 紧密容量 | 观测累加峰值 | 保守 INT32 上界 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A8W8 | 最大值基线 | 1.121379% | 11 | 192 KiB | 75,731 | 1,045,138 |
| A8W8 | 训练选倍率 | 1.046324% | 11 | 192 KiB | 91,514 | 1,048,359 |
| A10W8 | 最大值基线 | 0.567487% | 12 | 192 KiB | 304,702 | 4,205,241 |
| A10W8 | 训练选倍率 | 0.569080% | 12 | 192 KiB | 334,958 | 4,211,000 |
| A12W8 | 最大值基线 | **0.518930%** | 12 | 192 KiB | 1,220,226 | 16,845,652 |
| A12W8 | 训练选倍率 | 0.518930% | 12 | 192 KiB | 1,309,423 | 16,866,944 |
| A10W9 | 最大值基线 | 0.353190% | 11 | 216 KiB | 612,083 | 8,443,594 |
| A10W9 | 训练选倍率 | 0.353190% | 11 | 216 KiB | 672,278 | 8,455,158 |

这些候选都低于本科 3% 门槛；A12W8 是 W8 中误差最小者，适合作为低权重容量候选。A10W8 的训练选倍率在验证集最差层反而比基线高约 0.001593 个百分点，故优先保留静态最大值基线。A10W9 比 A12W8 误差更低，但 W9 需要额外 24 KiB 的 12 层紧密权重容量。此前 A12W10 基线为 0.147048%，精度仍明显更高；A12W8 的取舍是潜在存储收益，而非当前核上的最高精度。

所有上界按 `64×qmaxA×qmaxW + max(abs(INT32 bias))` 逐层计算并断言不超过 INT32 正上限；观测峰值来自这 480 条验证样本。前者是该冻结参数包在所述量化范围内的安全证明，后者不是最坏情况。

## 硬件意义与参数包

当前 RTL 接口对 A/W 各用一个 16 位字，对 bias/累加/输出用 INT32。因此 A8W8、A10W8、A12W8、A10W9 与已有 A12W10 **每层容器搬运仍是 147,840 B**；所有 W8 的 12 层权重在未打包状态下仍占 384 KiB。若以后实现 W8 双值打包，12 层权重可从 W10 的紧密理论 240 KiB 降到 192 KiB，差 48 KiB；但需要解包器、地址排列和总线/宏端口测量。当前没有这项硬件收益证据。

已把 **A12W8 最大值基线**导出为 [`ffn1_a12w8_static_sym_12labels_40.npz`](ffn1_a12w8_static_sym_12labels_40.npz)；[JSON 元数据](ffn1_a12w8_static_sym_12labels_40.json)含样本和参数 SHA-256、逐层尺度与 INT32 上界。NPZ 的 `layer01_weight` 等键与现有 RTL 测试台输入格式一致，权重仍存 INT16 容器。随后以每类 1 条、共 12 条验证样本完成 12 层 RTL 复测：A12W8 的 3,649,536 个 INT32 输出逐位一致，子集最差层 NRMSE 为 0.517576%；A8W8 的 8 位物理核也通过同规模复测。详见[硬件成本与周期对照](../../hardware/architecture/量化格式_硬件成本与周期对照_20261017.md)。

## 复跑

仓库根目录执行：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 software/compare_ffn1_calibration.py \
  --samples software/.build/quant_samples_12labels_40.npz \
  --train-select-per-class 10 --schemes A8W8 A10W8 A12W8 A10W9 \
  --output software/results/ffn1_w8_w9_candidates_12labels_40 --device cpu
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 software/export_ffn1_quantized.py \
  --samples software/.build/quant_samples_12labels_40.npz \
  --activation-bits 12 --weight-bits 8 \
  --output software/results/ffn1_a12w8_static_sym_12labels_40 --device cpu
```

逐层与七组训练候选记录在 `ffn1_w8_w9_candidates_12labels_40.csv/.json`；NPZ 保存候选选出的整数参数。此报告推荐的 A12W8 最大值参数使用独立导出包，避免把训练选倍率文件误作最大值基线。
