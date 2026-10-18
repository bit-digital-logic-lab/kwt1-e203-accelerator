# WORKER_REPORT（分支 codex/kwt-quant-calibration）

## 任务

在冻结的 **A12W10** 硬件容器与既有舍入/饱和规则下，只在新目录
`software/quant_experiments/` 做**至少 3 种训练划分标定方法**的可运行对照，
验证划分只评估不调尺度，绝不重训/改权重/改 RTL/改基线。指标统一为
12 层最差 NRMSE、输出最大误差、累加溢出范围、样本数、标签覆盖与容量。

## 改动

- 新增 `software/quant_experiments/calibrate_ffn1_methods.py`：A12W10 下三种无梯度标定对照。
  复用既有 `base_scales`/`encode_params`/`quantize_symmetric`（`calibrate_blackbox_ffn1.py`）
  与 `collect_activations`（`quantize_ffn1.py`），容器/舍入/饱和规则与已验证 A12W10 完全一致；
  脚本断言只接受 training/validation 划分、训练验证路径不重叠、位宽恒为 A12W10。
- 新增 `software/quant_experiments/README.md`：方法定义、合规边界、结果表、分位扫描、容量与未证事项。
- 新增小结果 `software/quant_experiments/ffn1_a12w10_methods.csv`（逐层 3 方法）与
  `.json`（元数据、容量、逐层记录、选择轨迹、分位事后扫描、样本/权重 SHA-256）。
- 更新 `software/results/README.md` 增加一节索引指向新对照。
- 覆盖本文件 `WORKER_REPORT.md`。
- 未改任何 RTL、驱动、综合、权重或主目录/其他分支文件；样本缓存写在 gitignore 的
  `software/.build/`，不进入提交。

## 三种标定方法

1. `max`：逐层激活绝对最大值 + 逐输出通道权重绝对最大值（baseline）。
2. `percentile`：逐层/逐通道幅值分位（候选 99.0/99.5/99.9/99.99），按训练子集 NRMSE 选分位，再量化并 clamp 截断。
3. `err_min`：以 `max` 尺度为基准，在训练子集上做无梯度坐标搜索 `(激活倍率, 权重倍率)` 最小化训练 NRMSE。

**梯度法主动排除**：题目禁止重训/微调/蒸馏，梯度类量化边界不清，故 `gradient_used=false`，只用训练集统计量与无梯度搜索。

## 实际执行的命令与退出结果

```bash
# 1) 从主目录只读复制既有 480/480 样本缓存到本分支（gitignore）
cp "/home/yang/Desktop/ic设计竞赛/software/.build/quant_samples_12labels_40.npz" software/.build/
sha256sum software/.build/quant_samples_12labels_40.npz
# -> 43d639fb64ea8d907db967b071010e41e68827368532880ba8a99b92769bbae8（与主目录一致）

# 2) 冒烟（2 层，退出码 0）
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  "/home/yang/Desktop/ic设计竞赛/software/.venv/bin/python" \
  software/quant_experiments/calibrate_ffn1_methods.py --device cuda --max-layers 2 \
  --output /tmp/opencode/ffn1_smoke

# 3) 完整对照（12 层，退出码 0，real 3m19s）
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  "/home/yang/Desktop/ic设计竞赛/software/.venv/bin/python" \
  software/quant_experiments/calibrate_ffn1_methods.py --device cuda
```

命令 3 输出（退出码 0）：

```
layer=12 done
max         worst_val_nrmse=0.147047% layer=11 max_abs_err=1.1931e-01 peak=4910460 headroom=3.16%
percentile  worst_val_nrmse=0.213332% layer=11 max_abs_err=3.8751e-01 peak=5781780 headroom=3.16%
err_min     worst_val_nrmse=0.147047% layer=11 max_abs_err=1.1931e-01 peak=5276333 headroom=3.16%
PASS wrote .../ffn1_a12w10_methods.csv and .../ffn1_a12w10_methods.json
```

数据：训练 480 / 验证 480，12 类各 40 条，训练验证路径交集 0；标定选择子集每类 5 条共 60 条，仅用于方法内选择；验证划分只评估。

## 结论（证据层级 L1：本轮软件整数路径实测）

- **`max` 并列最优**：最差层 NRMSE **0.147047%**（第 11 层），与既有 480/480 A12W10 对照报告值一致（控制点复核通过）。
- **`err_min` 无收益**：总体最差层与 `max` 相同（第 11 层选回 `(1.0,1.0)`）；仅第 2/3/5/7 层有 ≤0.0022 个百分点的微小改善，不改决定成绩的最差层。
- **`percentile` 明显变差**：最差层 **0.213332%**；固定分位扫描单调恶化（p99.99=0.213332%、p99.9=0.993163%、p99.5=3.924070%、p99.0=6.552544%），说明 A12W10 下尾部截断得不偿失。
- **输出最大误差**：`max`/`err_min` 为 **0.119312**，`percentile` 为 0.387509；峰值均出现在第 1 层。
- **无溢出**：三方法保守上界约 `6.78×10⁷`，仅占 INT32 的 3.16%；观测峰值最大 `5.78×10⁶`，均远小于 INT32。
- **容量不变**：A12W10 W10 紧密权重 240 KiB / 16 位容器 384 KiB / 权重尺度 24,576 B / INT32 bias 12 KiB，与标定方法无关。
- **不建议为标定引入分位或坐标搜索**，维持最大值静态尺度即可。

## 未证事项与证据边界

- 仅软件整数路径 + FP64 解码误差；**未重跑 RTL / E203 全仿真**，未改基线/RTL/权重。
- 未读测试划分，未测组委会隐藏 600 条评测；结论限于本 480/480 缓存。
- 本工作树无 Speech Commands V2 压缩包；样本缓存为只读复制并记录 SHA-256，完全独立复现预处理需在有数据集的机器上运行 `prepare_quant_samples.py`。
- `err_min` 只搜 7×7 固定倍率栅格；更细栅格未试，但第 11 层训练子集已选回 `(1.0,1.0)`。
- 容量为理论紧密打包值，未实现 W10 打包/解包。
