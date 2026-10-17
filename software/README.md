# KWT-1 FP32 软件基线

按上一级赛题导航的指定提交、`kwt1/flags.json` 和 `best_weights` 构建。模型参数是 12 层、单头注意力、`d_model=64`、前馈 `64→256→64`；保持预训练权重不变。软件位于本目录，输入资料从上一级 `资料/` 读取。

## 已跑通的范围

- `export_weights.py` 从 TensorFlow checkpoint 提取全部 162 个模型张量，共 607,308 个 FP32 参数；过滤优化器状态，并校验张量清单。
- `run.py` 用 PyTorch 运行输入投影、class token 与位置编码、12 层 postnorm 编码器和 12 类分类头；支持 CUDA 和 CPU。
- `--input` 接受 `.npy` 格式的有限 FP32 矩阵：`(98,40)` 是离线 MFCC 特征，`(99,64)` 是赛题规定的编码器输入。后者直接进入第一层，不再重复拼接 token 和位置编码。

### 单独导出电脑端首层编码器输入

```bash
python3 software/offline_encoder_input.py --input mfcc.npy --output encoder_input.npy
python3 software/test_offline_encoder_input.py -v
```

输入 `mfcc.npy` 必须是一条样本的有限 FP32 `98×40` MFCC（可由 `prepare_quant_samples.py` 的语音前处理得到）。脚本读取指定 KWT-1 checkpoint 导出的 FP32 权重，逐帧做 `98×40 @ 40×64 + bias → 98×64`，在第 0 行前置 class token，再加 `99×64` 位置编码，输出 FP32 `99×64` `.npy`。此配置没有 distillation token。测试对照现有 PyTorch 模型并检查 class token 和首帧顺序。

这个文件是**首层编码器输入**。本科必做的硬件输入是 12 个编码器层各自完成注意力、残差和第一处 LayerNorm 后的 `99×64` 张量；不能直接把首层编码器输入当成各层 FFN1 输入。逐层张量须通过完整 FP32 前向提取（`run.py` 的 `forward(..., capture_ffn=True)`）。音频到 MFCC 的生成也在本脚本范围之外。
- `verify_reference.py` 用 TensorFlow 运算交叉检查同一权重和输入的前馈输入、最终编码器输出及 logits。

## 运行

在本目录执行：

```bash
python3 run.py --smoke --device cuda --output .build/smoke.npz
python3 run.py --input path/to/encoder_input.npy --device cuda --output .build/result.npz
```

`--smoke` 使用固定随机 `98×40` 特征，只检查模型和设备能运行，输出标签不代表实际语音识别结果。`result.npz` 保存 `logits`（12 项）和 `encoder`（`99×64`）。也可用 `--device cpu`。

已导出的 `weights/kwt1_fp32.npz` 可供直接运行。本机现有 Python 环境提供 PyTorch CUDA。需要重新导出和验证时，在本目录执行：

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install tensorflow-cpu==2.19.1
.venv/bin/python export_weights.py
.venv/bin/python verify_reference.py
```

TensorFlow 仅用于读取旧 checkpoint 和交叉检查；GPU 前向由本机已有的 PyTorch CUDA 执行。原仓库注明用 TensorFlow 2.4.0rc1 测试，此处没有声称与原版音频预处理和完整评测逐位一致。目前没有对语音测试集做准确率评估，也没有把 FP32 软件结果当作 RTL 验证结果。

## 本科组第一前馈层量化试验

`prepare_quant_samples.py` 从上一级的 Speech Commands V2 压缩包直接抽样，按官方说话人哈希规则分为训练和验证，并根据指定 `flags.json` 的 16 kHz、480 点 Hann 窗、160 点步长、512 点 DFT、80 个 mel bin、40 个 DCT 系数生成 `98×40` 特征。选取每个目标词 10 条训练和 10 条验证语音，共 200 条；不解压整个数据集。

```bash
OPENBLAS_NUM_THREADS=4 python3 prepare_quant_samples.py --per-word 10 --output .build/quant_samples_10.npz
OPENBLAS_NUM_THREADS=4 python3 quantize_ffn1.py --samples .build/quant_samples_10.npz --device cuda
```

量化目标是 12 层各自的第一前馈线性计算：`99×64 @ 64×256 + bias → 99×256`。输入由 FP32 KWT-1 模型生成，正式量化比较使用训练样本标定、验证样本评估。五种整体张量方案分别为跨层全局最大值、逐层最大值、逐层 99.9 分位值、逐层 2 的幂尺度和逐条输入动态最大值；分别测试 INT8、INT16。整数输入与整数权重实际做整数矩阵乘法，在累加器尺度加入整数 bias，最终反量化后比较 FP32 输出。详细数据见 [量化试验报告](results/README.md) 和 `results/ffn1_ptq.csv`。

## 分组、混合位宽与 FP8

沿用同一批训练/验证样本执行：

```bash
OPENBLAS_NUM_THREADS=4 python3 quantize_parts_mixed_fp8.py --samples .build/quant_samples_10.npz --device cuda
```

脚本比较每 16 个输出通道一组、每输出通道一个尺度、输入维每 16 项一组，以及按层或输出通道选择 INT8/INT16 的混合精度。另测试 FP8 E4M3FN、E5M2 的原始值、逐层尺度和输出通道分组尺度。分组整数方案使用实际整数矩阵乘法；输入维分块在四个整数部分积之后用 FP32 合并。FP8 使用 PyTorch 的 FP8 编码/解码操作数和 FP32 乘加，不代表本机有原生 FP8 GEMM。结果和硬件代价见 [分组与混合量化报告](results/分组混合FP8报告.md)。

## 0.3% 精度目标与硬件取舍

另用 240 条训练和 240 条验证样本（10 个目标词、unknown、silence），测试 8–12 位激活/权重、逐层静态或逐行仿射输入量化，以及无梯度的权重尺度选择。实际整数矩阵乘法的结果与存储、控制代价见 [0.3% 位宽与硬件取舍](results/0.3百分比位宽与硬件取舍.md)。静态对称 A12W10 的最差层 NRMSE 为 0.14739%，12 层 FFN1 权重紧密存储为 240 KiB，当前选作高精度方案；完整 240 条验证集已在 Verilator 逐位复核。参数生成和误差边界见 [A12W10 量化记录](results/ffn1_a12w10_static_sym.md)。

## 无梯度黑箱标定

[`calibrate_blackbox_ffn1.py`](calibrate_blackbox_ffn1.py) 把 12 层 FFN1 整数乘加、INT32 偏置及输出解码作为只返回误差的黑箱。默认从训练集最大值建立 A11W10 静态对称尺度，然后对每层的激活尺度和 256 个权重列尺度的共同倍率各做一轮离散坐标搜索；优化目标只用 240 条训练样本的逐层 NRMSE。冻结尺度后在另 240 条验证样本上各评估一次，汇总最差层。测试划分被输入检查拒绝；不使用梯度、重训练或预训练权重修正。

在仓库根目录运行：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python3 software/calibrate_blackbox_ffn1.py \
  --samples software/.build/quant_samples_12labels_20.npz \
  --output software/results/ffn1_blackbox_a11w10
```

输出 CSV 包含逐层训练基线、选中训练误差、验证误差与累加器峰值；JSON 记录输入 SHA-256、样本路径、候选查询记录和冻结配置；NPZ 提供 12 层 INT16 容器权重、INT32 偏置及解码尺度。本轮验证最差第 11 层 **0.18216%**，低于赛题的 **3%** 门槛；五档倍率搜索均选择 1.0，说明此粗粒度搜索未改进原 A11W10 最大值尺度。NPZ 的 10 位权重尚未按 10 位紧密打包；此软件误差不证明 RTL 逐位一致、官方 600 样本评测或 150 MHz 综合。

## A11W10 输出尺度紧凑编码

[`evaluate_scale_encoding.py`](evaluate_scale_encoding.py) 将上述冻结模型的每通道输出尺度改存为 16 位 Q15 尾数和 8 位指数，并在相同验证集上重跑整数乘加。最差层从 0.18216306% 变为 0.18216464%；紧密存储的尺度元数据从 12 KiB 降到 9 KiB。范围、容量前提与逐位验证边界见 [试验报告](results/A11W10输出尺度Q15编码试验.md)。

## 多方案无梯度尺度标定重跑

[`compare_ffn1_calibration.py`](compare_ffn1_calibration.py) 在相同训练/验证缓存上重跑 A8W8、A11W10、A12W9、A12W10 的逐层最大值基线，并仅用分层选出的 60 条训练样本为每层比较七组静态尺度倍率。结果和规则边界见 [重跑报告](results/多方案无梯度尺度标定重跑.md)。
