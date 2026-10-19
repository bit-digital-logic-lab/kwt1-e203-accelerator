# 12 层冻结整数权重/偏置/尺度 manifest（独立 worker：layer_params）

本目录是该独立工作者的**唯一**产物目录（任务约定：只写 `graduate/ds_tasks/layer_params`）。
目标：为 KWT-1 编码器 12 层的 **FFN1 第一线性层**生成冻结的整数权重、整数偏置与量化尺度
manifest；**训练划分标定、验证划分只评估**；不修改官方模型权重；不重训、不用梯度、不用测试划分。

## 范围

- 只量化题目规定的 12 层首前馈线性：`99×64 @ 64×256 + bias(256) → 99×256`，每层 `64×256` 权重。
- 前处理、完整 FP32 模型（产生每层 `99×64` 输入）保持 FP32，作为离线参考——与题目数据边界一致。
- 注意力、LayerNorm、GELU LUT、FFN2、分类头不在本 manifest 范围（分别由
  `attn_scale`/`postnorm_scale`/`gelu_lut` 等任务负责）。
- 冻结方案：**A12W10**（激活 12 位，权重 10 位，有符号、零点 0）。该方案与研究生编码器
  RTL 现行冻结包 `software/results/ffn1_a12w10_static_sym.npz` 同族，便于对照。

## 文件

| 文件 | 作用 |
| --- | --- |
| `build_layer_manifest.py` | 生成 manifest：独立实现的量化 + 整数 GEMM + NRMSE，训练标定/验证评估 |
| `verify_manifest.py` | 独立门禁：从冻结产物重建并逐项断言（尺度来源、整数量化、INT32、验证 NRMSE） |
| `layer_manifest_A12W10.json` | 冻结 manifest（可读）：12 层尺度、量化规则、哈希、证据层级、硬件装载映射 |
| `layer_manifest_A12W10.npz` | 冻结整数张量：`layerNN_weight(int16)`、`layerNN_bias(int32)`、`layerNN_activation_scale(float64)`、`layerNN_weight_scales(float64×256)` |
| `logs/build.log` | 生成运行真实日志（退出码 0） |
| `logs/verify.log` | 独立门禁运行真实日志（退出码 0） |

## 数值规则（无梯度）

- 激活尺度：每层静态对称，`sx = max|训练激活| / 2047`，零点 0。
- 权重尺度：每输出通道对称，`sw[c] = max|W[:,c]| / 511`，零点 0，仅依赖冻结 FP32 权重。
- 偏置：`qb = round_ties_to_even(b / (sx·sw))`，有符号 INT32。
- 累加：有符号整数 GEMM + INT32 偏置，逐层核对保守上界 `K·qmax_a·qmax_w + max|qb| ≤ 2³¹−1`；
  仅在计算误差时做 FP64 反量化。
- **验证划分不参与任何尺度/阈值选择**；无动态逐样本尺度、无梯度、无测试划分。

## 复现

```bash
VENV=/home/yang/Desktop/ic设计竞赛/software/.venv/bin/python
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 $VENV graduate/ds_tasks/layer_params/build_layer_manifest.py --device auto
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 $VENV graduate/ds_tasks/layer_params/verify_manifest.py --device auto
```

两条命令退出码都是 **0**。`build` 复现既有已发布控制点：A11W10 最差 `0.181954%`、
A12W10 最差 `0.147047%`（均为第 11 层），权重/偏置整数编码与既有参考函数逐位相同。
`verify` 从冻结 npz 重建并全部通过（`VERIFY_PASS samples=480/480 worst=0.147047% layer=11`）。

## 证据层级

| 层级 | 状态 |
| --- | --- |
| L1 软件参考（整数路径） | **已证**（本任务；训练标定 / 验证评估，480/480） |
| L2 RTL 逐位 | 本任务**未产生**；既有 layer-0 链逐位证据见 `graduate/encoder/tb/run.sh`（本次未复跑） |
| L3 E203 SoC | **MISSING**（未在官方 E203/NICE 上运行） |
| L4 综合（150 MHz/面积/功耗） | **MISSING**（无目标 PDK/SDC） |
| L5 官方验收（600 条评测） | **MISSING**（未跑官方评测脚本） |
