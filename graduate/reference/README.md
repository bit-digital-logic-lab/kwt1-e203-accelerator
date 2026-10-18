# 研究生组第 4–6 项独立 FP32 Golden 与中间张量捕获

范围：**研究生必做第 4–6 项**（单头注意力 / 单层 PostNorm 编码器 / 十二层编码器与分类头）
的**独立 FP32 数值参考**与**中间张量逐阶段记录**。本目录不改动 `software/` 既有实现，
也不把 PC 端 MFCC 计入加速器：golden 的输入边界是官方模型的**首层编码器输入 `X[99,64]`**。

## 文件

| 文件 | 作用 |
| --- | --- |
| `kwt1_fp32_golden.py` | 纯 NumPy、无 TensorFlow/scipy 依赖的 FP32 golden；逐步实现第 4–6 项并支持中间张量捕获 |
| `capture_intermediates.py` | 对真实验证划分样本逐层捕获形状/取值范围/SHA-256，写小 JSON（不落张量） |
| `compare_official_tf.py` | 与**官方 Arm KWT-1 `transformer_utils.py` 原文代码**（TF）逐阶段独立比对 |
| `compare_torch_reference.py` | 与既有独立 **PyTorch** 实现 `software/run.py` 二次独立比对 |
| `test_kwt1_fp32_golden.py` | 可运行单元测试（仅需 NumPy） |
| `results/*.json` | 上述运行的小结果摘要 |

## 官方语义核对（`flags.json` + 官方源码）

`资料/ARM_KWT-1_237e4097/models_data_v2_12_labels/kwt1/flags.json`：
`num_layers=12, d_model=64, mlp_dim=256, heads=1, attention_type=time,
spectrogram_length=98, dct_num_features=40, label_count=12`；`prenorm` 与
`approximate_gelu` **不在 flags.json 中**，取 `model_parameters` 默认值 **PostNorm、精确 GELU**
（`eval.sh` 亦未覆盖）。

依据官方 `transformer_utils.py`：

| 项 | 官方实现 | golden |
| --- | --- | --- |
| Q/K/V 投影 | `Dense(64)`，`use_bias=False` | `X@W`，无 bias |
| 注意力分数 | `q @ kᵀ`，形状 `99×99` | 同 |
| 缩放 | `/ sqrt(dim_key) = /8` | `/√64` |
| Softmax | `tf.nn.softmax(axis=-1)` 逐行 | 减行最大值后按行归一，行和 =1 |
| 加权 | `weights @ v` | 同 |
| 输出投影 | `combine_heads Dense(64)+bias` | 同 |
| 编码器（PostNorm） | `attn; out1=LN1(x+attn); ffn; out=LN2(out1+ffn)` | 同 |
| FFN | `Dense(64→256) → GELU → Dense(256→64)` | `99×256` 中间，精确 `0.5x(1+erf(x/√2))` |
| LayerNorm | Keras，总体方差，`eps=1e-6` | 同 |
| 分类头 | 类别 token `x[:,0]` → `Dense(12)` | 同（12 维 logits）|
| 12 层换参 | 12 个独立 `TransformerBlock`，逐层不同权值 | 逐层 SHA-256 互异，前 k 层与全量前 k 层一致 |

## 真实验证划分样本（仅评估，不调参）

来自 `software/validation/real_input_chain_20261017`，Speech Commands V2 官方 SHA-1
说话人划分的 **validation** 三分割，跨类别 3 条：`_unknown_`(four)、`no`、`yes`。
每条 `X[99,64]` 的 SHA-256 与 manifest 一致后，golden 推理的 top-1 分别为
`_unknown_ / no / yes`（与真值一致，仅作 sanity check，非精度结论）。

## 运行与实测退出码

```bash
# 1) 单元测试（系统 python3，仅 NumPy）
python3 graduate/reference/test_kwt1_fp32_golden.py          # 12 tests OK, exit 0

# 2) 中间张量捕获（真实验证划分 3 条）
python3 graduate/reference/capture_intermediates.py          # exit 0

# 3) 官方 TF 原文代码逐阶段独立比对（需 TF 解释器）
/home/yang/Desktop/ic设计竞赛/software/.venv/bin/python graduate/reference/compare_official_tf.py   # exit 0

# 4) PyTorch 独立实现二次比对
/home/yang/Desktop/ic设计竞赛/software/.venv/bin/python graduate/reference/compare_torch_reference.py  # exit 0
```

实测（3 条 validation 样本、12 层、逐阶段）：

| 比对 | 最大绝对误差 | 阈值 | 退出码 |
| --- | ---: | ---: | ---: |
| golden vs 官方 TF 原文代码 | `9.298e-06` | `1e-4` | 0 |
| golden vs PyTorch `run.py` | `4.768e-06` | `1e-4` | 0 |

官方 TF 比对单阶段最差：`softmax 8.8e-07`、`attended_v 1.8e-06`、
`attn_projected 2.1e-06`、`ln1 4.1e-06`、`ffn_hidden 4.4e-06`、`gelu 4.4e-06`、
`ffn_out 3.9e-06`、`layer_output 9.3e-06`、`logits 9.5e-07`。

## 已证 / 未证

**已证（本目录脚本，真实退出码 0）**

- 第 4–6 项操作顺序与官方 `transformer_utils.py` 一致：PostNorm、`QKᵀ/√64`、逐行 Softmax、
  加权 V、输出投影、两次残差 + 两次 LayerNorm、`64→256→64` 精确 GELU、类别 token 分类头。
- 12 层逐层换参：12 组参数 SHA-256 互异；golden 在 3 条**真实验证划分**样本上与官方 TF
  原文代码全 12 层逐阶段一致（≤ `9.3e-06`），并与独立 PyTorch 实现一致（≤ `4.8e-06`）。
- 中间张量（含 `99×99` 注意力、`99×256` FFN）的形状、取值范围、SHA-256 已记录
  （`results/golden_intermediates.json`），未落大张量。

**未证 / 边界**

- 官方层代码写于 TF 2.4 + `tensorflow_addons`；本机 TF 2.19 无 `tensorflow_addons`、无
  `keras.layers.experimental.preprocessing`，比对时装了**两个仅用于 import 的 shim**
  （`GELU→tf.nn.gelu(exact)`、`Rescaling→Rescaling`），二者均不在被执行路径上；
  这是**框架版本差异**，不是官方 2.4 位级运行。
- golden 输入取自既有真实数据管线缓存的 `X[99,64]`（该管线此前对 TF checkpoint 投影误差
  ≤`1.4e-06`）；**未**在本目录重跑官方 `speech_features` MFCC，也未把 MFCC 纳入加速器边界。
- 未测官方 600 条评测集准确率、未做量化/定点精度、未接 RTL/E203；本目录只提供 FP32 golden。
