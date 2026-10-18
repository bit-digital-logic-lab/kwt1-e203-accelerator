# WORKER_REPORT

- 分支：`codex/kwt-preprocess`
- 任务：审计电脑端音频/特征到首层输入的 `98×40 → 99×64` 路径，以及各层 FFN1
  输入提取边界；补独立数值/变形校验；两条不同类别真实样本上机；CPU；不改 RTL
  与量化配置。

## 改动

新增：

- `software/audit_ffn1_input_boundary.py`：独立审计脚本。NumPy float64 自写
  FFT MFCC、`99×64` 投影、12 层前向，逐层捕获 `LayerNorm1(x + attn_combine)`；
  参考值从不取被测实现输出。含 E3 变形不变量（投影加性/齐次、class token、
  FFN1 token 置换等变）与错误边界候选判别（残差、注意力、层输入、LN2）。
- `software/test_ffn1_input_boundary.py`：unittest 包装，用共享 `.build` 缓存的
  两条不同类别真实样本做快速回归。
- `software/validation/ffn1_input_boundary_audit/README.md`：证据层级、命令、
  实测值、不可外推边界。
- `software/validation/ffn1_input_boundary_audit/manifest.json`：机读结果。

修改：

- `software/README.md`：在首层输入小节后补一段新审计的说明与运行命令。

未改动：任何 RTL、量化配置、`run.py`、`offline_encoder_input.py`、
`prepare_quant_samples.py`。共享只读输入：`/home/yang/Desktop/ic设计竞赛/资料/数据集/
speech_commands_v0.02.tar.gz`、`/home/yang/Desktop/ic设计竞赛/software/.build/
quant_samples_12labels_40.npz`。

## 实际执行的命令与退出结果

解释器：`/home/yang/miniconda3/bin/python`（Python 3.12.12，NumPy 2.5.2，
PyTorch 2.5.1+cu121；强制 CPU）。该 worktree 下 `python3` 无 PyTorch。

| 命令 | 退出码 | 结果 |
| --- | ---: | --- |
| `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/yang/miniconda3/bin/python software/audit_ffn1_input_boundary.py --sweep 60` | 0 | 2 条真实 WAV（`yes`、`four`→`_unknown_`）+ 60 条缓存验证样本全部通过；manifest 两次运行 SHA-256 均为 `8f6dcea3a168cd06cc448b54fe35e96c4e797c31940ca064995a479bde3df57c` |
| `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/yang/miniconda3/bin/python software/audit_ffn1_input_boundary.py --cache` | 0 | 缓存两条不同类别快速路径通过 |
| `cd software && /home/yang/miniconda3/bin/python -m unittest test_offline_encoder_input test_ffn1_input_boundary -v` | 0 | 5 tests OK（含原有 2 个测试无回归） |
| `/home/yang/miniconda3/bin/python -m py_compile software/audit_ffn1_input_boundary.py software/test_ffn1_input_boundary.py` | 0 | 编译通过 |

## 关键实测值（证据层级见下）

- 权重 `software/weights/kwt1_fp32.npz`：`sha256=8f359abf7adbe287b8f89fca98af46194f98f90f2146784731bff551a155825a`。
- 样本 WAV SHA-256：`yes/676f8138_nohash_0.wav` = `b71582f6…b0e8b45`，
  `four/750e3e75_nohash_0.wav` = `2ad981ba…8bcee9f`。
- `98×40 → 99×64` 独立 golden 最大绝对差：两样本 `1.32914507e-06`、
  `9.58424968e-07`；60 条抽样 `2.56496309e-06`。
- 12 层 FFN1 捕获最大绝对差：两样本 `3.39695828e-06`、`3.10831201e-06`；
  60 条抽样 `6.90650716e-06`。
- 最近错误边界距离：两样本 `0.332`、`0.345`；60 条抽样 `0.18463343`
  （与正确边界差约 5 个数量级，边界判别成立）。
- 形状：MFCC FP32 `98×40`，编码器输入 FP32 `99×64`，12 层 FFN1 均 FP32 `99×64`。
- 变形：投影加性 `2.95042992e-06`，齐次 `2.38418579e-07`，class token 行严格一致，
  FFN1 token 置换等变 `3.63495201e-06`。

## 证据层级

- **E2（独立重实现）**：上述所有编码器/FFN1 数值差。参考为 NumPy float64 独立
  实现，仅用冻结权重 NPZ 与真实 WAV/缓存特征，不取被测输出。
- **E3（变形不变量）**：加性、齐次、class token、置换等变。
- **E4（自洽/形状）**：形状、dtype、有限性、12 层互异。
- **E1（独立外部）**：本轮**未新增**。本机无 TensorFlow
  （`tensorflow_available=false`），未直接读原始 checkpoint，也未重建官方 Keras
  模型。既有 E1 见 `software/validation/real_input_chain_20261017/manifest.json`
  （官方 `SpeechFeatures` MFCC 最大绝对差 `6.91711903e-05`，TF checkpoint 投影
  `1.43051147e-06`），本报告不重复声称。

## 未证事项

- 未证明 RTL、定点/量化模型或 E203 SoC 联调正确；未改任何 RTL 或量化配置。
- 未用官方 Keras 模型/原始 checkpoint 做本轮交叉（缺 TensorFlow）。
- 抽样（2 条真实 + 60 条缓存）不是赛题官方 600 条评测，不能替代正式精度门槛。
- golden 为 float64、被测为 float32，记录的是最大绝对差，不是逐位一致。
- 未评估完整 12 层端到端精度、分类准确率或性能/面积。
