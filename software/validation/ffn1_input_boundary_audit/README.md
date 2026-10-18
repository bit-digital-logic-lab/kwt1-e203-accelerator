# FFN1 输入边界与 98×40 → 99×64 独立审计（2026-10-18）

本目录由 [`software/audit_ffn1_input_boundary.py`](../../audit_ffn1_input_boundary.py) 生成，
审计两件事：

1. 电脑端 **音频/特征 → 首层编码器输入** 的 `98×40 → 99×64` 映射；
2. 赛题 **12 层 FFN1 输入提取边界**，即每层注意力、残差和第一处 LayerNorm
   之后的 `99×64` 张量（`run.py` 的 `forward(..., capture_ffn=True)`）。

被测对象是软件实现；参考值（golden）**从不**从被测实现的输出反推。

## 证据层级

| 级别 | 含义 | 本目录使用 |
| --- | --- | --- |
| E1 | 独立外部参考（官方 checkpoint / 官方 `SpeechFeatures`） | 本分支**未新增**；已有的 `validation/real_input_chain_20261017/manifest.json` 提供 |
| E2 | 独立重实现参考（不同实现/精度，仅用冻结权重与真实音频） | 全部数值交叉校验 |
| E3 | 变形不变量（不使用 golden，只用被测实现自身） | 加性、齐次、class token、token 置换等变 |
| E4 | 自洽/形状检查 | 形状、dtype、有限性、12 层互异 |

## E2 独立参考的实现边界

参考路径为 NumPy float64 的独立实现：

- MFCC：自写 `np.fft.rfft` + 自写 mel 滤波器组 + 自写 DCT，**不调用**
  `prepare_quant_samples.mfcc_matrices`。被测的 `prepare_quant_samples`
  RDFT 路径只作为对照量被记录。
- `99×64` 投影：真实权重张量 + `patch_proj` + class token + 位置编码。
- 12 层前向：显式 softmax、population-variance LayerNorm、`erf` GELU，
  逐层捕获 `LayerNorm1(x + attn_combine)`。

权重来源是冻结的 `software/weights/kwt1_fp32.npz`（`sha256=8f359abf…55825a`）。
该 NPZ 由 `export_weights.py` 从原始 `best_weights` 导出；本机**没有 TensorFlow**
（`tensorflow_available=false`），因此本轮没有直接读取原始 checkpoint，也没有
重建官方 Keras 模型。NPZ 只是冻结参数工件，不参与被测前向路径。

## 运行

`python3` 在本机指向无 PyTorch 的解释器；使用带 PyTorch 的解释器并强制 CPU：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/yang/miniconda3/bin/python software/audit_ffn1_input_boundary.py --sweep 60
```

快速回归（使用共享 `.build` 缓存，两条不同类别）：

```bash
cd software && /home/yang/miniconda3/bin/python -m unittest test_ffn1_input_boundary -v
```

## 实测结果（`manifest.json`）

两条真实 WAV（tar 内原始音频，不同类别：`yes` 目标词、`four` → `_unknown_`）：

| 校验 | `yes/676f8138_nohash_0.wav` | `four/750e3e75_nohash_0.wav` |
| --- | ---: | ---: |
| WAV SHA-256 | `b71582f6…b0e8b45` | `2ad981ba…8bcee9f` |
| MFCC 独立 FFT 路径 vs 项目 RDFT 路径 | `2.32458115e-05` | `7.15255737e-05` |
| `98×40 → 99×64` 独立 golden 最大绝对差 | `1.32914507e-06` | `9.58424968e-07` |
| 12 层 FFN1 捕获最大绝对差 | `3.39695828e-06` | `3.10831201e-06` |
| 最近错误边界（残差/注意力/层输入/LN2）距离 | `0.332` | `0.345` |

- 编码器输入为 FP32 `99×64`，每层 FFN1 捕获均为 FP32 `99×64`，共 12 个，
  形状见 manifest。
- 12 层捕获互不相同（两条样本的 12 层两两最小最大绝对差分别为
  `1.34266663`、`1.14747667`）。
- 变形不变量：投影加性 `2.95042992e-06`、齐次 `2.38418579e-07`、
  class token 行严格一致、FFN1 token 置换等变 `3.63495201e-06`。
- 扩展抽样 `--sweep 60`（12 个类别随机验证样本）：编码器最大绝对差
  `2.56496309e-06`，FFN1 最大绝对差 `6.90650716e-06`，最近错误边界
  最小距离 `0.18463343`。

门槛：MFCC `1e-3`、编码器 `2e-5`、FFN1 `5e-5`、变形 `1e-3`、边界判别余量 `1e-2`；
任一项超限即以非零退出码终止。两次运行退出码均为 0。

## 不能外推的结论

- 本审计覆盖的提取点就是下游量化脚本（`quantize_ffn1.py` 等）使用的同一
  `forward(..., capture_ffn=True)` 边界；但审计只验证该边界的数值与位置，
  不验证任何量化格式、尺度或 RTL。
- 这是**软件 FP32 数值路径**审计（E2+E3）。不证明 RTL、定点/量化模型或
  E203 SoC 联调正确；未改动任何 RTL 或量化配置。
- 没有 TensorFlow，故未用官方 Keras 模型或原始 checkpoint 做本轮 E1 交叉；
  官方 `SpeechFeatures` 的 E1 对照见 `validation/real_input_chain_20261017`。
- `--sweep` 与两条真实样本都**不是**赛题官方 600 条评测，不能替代正式精度门槛。
- golden 在 float64 下计算，被测为 float32，故记录的是最大绝对差而非逐位一致；
  逐位一致性需在同一数值格式下另行验证。
