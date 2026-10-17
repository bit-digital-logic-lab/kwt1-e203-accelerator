# 真实语音到首层编码器输入校验（2026-10-17）

运行。官方特征层依赖源码需先从指定提交取到临时目录（本地资料快照未包含这些层）：

```bash
git clone --filter=blob:none --no-checkout https://github.com/ARM-software/keyword-transformer.git /tmp/kwt-official-237e4097
git -C /tmp/kwt-official-237e4097 sparse-checkout init --cone
git -C /tmp/kwt-official-237e4097 sparse-checkout set kws_streaming/layers
git -C /tmp/kwt-official-237e4097 checkout 237e4097afd4f6851e0b82f0a8d65c2dfe8eae30
git -C /tmp/kwt-official-237e4097 rev-parse HEAD
```

随后在仓库根目录运行：

```bash
python3 software/validate_real_input_chain.py
software/.venv/bin/python software/validate_tf_checkpoint_input.py
PYTHONPATH=/tmp/kwt-official-237e4097 software/.venv/bin/python software/validate_official_tf_mfcc.py
```

数据源为本地 Speech Commands V2 压缩包中的 6 条真实 WAV，含 training/validation 两种官方 SHA1 说话人划分及 `yes`、`no`、`go`、`_unknown_` 四类。每条 archive member 路径、原始 WAV SHA256、PCM 参数、MFCC/编码器输入文件名及哈希见 [manifest.json](manifest.json)。6 条均为单声道、16 kHz、16-bit PCM、16000 帧。没有使用缓存的 MFCC 作为输入；每次从压缩包 WAV 重新生成。

数值结果：

| 校验 | 6 条中最大绝对误差 |
| --- | ---: |
| MFCC 手写 RDFT 路径与 NumPy FFT 频谱路径 | `6.52968884e-05` |
| MFCC 与 ARM 指定提交官方 `SpeechFeatures` 非流式推理路径 | `6.91711903e-05` |
| 离线脚本与独立 `einsum` 公式 | `9.53674316e-07` |
| 离线脚本与 TensorFlow 2.19.1 直接读取原始 checkpoint 后的 `tf.linalg.matmul`、class token、位置编码 | `1.43051147e-06` |
| 样本差分变形校验：`Δencoder[1:] = ΔMFCC @ kernel` | `2.38418579e-06` |

6 条首行 class token 完全相同；所有 MFCC 为 FP32 `98×40`，导出编码器输入为 FP32 `99×64`。上述误差均低于脚本阈值并以非零退出码保护。

**边界：** 官方特征层源码来自临时目录 `/tmp/kwt-official-237e4097`，Git HEAD 为 `237e4097afd4f6851e0b82f0a8d65c2dfe8eae30`。直接使用 `SpeechFeatures.get_params(flags)`、`Modes.NON_STREAM_INFERENCE` 和 `tf.audio.decode_wav`，无训练增强。逐样本官方 MFCC 与本地 NumPy 结果的最大绝对误差为 `6.91711903e-05`，不是位级一致。TensorFlow checkpoint 投影比对从本地 MFCC 开始，参数直接读取原始 `best_weights`，与离线脚本所用 NPZ 导出文件分离。这里的 `99×64` 是首层编码器输入；本科必做的 12 层 FFN1 输入仍需完整 FP32 前向提取。
