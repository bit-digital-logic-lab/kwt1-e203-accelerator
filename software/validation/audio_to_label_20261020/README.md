# WAV 到 12 类输出的软件核对

本目录的 `results.json` 由 `software/validate_audio_to_label.py` 生成。脚本从 Speech Commands V2 原始压缩包重新读取 WAV，按项目 MFCC 实现生成 `98×40`，经 FP32 输入投影、class token 和位置编码得到 `99×64`，再运行 12 层 FP32 KWT-1 与电脑端 FP32 分类头。六条单独留档的真实 WAV 还逐条用原始 TensorFlow checkpoint 计算完整 logits，与 PyTorch 实现独立比较。

`quantized_results.json` 由 `software/validate_quantized_audio_to_label.py` 生成。它用已冻结并校验 SHA-256 的 p027_precision 参数包运行软件整数编码器，然后只用 FP32 分类头输出 12 类。它检验量化对类别的影响，但**不是 RTL 或 SoC 输出**。

## 实测口径

- 本机复跑结果：FP32 全链在 480 条本地验证样本上为 **470/480 = 97.917%**；冻结整数编码器接 FP32 分类头也是 **470/480**，相对 FP32 **0 次 top-1 翻转**。六条单独抽取的真实 WAV 两条路径均为 **6/6**。整数路径的合并 logits NRMSE 为 **0.07883%**，最大绝对差 **0.02626**。这是分类结果的误差口径，区别于编码器张量 NRMSE。12 类各取一条，用原始 TensorFlow checkpoint 复算 logits，最大绝对差 **1.43×10⁻⁶**。
- 验证缓存共 480 条，每类 40 条；其中 440 条是真实 WAV，40 条 `_silence_` 是按数据准备脚本定义的全零波形。脚本可重新从原始 WAV/零波形生成全部 MFCC，并逐元素对照缓存。
- 验证集采用本地 Speech Commands 选样和预处理，不是官方隐藏 600 条。`_unknown_` 来自非目标词；`_silence_` 是合成零波形。本地准确率不能作为官方测试集准确率。
- 权重和量化参数没有在这些验证样本上重新标定；整数参数包由另一分支的训练数据冻结。FP32 分类头保持在电脑端，不属于 RCL 量化对象。
- `results.json` 记录全部样本的真值、预测、分类间隔、SHA-256、重建 MFCC 最大差、TensorFlow/PyTorch logits 最大差；`quantized_results.json` 另记录量化前后翻转、准确率和 logits 误差。

## 重跑

在项目根目录：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 \
  software/.venv/bin/python software/validate_audio_to_label.py \
  --device cpu --rebuild-cache-wavs

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
  python3 software/validate_quantized_audio_to_label.py \
  --pack /home/yang/kwt-worktrees/ds-p027-precision/graduate/ds_tasks/p027_precision/frozen \
  --rebuild-cache-wavs
```

`--pack` 所指的冻结包目前在独立的 p027_precision 工作树，尚未并入此分支。第二条命令会验证参数包哈希，不调用标定程序。两个脚本都支持 `--limit N` 做较小的本地重跑。

## 验证边界

完整 WAV→类别的软件数值路径和冻结整数模型可在本机复跑。当前没有从 RCL RTL/SoC 端接收最终 12 层编码器输出并接分类头，所以这里的结果不能证明硬件端到端正确或满足赛题时序、面积和功耗。独立 TensorFlow 完整模型核对仅覆盖六条真实 WAV；440 条本地验证 WAV 使用重建特征和 PyTorch 模型。
