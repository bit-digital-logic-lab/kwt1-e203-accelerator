# WORKER_REPORT

- 分支：`codex/kwt-preprocess600`
- 任务：扩大“真实音频 → 12 层 `X[layer]`”离线输入链的本地验证覆盖；核对训练/
  验证/测试划分与类别；生成 ≥120 条跨类别验证样本的 12 层输入与独立检查
  manifest，并对其中样本用现有独立 NumPy/变形方法校对 `98×40 → 99×64` 与 12 层
  FFN1 边界；不改主目录/其他分支，不 push。
- 边界：只改本 worktree；不开子智能体；共享缓存只读；不从被测输出制造 golden；
  不把本地样本称官方隐藏 600。

## 改动

新增（`software/validation/large_scale/`）：

- `build_validation_inputs.py`：本地大规模验证脚本。按
  `prepare_quant_samples.split_for` 的官方 10/10/80 哈希规则**只取验证划分**，
  每标签取 `sha256(name)` 前 50 条，跨 12 类共 600 条（550 真实音频 + 50 官方
  零波形 `_silence_`）。逐条生成 `98×40`、`99×64` 与 12 层 `X[layer]`，记录
  样本 ID、标签、WAV/权重 SHA、形状、dtype、有限性、12 层互异、逐层捕获 SHA、
  输出摘要（argmax/标签/logits 范围），并与 NumPy float64 独立参考逐层比对。
- `manifest_summary.json`：紧凑机读摘要（600 条指纹/误差/通过情况 + 汇总），
  无大数组。
- `README.md`：方法、证据层级、划分核对、实测、可复现命令、不可外推边界。

修改：

- `WORKER_REPORT.md`：本报告。

未改动：任何 RTL、量化配置、`run.py`、`offline_encoder_input.py`、
`prepare_quant_samples.py`、`audit_ffn1_input_boundary.py`。大数组与逐层 SHA
全量清单写入 git 忽略的 `software/.build/large_scale/manifest_full.json`（不提交）。

## 划分与类别核对

- 12 类来自 `labels.txt`（`_silence_`、`_unknown_` + 10 目标词）。
- 全量扫描 `105,829` 个 WAV 成员；验证划分候选 `9,981` 条
  （`_unknown_` `6,278`，其余 10 词各 `350`–`406`）。
- **测试划分从不读取、不用于选择/调参**；脚本仅保留 `validation`。

## 实际执行的命令与退出结果

解释器：`/home/yang/miniconda3/bin/python`（Python 3.12.12，NumPy 2.5.2，
PyTorch 2.5.1+cu121；强制 CPU）。

| 命令 | 退出码 | 结果 |
| --- | ---: | --- |
| `/home/yang/miniconda3/bin/python -m py_compile software/validation/large_scale/build_validation_inputs.py` | 0 | 编译通过 |
| `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/yang/miniconda3/bin/python software/validation/large_scale/build_validation_inputs.py --per-label 3`（烟测） | 0 | 36 条通过 |
| `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /home/yang/miniconda3/bin/python software/validation/large_scale/build_validation_inputs.py --per-label 50` | 0 | **600 条全部通过，失败 0** |

耗时（同一进程，CPU）：选样 `51.755 s` + 评估 `154.308 s` = 总计 `206.313 s`
（约 3.4 分钟，无无界耗时）。

## 关键实测值

- 样本数 `600`：真实音频 `550`（11 类各 50）+ 合成 `_silence_` `50`；失败 `0`。
- 权重 `software/weights/kwt1_fp32.npz`：
  `sha256=8f359abf7adbe287b8f89fca98af46194f98f90f2146784731bff551a155825a`。
- MFCC 独立 FFT vs 项目 RDFT 最大绝对差 `8.28206539e-04`（阈值 `1e-3`）；
  最坏样本 `./on/5fadb538_nohash_3.wav`，绝对差约 `5e-5` 倍特征 std，属
  float32 DFT 矩阵 vs float64 FFT 舍入。
- `98×40 → 99×64` 独立 golden 最大绝对差 `2.94790213e-06`（阈值 `2e-5`）。
- 12 层 FFN1 捕获最大绝对差 `8.89790551e-06`（阈值 `5e-5`）；最坏样本
  `./off/794cdfc5_nohash_1.wav`。
- 最近错误边界最小距离 `1.84633583e-01`（≫ 正确边界 + 余量；边界判别成立）。
- 12 层互异最小两两最大绝对差 `8.55416894e-01`（> 0）。
- 端到端真实链（项目 MFCC → 投影 → 前向）FFN1 最大绝对差 `6.16060285e-04`。
- 变形：投影加性 `2.46614218e-06`、齐次 `2.38418579e-07`、FFN1 token 置换等变
  `2.74181366e-06`、class token 行严格一致。
- **失败样本：无。**

## 证据层级

- **E2（独立重实现）**：600 条全部对 NumPy float64 独立参考交叉；参考只用冻结
  权重与真实 WAV，不取被测输出。
- **E3（变形不变量）**：投影加性/齐次、class token、FFN1 token 置换等变。
- **E4（自洽/形状）**：形状、dtype、有限性、12 层互异、逐层捕获 SHA。
- **E1（独立外部）**：本轮未新增（本机无 TensorFlow）；既有 E1 见
  `software/validation/real_input_chain_20261017/manifest.json`。

## 未证事项

- 600 条为**本地验证划分**样本，**不是**官方隐藏 600，不能替代正式精度门槛。
- 未证明任何量化/定点模型、RTL、E203 SoC 联调；未改 RTL/量化配置。
- 无 TensorFlow，未用官方 Keras 模型/原始 checkpoint 做本轮 E1 交叉。
- golden 为 float64、被测 float32，记录最大绝对差而非逐位一致。
- 未评估分类准确率、完整 12 层端到端分类精度或性能/面积/功耗。
- 端到端真实链最大差 `6.16e-04` 主要来自最坏 MFCC 样本，非边界审计误差。
