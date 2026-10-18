# 本地大规模验证：真实音频 → 12 层 `X[layer]` 离线输入链（2026-10-18）

本目录由 [`build_validation_inputs.py`](build_validation_inputs.py) 生成，扩大
[`audit_ffn1_input_boundary.py`](../../audit_ffn1_input_boundary.py) 已建立的两点
独立审计（`98×40 → 99×64` 与 12 层 FFN1 输入边界）的验证规模，覆盖从**原始真实
音频**到**每层 `X[layer]`** 的整条离线输入链。

> **这不是赛题官方隐藏的 600 条。** 官方 600 条在提交冻结后由组委会从未公开的
> 样本中抽取；本目录用的是按官方划分规则从 `speech_commands_v0.02` 的**验证划分**
> 本地选取的样本，只能作为本地功能/数值证据，不能替代正式精度门槛。

## 与题面/补充分析的关系

- `题目/README.md` 第 4 节：官方数据为 Google Speech Commands V2；训练划分做无
  梯度标定，验证划分调试功能/精度；**测试划分不能用于调权重、量化参数或数值配置**。
- 用户补充分析 `本科组题目深度分析与问题清单_20261018.md` 的 **P1-5**：600 条
  规模上“任意样本 → 12 层 `X[layer]`”的完整 FP32 前向可复现性尚未验证。
- `doc/本科FFN1数据与硬件边界核对_20261017.md`：12 层 `X[layer]` 是每层注意力、
  残差和第一处 LayerNorm 之后的 `99×64` 张量，由电脑端离线提取，不是首层编码器输入。

## 划分与类别核对（只用验证划分）

- 类别：`labels.txt` 的 12 类（`_silence_`、`_unknown_` + 10 个目标词）。
- 划分：直接复用 `prepare_quant_samples.split_for`（官方 10 % 验证 / 10 % 测试 /
  80 % 训练 的 SHA-1 百分位规则）。**只保留 `validation`**；`testing` 划分在整个
  流程中从不读取、从不用于选择或调参。
- 本次全量扫描 `105,829` 个 WAV 成员，其中验证划分候选 `9,981` 条
  （`_unknown_` `6,278`，其余 10 词各 `350`–`406`）。
- 选样规则：每个标签按 `sha256(member name)` 升序取前 `N`（默认 50），确定性、
  可复现；`_background_noise_` 从不作为样本。

## 样本规模

- **600 条**：11 个真实音频标签各 50 条（共 **550 条真实音频**）+
  `_silence_` 50 条（官方 12 类中的零波形特殊标签，**合成、非真实音频**，单列）。
- `_silence_` 单独标注，不混入“真实音频”计数。

## 独立参考（golden 从不取自被测输出）

参考路径为 NumPy float64 独立实现（来自 `audit_ffn1_input_boundary.py`）：

- MFCC：自写 `np.fft.rfft` + 自写 mel 滤波器组 + 自写 DCT，**不调用**
  `prepare_quant_samples.mfcc_matrices`。被测 RDFT 路径仅作对照量记录。
- `98×40 → 99×64`：真实权重 + `patch_proj` + class token + 位置编码。
- 12 层前向：显式 softmax、population-variance LayerNorm、`erf` GELU，逐层捕获
  `LayerNorm1(x + attn_combine)`。

边界判别使用错误候选（pre-LN1 残差、注意力组合、层输入、LN2）：捕获点若取错位置
无法通过。变形不变量（投影加性/齐次、class token、FFN1 token 置换等变）不使用
golden。形状/dtype/有限性/12 层互异为自洽检查。

冻结权重：`software/weights/kwt1_fp32.npz`
`sha256=8f359abf7adbe287b8f89fca98af46194f98f90f2146784731bff551a155825a`。
共享输入只读：`资料/数据集/speech_commands_v0.02.tar.gz`（`2,428,923,189 B`）与
`/home/yang/Desktop/ic设计竞赛/software/.build/quant_samples_12labels_40.npz`。
软件缓存从不写回，也不从被测输出反推 golden。

## 运行与产物

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/yang/miniconda3/bin/python \
  software/validation/large_scale/build_validation_inputs.py --per-label 50
```

- 大数组与逐层 SHA 全量清单写入 git 忽略目录
  `software/.build/large_scale/manifest_full.json`（`5.6 MB`，**不提交**）。
- 提交的是本目录的紧凑摘要 `manifest_summary.json`（`~0.55 MB`，仅哈希/形状/
  误差/摘要，无数组）与本方法文档、脚本。
- 退出码：0 表示 600 条全部在阈值内。任一 MFCC/编码器/FFN1/边界判别不通过则打印
  失败样本并返回非零。

## 实测结果（2026-10-18，CPU）

| 项 | 结果 |
| --- | ---: |
| 样本数 / 真实音频 / `_silence_` | `600` / `550` / `50` |
| 失败样本 | `0` |
| MFCC 独立 FFT vs 项目 RDFT 最大绝对差 | `8.28206539e-04`（阈值 `1e-3`） |
| `98×40 → 99×64` 独立 golden 最大绝对差 | `2.94790213e-06`（阈值 `2e-5`） |
| 12 层 FFN1 捕获最大绝对差 | `8.89790551e-06`（阈值 `5e-5`） |
| 最近错误边界最小距离 | `1.84633583e-01`（≫ 正确边界 5e-5 + 余量 1e-2） |
| 12 层互异（全样本最小两两最大绝对差） | `8.55416894e-01`（> 0） |
| 端到端真实链（项目 MFCC → 投影 → 前向）FFN1 最大绝对差 | `6.16060285e-04` |
| 投影加性 / 齐次 / 置换等变 | `2.466e-06` / `2.384e-07` / `2.742e-06` |
| class token 行严格一致 | 是 |

- 最坏 MFCC 样本 `./on/5fadb538_nohash_3.wav`：绝对差 `8.282e-04` 出现在特征
  值 `≈0.98`（该样本特征 std `≈16.7`、min `≈-349.5`），即约 `5e-5` 倍 std、
  `2.4e-6` 倍最大幅值；mel-log 级差 `3.8e-03`。差异量级符合 float32 DFT 矩阵
  vs float64 FFT 的舍入，非建模错误。
- 最坏 FFN1 样本 `./off/794cdfc5_nohash_1.wav`：`8.898e-06`。
- 端到端真实链最大差 `6.16e-04` 主要来自最坏 MFCC 样本的 `8.3e-04` 特征差经
  投影传播；边界审计（使用独立 MFCC 作为输入）本身隔离了该误差。
- 耗时（同机单进程 CPU）：选样 `51.755 s` + 评估 `154.308 s` = `206.313 s`。

## 不能外推的结论

- 600 条为**本地验证划分**样本，**不是**官方 600 条评测，不能替代正式精度门槛。
- 覆盖的是软件 FP32 离线链路（E2+E3+E4）；证明的是 `98×40 → 99×64` 与 12 层
  `X[layer]` 提取的数值与位置，**不证明**任何量化/定点模型、RTL、E203 SoC 联调。
- 未改任何 RTL、量化配置、`run.py`、`offline_encoder_input.py`、
  `prepare_quant_samples.py`。
- 无 TensorFlow，未用官方 Keras 模型/原始 checkpoint 做本轮 E1 交叉；官方
  `SpeechFeatures` MFCC 的 E1 对照见
  `software/validation/real_input_chain_20261017/manifest.json`。
- golden 为 float64、被测为 float32，记录的是最大绝对差，不是逐位一致。
- 未评估分类准确率、端到端 12 层分类精度、性能/面积/功耗。
