# WORKER_REPORT — codex/kwt-grad-golden（研究生组第 4–6 项 FP32 Golden）

- 分支：`codex/kwt-grad-golden`（worktree `/home/yang/kwt-worktrees/grad-golden`）
- 范围：研究生必做**第 4–6 项**（单头注意力 / 单层 PostNorm 编码器 / 十二层编码器 + 分类头）的
  **独立纯 NumPy FP32 golden** 与**中间张量逐阶段捕获**；只在 `graduate/reference/` 增加脚本、
  小结果与测试，并在根 `WORKER_REPORT.md` 记录命令、退出码、样本数与误差。
- 边界：只在本 worktree 内操作；未开子智能体；未 `checkout`/改主目录或他人分支；
  未 `merge`/`rebase`/`cherry-pick`/`push`；未改 `software/` 既有实现；未落大张量/缓存；
  **未**把 PC 端 MFCC 计入加速器（golden 输入边界 = 官方首层编码器输入 `X[99,64]`）。
- 前情：本 worktree 根 `WORKER_REPORT.md` 原为 `codex/kwt-integration` 的报告，依任务要求改写为
  本报告；原内容不作删除，可从 Git 历史（`main`/`3d9ae98`）取回。

## 交付物

`graduate/reference/`：

| 文件 | 说明 |
| --- | --- |
| `kwt1_fp32_golden.py` | 纯 NumPy FP32 golden（第 4–6 项），含中间张量捕获接口；无 TF/scipy 依赖 |
| `capture_intermediates.py` | 真实验证划分样本逐层捕获形状/范围/SHA-256 → 小 JSON |
| `compare_official_tf.py` | 与官方 Arm KWT-1 `transformer_utils.py` 原文代码（TF）逐阶段独立比对 |
| `compare_torch_reference.py` | 与既有独立 PyTorch 实现 `software/run.py` 二次独立比对 |
| `test_kwt1_fp32_golden.py` | 可运行单元测试（仅需 NumPy） |
| `results/golden_intermediates.json` | 3 样本 ×12 层逐阶段统计+SHA（148 KB，无张量） |
| `results/official_tf_comparison.json` / `results/torch_reference_comparison.json` | 独立比对误差 |
| `README.md` | 语义核对表、命令、已证/未证 |

## 官方语义核对结论

`flags.json`：`num_layers=12, d_model=64, mlp_dim=256, heads=1, attention_type=time,
spectrogram_length=98, dct_num_features=40, label_count=12`；`prenorm`/`approximate_gelu`
未在 flags.json 中，取 `model_parameters` 默认 → **PostNorm + 精确 GELU**（`eval.sh` 亦未覆盖）。
逐项核对：Q/K/V `Dense(64)` **无 bias**；分数 `99×99`；`/√64`；逐行 Softmax（行和 =1）；
`softmax@V`；`combine_heads` 输出投影带 bias；`out1=LN1(x+attn)`、`out=LN2(out1+ffn)`
两次残差/两次 LayerNorm（eps=1e-6，总体方差）；FFN `64→256→64` 精确 GELU
`0.5x(1+erf(x/√2))`；类别 token `x[:,0]` → `Dense(12)`；12 层权值 SHA-256 互异。

## 实跑（实际退出码）

| 命令 | 退出码 |
| --- | ---: |
| `python3 graduate/reference/test_kwt1_fp32_golden.py` | **0**（`Ran 12 tests ... OK`） |
| `python3 graduate/reference/capture_intermediates.py` | **0** |
| `.venv/bin/python graduate/reference/compare_official_tf.py` | **0** |
| `.venv/bin/python graduate/reference/compare_torch_reference.py` | **0** |

（TF/torch 解释器：`/home/yang/Desktop/ic设计竞赛/software/.venv/bin/python`，TF 2.19.1 / torch 2.5.1，
只读工具链，未改动该目录。）

## 样本数

真实验证划分（Speech Commands V2 官方 SHA-1 说话人划分的 **validation**，跨类别 **3** 条）：
`four/750e3e75_nohash_0.wav`(`_unknown_`)、`no/45692b02_nohash_3.wav`(`no`)、
`yes/ec21c46b_nohash_1.wav`(`yes`)。每条 `X[99,64]` 的 SHA-256 与
`software/validation/real_input_chain_20261017/manifest.json` 一致后使用；
`sample_count=3`、`12 层 × 8 中间阶段`。验证划分**仅用于评估**，未用于调参。

## 误差

| 比对 | 3 样本最大绝对误差 | 阈值 | 退出码 |
| --- | ---: | ---: | ---: |
| golden vs 官方 TF 原文代码（全 12 层逐阶段） | `9.298e-06` | `1e-4` | 0 |
| golden vs PyTorch `run.py`（逐层 LN1/末层/logits） | `4.768e-06` | `1e-4` | 0 |

官方 TF 单阶段最差：`softmax 8.8e-07`、`attended_v 1.8e-06`、`attn_projected 2.1e-06`、
`ln1 4.1e-06`、`ffn_hidden 4.4e-06`、`gelu 4.4e-06`、`ffn_out 3.9e-06`、
`layer_output 9.3e-06`、`logits 9.5e-07`。golden 首次运行即被测试发现 GELU 漏乘 `1/√2`
（误差 8.4e-02），修正后通过；该 bug 已修复，非结果粉饰。

## 已证

- 第 4–6 项操作顺序与官方源码一致，并在 3 条**真实验证划分**样本上，与**官方 TF 原文代码**
  全 12 层逐阶段一致（≤`9.3e-06`），另与独立 PyTorch 实现一致（≤`4.8e-06`）。
- 12 层逐层换参：参数 SHA-256 12 组互异；`num_layers=k` 输出与全量前 k 层一致。
- 中间张量（`99×99` 注意力、`99×256` FFN、两次 LayerNorm 等）形状/范围/SHA 已记录，未落张量。

## 未证（证明边界）

1. 官方层代码写于 TF 2.4 + `tensorflow_addons`；本机 TF 2.19 缺 `tensorflow_addons` 与
   `keras.layers.experimental.preprocessing`，比对时装了**两个仅 import 用的 shim**
   （`GELU→tf.nn.gelu(exact)`、`Rescaling→Rescaling`），均不在被执行路径上。这是**框架版本差异**，
   不是官方 TF 2.4 位级运行。
2. golden 输入取既有真实数据管线缓存的 `X[99,64]`（该管线此前对 TF checkpoint 投影误差 ≤`1.4e-06`）；
   本目录**未**重跑官方 `speech_features` MFCC，MFCC 不属加速器范围。
3. 未测官方 600 条评测集准确率 / NRMSE；未做量化（A12W10 等）或 RTL/E203 对照；本目录只给 FP32 golden。
4. 3 条样本的 top-1 命中仅作 sanity check，**不是**精度结论。
