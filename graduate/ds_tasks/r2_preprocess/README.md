# r2_preprocess — WAV/MFCC → 99×64 首层编码器输入的独立逐元素校对

本目录是第二轮研究生任务 `r2_preprocess` 的**独立**成果，只覆盖**电脑端离线预处理链**：

```
真实 WAV(1 s/16 kHz/单声道) → MFCC 98×40 → 投影 40→64 → 前置 class token(1) + 位置编码 → 99×64
```

对齐赛题截图 [`题目/640.jpg 6`](../../../题目/640.jpg%206)（“离线完成音频规范化、`98×40` MFCC 特征提取、
输入投影、分类 token 拼接及位置编码，提供 `99×64` 的首层编码器输入”）与
[`题目/640.jpg 7`](../../../题目/640.jpg%207)（数据集划分与“训练集标定 / 验证集验证 / 不得用测试集调整数值配置”）。

## 校验对象（两条）

1. **98 → 99 的补位**：行 `0` = `class_emb + pos_emb[0]`；行 `1..98` = `MFCC @ kernel + bias + pos_emb[1..98]`。
   同时用**判别性对照**排除错位假设（位置编码整体错位一格、完全不加位置编码、把第 0 帧当作 class token）。
2. **文件序列化**：逐个 `.npy` 解析头部（magic/版本/`descr`/`fortran_order`/`shape`），核对
   载荷字节数 = `prod(shape)×4`、C 连续、`array.tobytes()` 的 SHA-256 与既有 `manifest.json` 一致、
   用生产写入器 `np.save(..., allow_pickle=False)` 重序列化后**与原文件逐字节相同**。

## 文件

| 文件 | 作用 |
| --- | --- |
| `verify_preprocess.py` | 独立校验器：float64 NumPy 自写 FFT-MFCC（rfft + 显式 mel + 正交 DCT-II）、独立 `98→99` 构造、`.npy` 头部解析与往返；参考值从不取被测输出 |
| `repro.sh` | 一键复现；日志与退出码写入 `logs/` |
| `results.json` | 机读结果（逐样本误差、序列化字段、门控、证据层级） |
| `logs/verify_preprocess.log` | 本次实跑 stdout（全部 `[PASS]`） |
| `logs/exit_codes.txt`、`logs/sources.sha256` | 退出码与脚本/结果哈希 |

## 实跑（2026-10-19，worktree 根目录）

```bash
bash graduate/ds_tasks/r2_preprocess/repro.sh        # 退出码 0，R2_PREPROCESS_EXIT=0
```

解释器 `/home/yang/miniconda3/bin/python`（Python 3.12.12，NumPy 2.5.2）。只读输入：
主工程 gitignored 数据集 `speech_commands_v0.02.tar.gz`、冻结权重 `software/weights/kwt1_fp32.npz`
（`sha256=8f359abf…`）、`flags.json`（`sha256=fcd480d3…`）、`software/validation/real_input_chain_20261017/*.npy`。

## 关键实测值（13 条真实样本）

- **6 条有存档 `.npy` 的真实样本**（3 训练 + 3 验证，覆盖 `yes/no/go/_unknown_`）：
  - MFCC：独立 float64 实现 vs 存档 FP32，最大绝对差 **7.15e-05**（逐元素 3,920/条，共 23,520）。
  - `99×64` 同输入重建（存档 MFCC → 生产 `convert()`）：最大绝对差 **0**，重序列化后 `.npy` **逐字节相同**。
  - 独立端到端（自写 MFCC + 自写 `98→99`）vs 存档：最大绝对差 **2.67e-05**，最差 NRMSE **2.60e-06**（逐元素 6,336/条）。
  - class token 行：最大绝对差 **0**，且 6 条样本第 0 行**完全相同**（同一 SHA-256）。
  - 位置编码错位判别：正确偏移误差 **0**；错位一格 **0.624**、完全不加 **0.652**、第 0 帧误作 token **2.13**。
- **额外 7 条真实样本**（补齐 `up/down/left/right/on/off/stop`，合计覆盖全部 10 个目标词）：
  独立路径 vs 生产路径 MFCC 最大绝对差 **1.27e-4**、`99×64` 最大绝对差 **4.43e-5**。
- **行数来历**：`(16000−480)/160+1 = 98` 帧，`98 + 1(class) = 99` 行，与 `flags.json` 的 `spectrogram_length=98` 一致。

## 划分纪律

- 划分标签由官方哈希规则 `prepare_quant_samples.split_for()` 决定；6 条存档样本 = 3 训练 + 3 验证。
- **本轮不做任何标定**：无尺度、无量化参数、无阈值从数据拟合。所有容差均为**先验** float32 舍入常数
  （`1e-3` / `2e-5`，沿用生产脚本既有守卫，未在任何划分上调参）。验证样本**只用于评估**。

## 证据层级

| 类别 | 本轮结论 | 层级 / 说明 |
| --- | --- | --- |
| 软件参考 | **已证（本任务）** | E2 独立重实现 + E3 变形/判别不变量 + E4 序列化自洽 |
| RTL 逐位 | **未证（本任务）** | 预处理为电脑端离线，本任务不涉及 RTL |
| E203 | **未证（本任务）** | 未跑真实 E203 指令流 / System Bus / NICE |
| 综合 | **未证（本任务）** | 无 PDK/SDC，无 150 MHz 时序/面积/功耗 |
| 官方验收 | **未证（本任务）** | 13 条自选真实样本，非官方 600 条评测集 |

不宣称：RTL/E203/综合/官方精度任一通过；本任务只签发“离线预处理链在 13 条真实样本上与独立实现逐元素一致、`.npy` 序列化无损可复现”。
