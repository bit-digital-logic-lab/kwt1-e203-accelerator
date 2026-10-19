# WORKER_REPORT — r2_preprocess（WAV/MFCC → 99×64 首层编码器输入的独立逐元素校对）

- 分支 / worktree：`codex/kwt-ds-grad-r2_preprocess` @ `/home/yang/kwt-worktrees/ds-grad-r2_preprocess`
- 基线：`main@7b6abee`（接下来只读参考既有 `software/` 与 `graduate/encoder/`）
- 范围：**只写** `graduate/ds_tasks/r2_preprocess/` 与根 `WORKER_REPORT.md`；其余目录只读。
- 边界：只在本 worktree 操作；未开子智能体；未改其他 worktree；未 `merge`/`rebase`/`push`；
  只读使用主工程 `/home/yang/Desktop/ic设计竞赛` 的 gitignored 数据集；未提交大数据/密钥/缓存。
- 对齐题目：[`题目/640.jpg 6`](题目/640.jpg%206)（离线 `MFCC 98×40 →` 投影、class token、位置编码 `→ 99×64`）、
  [`题目/640.jpg 7`](题目/640.jpg%207)（训练集标定 / 验证集验证、不得用测试集调整数值配置）。

## 1. 结果

**判定：离线预处理链在 13 条真实语音样本上与独立实现逐元素一致，`98→99` 补位与 `.npy` 序列化均核对通过；11/11 门控 PASS，退出码 0。**

- **`98→99` 补位（核心）**：行 `0` = `class_emb + pos_emb[0]`；行 `1..98` = `MFCC @ kernel + bias + pos_emb[1..98]`。
  6 条存档样本上，class token 行同输入重建误差 **0**、class token 行跨 6 条完全相同（同一 SHA-256）；
  位置编码**正确偏移**误差 **0**，而**错位一格 0.624**、**完全不加 0.652**、**第 0 帧误作 token 2.13** —— 判别性对照成立。
- **行数来历**：`(16000−480)/160 + 1 = 98` 帧，`+1`（class token）`= 99`，与 `flags.json` 的
  `spectrogram_length=98` 一致。
- **逐元素数值（6 条存档真实样本，MFCC 各 3,920 元、编码器各 6,336 元）**：
  - 独立 float64 MFCC vs 存档 FP32：最大绝对差 **7.15e-05**（阈值 1e-3）；
  - 同输入 `99×64` 重建（存档 MFCC → 生产 `offline_encoder_input.convert()`）：最大绝对差 **0**，
    用生产写入器重序列化后 `.npy` 与存档**逐字节相同**；
  - 独立端到端（自写 MFCC + 自写 `98→99`）vs 存档：最大绝对差 **2.67e-05**，最差 NRMSE **2.60e-06**。
- **`.npy` 序列化**：`\x93NUMPY` **v1.0**、`descr '<f4'`、`fortran_order False`、C 连续；
  载荷字节 = `prod(shape)×4`（MFCC 15,680 / 编码器 25,344）；`array.tobytes()` 的 SHA-256 与
  `manifest.json` 的 `mfcc_sha256`/`encoder_sha256` **全部一致**；往返重序列化**逐字节相同**且数组相等。
- **13 条真实样本**：6 条存档（3 训练 + 3 验证，`yes/no/go/_unknown_`）+ 7 条额外（补齐
  `up/down/left/right/on/off/stop`，覆盖全部 10 个目标词）。额外样本上独立路径 vs 生产路径：
  MFCC 最大绝对差 **1.27e-4**、`99×64` 最大绝对差 **4.43e-5**。
- **划分纪律**：划分沿用官方哈希规则；**本轮不做任何标定**（无尺度/量化/阈值从数据拟合）。
  所有容差为**先验** float32 舍入常数（`1e-3` / `2e-5`，沿用生产脚本既有守卫），**验证样本只用于评估**。
- **证据层级**：*软件参考* = **已证**（E2 独立重实现 + E3 判别不变量 + E4 序列化自洽）；
  *RTL 逐位*、*E203*、*综合*、*官方验收* = **本任务未证**（见 §3）。

## 2. 复现命令与退出码（本 worktree 根目录，2026-10-19 实跑）

| 命令 | 退出码 | 结果 / 样本数 |
| --- | ---: | --- |
| `bash graduate/ds_tasks/r2_preprocess/repro.sh` | **0** | `R2_PREPROCESS_EXIT=0`；`gates=11/11 all_pass=True` |
| `/home/yang/miniconda3/bin/python graduate/ds_tasks/r2_preprocess/verify_preprocess.py` | **0** | 13 条真实样本；6 条存档逐元素 + 序列化核对，7 条独立 vs 生产路径交叉 |

解释器 `/home/yang/miniconda3/bin/python`（Python 3.12.12，NumPy 2.5.2；强制 CPU，
`OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4`）。日志与退出码：`logs/verify_preprocess.log`、
`logs/exit_codes.txt`（`verify_preprocess.py=0`）；脚本/结果哈希：`logs/sources.sha256`。

关键产物 sha256：`verify_preprocess.py=12d41925…`、`repro.sh=45f89286…`、`results.json=cc6c742a…`、
`logs/verify_preprocess.log=f8e60ec9…`。只读输入：权重 `software/weights/kwt1_fp32.npz=8f359abf…`、
`flags.json=fcd480d3…`、数据集 `…/speech_commands_v0.02.tar.gz`（单次流式扫描 98,399 个成员）。

## 3. 未证事项

1. **RTL 逐位未证**：预处理为电脑端离线阶段，本任务完全不涉及 RTL；`99×64` 的 RTL 逐位一致由其它
   研究生任务单独负责，本报告不声明。
2. **E203 未证**：未在真实 E203 指令流 / System Bus / NICE 上运行任何数据搬运或联调。
3. **综合未证**：无目标 PDK / SDC / SRAM 宏，未做 150 MHz 时序、面积、功耗签核。
4. **官方验收未证**：仅 13 条自选真实样本；不是官方 600 条评测集，不能替代正式精度/准确率门槛，
   也未做完整 12 层端到端或分类头评测。
5. **非位级一致（数值）**：MFCC 独立实现为 float64、被测为 float32，报告的是浮点绝对差，**不是逐位一致**；
   `encoder_same_input=0` 与往返字节相同只覆盖“同一输入 + 序列化”，不覆盖浮点路径的逐位性。
6. **仅首层编码器输入**：本任务只覆盖 `98×40 → 99×64`；12 层 FFN1 输入、注意力、分类头均不在范围。
7. **未用官方 TF 复核**：本机无 TensorFlow，未重跑官方 `SpeechFeatures` / 原始 checkpoint（既有 E1
   见 `software/validation/real_input_chain_20261017/manifest.json`，本报告不重复声称）。
8. 不宣称“RTL/E203/综合通过”“官方精度达标”或“研究生 4–6 项全部通过”。
