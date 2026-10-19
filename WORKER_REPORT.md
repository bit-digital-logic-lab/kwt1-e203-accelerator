# WORKER_REPORT — codex/kwt-ds-grad-dataset（研究生任务：数据集核查）

- 分支 / worktree：`codex/kwt-ds-grad-dataset` @ `/home/yang/kwt-worktrees/ds-grad-dataset`
- 范围：核查研究生任务的**训练/验证划分、哈希、样本重叠、现有校准泄漏**；产物只写 `graduate/ds_tasks/dataset/`，另写本报告。
- 边界：只读使用主工程 `/home/yang/Desktop/ic设计竞赛` 的 gitignored 数据；未开子智能体；未改其它 worktree/`main`/`hardware/`；未 `merge`/`rebase`/`push`；未提交大数据/密钥/缓存。
- 详细报告：[`graduate/ds_tasks/dataset/REPORT.md`](graduate/ds_tasks/dataset/REPORT.md)；机器结果 [`logs/audit.json`](graduate/ds_tasks/dataset/logs/audit.json)。
- 注：本 worktree 基线继承的 grad-encoder 版 `WORKER_REPORT.md` 已保存在 git 历史（`5136ca1`）。

## 1. 结果

- **划分（通过）**：按官方 `which_set`（SHA1 去 `_nohash_`，10%/10%）重算 5 个缓存（200/240/480/960 条及 40 条 legacy）与 6 条真实样本，真实 WAV 标签 **0 处不一致**；合成静音按划分生成，单独统计。
- **哈希（通过）**：区分两类哈希——**文件容器哈希**（整文件字节）与**张量内容哈希**（`np.load(f).tobytes()`）。权重 `8f359abf…`、`flags.json` `fcd480d3…`、A12W10 参数包 `a77f60fb…`、归档 2,428,923,189 字节 / 105,878 成员 / 105,835 WAV、6 条真实 WAV **全部一致**；6 条 `sample_0X_{mfcc,encoder}.npy` 的**张量内容哈希 12/12 与 manifest 记录一致**（`software/validate_real_input_chain.py` 第 93/95 行即按 `array.tobytes()` 定义该字段）。早期“12 处不一致”是把 `.npy` **容器哈希**去比 manifest **内容哈希**的量纲误报，已修正：两类哈希在 `logs/audit.json`（`hashes.items.sample_XX_*`）与日志中分列 `tensor_content` / `file_container`，**只用内容对内容比较**，容器哈希仅登记（manifest 对其无期望值）。独立 40→64 投影复算误差 0.0（≤ manifest 自记上限 `4.77e-07…9.54e-07`），二次确认内容一致。
- **重叠（通过）**：各缓存 train/val/test 路径两两 0 重叠；跨缓存 0 重叠；真实样本特征张量 0 重复。
- **校准泄漏（发现 2 处）**：① 单层编码器 `gen_vectors.py` 硬编码**验证划分** `sample_00`，`attn_int_model.py`/`kwt_postnorm_ref.calibrate()` 据此求静态尺度——换用训练 `sample_01` 后注意力 `scale_x/scale_o`、PostNorm `s_x/s_a/s_r1` 变化约 **16.5%–16.8%**，`eps_q1` 差 9597、`KC` 差 1856（RTL 激励尺度随样本变化，违反“验证划分仅评估”）；② legacy `quant_samples.npz` 仍含 **20 条测试划分**样本，`results/README.md` 自述开发中被看过（已知污染）。FFN1 量化链（A12W10/blackbox）**干净**：`validation_used_for_calibration=False`、训练/验证 240/240 不相交、选中倍率全为 `(1.0,1.0)`、`s_x/sx` 与样本无关。
- **判定**：脚本 `VERDICT FAIL (rule violations)`，退出码 **1**（检出 **3 条**违规，全部为真实泄漏——验证样本标定 2 处 + 旧测试缓存 1 处；无哈希类误报，非故障）。证据层级：全部为**软件参考（CPU NumPy）**；RTL 逐位仅引用 grad-encoder 且继承泄漏；E203/综合/官方验收**未证**。

## 2. 复现命令与退出码

仓库根 `/home/yang/kwt-worktrees/ds-grad-dataset`：

| 命令 | 退出码 | 结果 |
| --- | ---: | --- |
| `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 graduate/ds_tasks/dataset/audit_dataset.py > graduate/ds_tasks/dataset/logs/audit.log 2>&1` | **1** | 全量核查（含 2.3 GB 归档）：3 条真实泄漏违规；哈希全通过（23 条） |
| `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 graduate/ds_tasks/dataset/audit_dataset.py --skip-archive > graduate/ds_tasks/dataset/logs/audit_quick.log 2>&1` | **1** | 快速核查（跳过归档）：同上，哈希条目 17 |

日志与退出码落盘：`logs/audit.log`、`logs/audit_quick.log`、`logs/audit_exit.txt`、`logs/audit.json`（样本数、每条哈希、逐层尺度对照）。退出码语义：0 全通过 / 1 规则违规 / 2 基础设施错误。

## 3. 未证事项

1. 只做数据来源/划分/重叠/泄漏审计（软件参考层级），**不证明** RTL 逐位、E203 SoC、综合、官方 600 条验收。
2. manifest 只记录**张量内容哈希**，不记录 `.npy` **容器哈希**，故“容器字节是否与生成时逐字节相同”无法校验（容器哈希已登记，无外部期望值可比）；容器差异与内容一致性无关。
3. 无法在本目录验证 60 条真实音频的**说话人身份**是否与训练集分离，只复现官方文件名哈希规则。
4. `_silence_` 合成静音是否与组委会官方静音流程完全一致未证。
5. 未比对 Speech Commands V2 归档的官方发布 SHA-256（题目未给期望值）。
6. 第 1–11 层标定未核查（现有 RTL 只标定第 0 层）。
7. 修复建议（重定标用训练样本、哈希语义显式化、隔离 testing 行）仅为建议，本任务未改编码器文件；manifest 哈希无需改值（内容哈希 12/12 一致）。本任务只改了 `graduate/ds_tasks/dataset/` 内文件与本报告：`audit_dataset.py`（哈希种类分离）、`REPORT.md`、`logs/`（实跑重生成）。
