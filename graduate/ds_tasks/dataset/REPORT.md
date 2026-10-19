# 数据集核查报告：训练/验证划分、哈希、样本重叠、校准泄漏

- 任务目录：`graduate/ds_tasks/dataset/`
- 核查脚本：[`audit_dataset.py`](audit_dataset.py)
- 机器可读结果：[`logs/audit.json`](logs/audit.json)；运行日志：[`logs/audit.log`](logs/audit.log)、[`logs/audit_quick.log`](logs/audit_quick.log)；退出码：[`logs/audit_exit.txt`](logs/audit_exit.txt)
- 规则依据：题目截图 [`题目/640.jpg 5`](../../../题目/640.jpg%205) 第 3 条（无梯度标定、不重训练）、[`题目/640.jpg 7`](../../../题目/640.jpg%207) 第 1–5 条（Speech Commands V2；训练划分标定、验证划分功能/精度、测试划分不得调参）、[`题目/640.jpg 8`](../../../题目/640.jpg%208) 第 5 条（组委会另选 600 条评测集，不发布校准/验证包）。
- 边界：只在本独立 worktree 内操作；未开子智能体；未改其它 worktree / `main` / `hardware/`；只读取主工程 `/home/yang/Desktop/ic设计竞赛` 的 gitignored 数据集与缓存；未提交大数据、密钥或缓存。

## 0. 结论摘要

| 核查项 | 结论 | 说明 |
| --- | --- | --- |
| 训练/验证划分标签 | **通过** | 5 个缓存 + 6 条真实样本共 1920+6 条，按官方 SHA1 说话人哈希规则重算，真实 WAV 标签 0 处不一致；合成静音按划分单独生成，不计入 |
| 哈希 | **通过** | 权重、flags、数据归档、A12W10 参数包、6 条 WAV 全部一致；6 条 `mfcc/encoder .npy` 的**张量内容哈希 12/12 与 manifest 记录一致**（§3）。manifest 记录的是 `np.load(f).tobytes()` 内容哈希，`.npy` **文件容器哈希另记存档、不与前者比较** |
| 样本重叠 | **通过** | 各缓存 train/val/test 路径两两 0 重叠；跨缓存 0 重叠；真实样本特征张量 0 重复 |
| 现有校准泄漏 | **发现 2 处** | ① 单层编码器的注意力/PostNorm 静态尺度由**验证划分样本 `sample_00`** 求得；② 早期缓存 `quant_samples.npz` 仍含 20 条**测试划分**样本，曾被开发调试看过（已在 `results/README.md` 自述） |

脚本整体判定 `VERDICT FAIL (rule violations)`，退出码 **1**（检测到规则违规），共 **3 条**违规（§5）：全部为真实存在的样本标定泄漏与旧测试缓存，**无哈希类误报**。

## 1. 证据层级标注

| 层级 | 本报告状态 | 依据 |
| --- | --- | --- |
| **软件参考（CPU NumPy）** | **本目录全部结论** | `audit_dataset.py` 实跑；`logs/audit.json` |
| **RTL 逐位** | 未在本目录产生；仅引用既有 | grad-encoder `WORKER_REPORT.md` 的 `TB_PASS`/`VERIFY_PASS`（Verilator，`cycles=4,700,560`）。其激励/尺度来自**被污染的验证样本**，故该逐位证据继承泄漏（§4.2） |
| **E203** | 未证 | 无真实 E203 指令流/System Bus/NICE；仅有 E203-like 事务模型 |
| **综合** | 未证 | 无目标 PDK/SDC/150 MHz STA/面积 |
| **官方验收** | 未证 | 组委会 600 条评测集未公布，不发布校准/验证包 |

> 数据集核查属于**软件参考/数据来源**层级，不能推导出 RTL 逐位、E203、综合或官方验收结论。

## 2. 训练/验证划分核查（通过）

官方规则（指定提交 `kws_streaming/data/input_data.py:which_set`）：对文件名去掉 `_nohash_.*` 后取 SHA1，映射到 `[0,100)`，`<10` 为 validation、`<20` 为 testing、其余 training。`flags.json` 为 `validation_percentage=10`、`testing_percentage=10`。脚本逐条重算并比对缓存中的 `partitions` 字段：

| 缓存（主工程 `software/.build/`） | 总数 | training | validation | testing | 真实 WAV 标签不一致 | 合成静音（有意的划分内生成） |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `quant_samples.npz` | 40 | 20 | 0 | 20 | 0 | 0 |
| `quant_samples_10.npz` | 200 | 100 | 100 | 0 | 0 | 0 |
| `quant_samples_12labels_10.npz` | 240 | 120 | 120 | 0 | 0 | 10 / 10 |
| `quant_samples_12labels_20.npz` | 480 | 240 | 240 | 0 | 0 | 20 / 20 |
| `quant_samples_12labels_40.npz` | 960 | 480 | 480 | 0 | 0 | 40 / 40 |
| 真实样本 manifest | 6 | 3 | 3 | 0 | 0 | 0 |

- `_silence_/*.wav` 是脚本按划分生成的 1 秒全零波形，不是 Speech Commands 归档成员，官方哈希规则对其不适用；它们与所属划分一致（题目规定静音按官方流程在各划分内生成）。
- 合成静音在训练与验证划分下**逐字节相同**（全零），审计将其排除在“重复张量”之外，因为它按设计属于两个划分。

## 3. 哈希核查（通过）

哈希分**两类**，分别计算、分别记录、**互不比较**：

- **文件容器哈希** = 整个文件字节的 SHA-256（`.npz`/`.json`/`.tar.gz`/`.wav`）。
- **张量内容哈希** = `np.load(file).tobytes()` 的 SHA-256（`.npy`）。这是 manifest 记录的种类：`software/validate_real_input_chain.py` 第 93/95 行把 `mfcc_sha256`/`encoder_sha256` 定义为 `hashlib.sha256(array.tobytes())`。

| 对象 | 哈希种类 | 结果 |
| --- | --- | --- |
| `software/weights/kwt1_fp32.npz` | 容器 | 一致（`8f359abf…825a`，与 manifest、A12W10 JSON 双重一致） |
| `flags.json` | 容器 | 一致（`fcd480d3…4c58`） |
| `speech_commands_v0.02.tar.gz` | 容器 | 记录 2,428,923,189 字节，SHA-256 `af14739e…ff58`，105,878 个成员 / 105,835 个 WAV，可完整流式读取；题目未给外部期望值，只登记 |
| 6 条真实 WAV（归档内） | 容器 | **全部一致**，与 manifest `wav_sha256` 逐字节吻合 |
| `software/results/ffn1_a12w10_static_sym.npz` | 容器 | 一致（`a77f60fb…918e`，与 A12W10 JSON 的 `parameter_sha256` 一致） |
| 6 条 `sample_0X_mfcc.npy` / `sample_0X_encoder.npy` | **张量内容** | **12/12 全部一致**，与 manifest `mfcc_sha256`/`encoder_sha256` 逐条相等 |

> **误报更正（哈希种类混淆）**：早期审计用 `sha256(文件字节)` 去比对 manifest 的 `mfcc_sha256`/`encoder_sha256`，报出 12 处“不一致”。这是**量纲不同的误报**——`.npy` 容器额外包含 `\x93NUMPY` 魔数与形状/dtype/版本头，同一张量被不同 NumPy 版本重存后容器哈希即变，而内容哈希不变。现脚本对每条样本同时输出 `tensor_content` 与 `file_container` 两个字段（`logs/audit.json` → `hashes.items.sample_XX_{mfcc,encoder}`，日志中逐行打印），**只用内容哈希对内容哈希比较**；容器哈希仅存档登记，manifest 对其没有期望值（`hashes.hash_kinds` 字段写明该约定）。

第二重独立证据（内容一致性）：用只读的 `software/offline_encoder_input.py` 的 40→64 投影，从当前 `mfcc.npy` 重算 `encoder.npy`，最大绝对误差 **0.0**，且不超过 manifest 自记的 `encoder_independent_max_abs_error`（`4.77e-07…9.54e-07`）。即已提交的 `.npy` 张量内容与生成时的独立校验**完全一致**。

## 4. 样本重叠核查（通过）

- 每个缓存内 training/validation/testing 路径两两交集为 **0**；`quant_samples.npz` 的 train/test 亦为 0。
- 训练与验证的 `98×40` MFCC 特征张量（排除合成静音）无一条重复。
- 跨缓存：单层编码器使用的 `sample_00`（`four/750e3e75_nohash_0.wav`）在**任何缓存的训练划分中都不出现**——它只属于**验证划分**。这本身是泄漏来源（§5 ①）。

## 5. 校准泄漏核查（发现 2 处违规）

### 5.1 FFN1 量化链（干净）

- `ffn1_a12w10_static_sym.json`：`validation_used_for_calibration=False`，`training_count=240`、`validation_count=240`。
- `ffn1_blackbox_a11w10.json`：训练/验证路径各 240 条且**不相交**；搜索目标只用训练 NRMSE；12 层选中的倍率**全为 `(1.0,1.0)`**（未利用验证改进）。
- 复算确认：`s_x`（FFN1 输入尺度）、`sx`（PostNorm `h` 尺度）在 `sample_00`(验证) 与 `sample_01`(训练) 下**完全相同**（rel_diff = 0.0），说明 FFN1 参数包确实与样本无关、来自训练划分。

### 5.2 单层编码器数值配置（**违规**：用验证样本求尺度）

`graduate/encoder/vectors/gen_vectors.py` 硬编码
`SAMPLE = software/validation/real_input_chain_20261017/sample_00_encoder.npy`，而 manifest 明确 `sample_00` 的划分是 **validation**。运行链：
`attn_int_model.py --sample 0`（注意力全部静态尺度）与 `kwt_postnorm_ref.calibrate()`（PostNorm 尺度/LUT/输出尺度）。用**训练划分的 `sample_01`** 重跑同一代码，冻结配置发生系统性变化：

| 系数 | 验证 `sample_00` | 训练 `sample_01` | 相对差 / 绝对差 |
| --- | ---: | ---: | ---: |
| 注意力 `scale_x`（输入尺度） | 1.356254e-03 | 1.629903e-03 | **16.79%** |
| 注意力 `scale_o`（输出尺度） | 2.475293e-05 | 2.974730e-05 | **16.79%** |
| 注意力 `KC`（softmax 常量） | 6680 | 4824 | 差 1856 |
| PostNorm `s_x` | 1.356254e-03 | 1.629903e-03 | **16.79%** |
| PostNorm `s_a`（注意力输出重定标） | 2.712451e-04 | 3.246386e-04 | **16.45%** |
| PostNorm `s_r1` | 1.449935e-03 | 1.742839e-03 | **16.81%** |
| PostNorm `s_out` | 1.831582e-03 | 1.868519e-03 | 1.98% |
| PostNorm `eps_q1` | 31173 | 21576 | 差 9597 |
| PostNorm `mgn`（GELU 归一化常量） | 267644 | 264106 | 差 3538 |

即：RTL 激励/参数的**量化尺度与 requant 常量随样本变化**，当前取的是验证样本，直接违反“验证划分仅评估、不得用验证样本求尺度/阈值”。该污染沿证据链传播到 grad-encoder 的“RTL 逐位”结论（逐位一致只证明 RTL 与**同一被污染的整数模型**一致，不代表合规标定）。

### 5.3 测试划分污染（**违规**：曾被开发调试看过）

- 迁移过程留下的 `software/.build/quant_samples.npz` 仍含 **20 条 testing 划分**样本。
- `software/results/README.md` 自述：开发中曾用少量测试划分音频试跑，发现规则限制后正式结果改用训练标定/验证比较，但“那些已看过的测试音频不能再作为完全未触碰的最终保留集”。
- 影响：若组委会 600 条评测集按官方 `which_set` 选取，这 20 条中的对应该类样本已非“未见”；虽未进入最终冻结参数（A12W10/blackbox 均只用 training/validation），但应作**已知污染**登记并避免在最终结论中当作盲测。

## 6. 复现命令与退出码

在仓库根 `/home/yang/kwt-worktrees/ds-grad-dataset` 执行：

```bash
# 完整核查（含 2.3 GB 归档哈希与流式读取）
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python3 graduate/ds_tasks/dataset/audit_dataset.py > graduate/ds_tasks/dataset/logs/audit.log 2>&1
echo "EXIT=$?"   # -> EXIT=1（检测到规则违规）

# 快速核查（跳过归档扫描）
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python3 graduate/ds_tasks/dataset/audit_dataset.py --skip-archive \
  > graduate/ds_tasks/dataset/logs/audit_quick.log 2>&1
echo "EXIT=$?"   # -> EXIT=1
```

| 命令 | 退出码 | 说明 |
| --- | ---: | --- |
| `python3 audit_dataset.py` | **1** | 划分 / 哈希（含 12/12 张量内容哈希）/ 重叠 / 内容一致性全部通过；剩 3 条真实泄漏违规（§0、§5） |
| `python3 audit_dataset.py --skip-archive` | **1** | 同上，不含归档 SHA-256（哈希条目 23 → 17） |

退出码语义：`0` 全部通过；`1` 检出规则违规；`2` 基础设施错误。退出码 **1 是本任务要登记的发现**，不是脚本故障。`logs/_tmp_calibration/` 是标定复算的运行期 scratch 目录（每次运行重新生成，不提交）。

## 7. 建议修复（不改本目录外文件，供工程侧执行）

1. **编码器标定改用训练划分样本**：把 `graduate/encoder/vectors/gen_vectors.py` 的 `SAMPLE` 换成 training 划分样本（如 `sample_01`），或在 `attn_int_model.py`/`kwt_postnorm_ref.calibrate()` 增加“训练标定 + 验证评估”两段式入口；重跑 RTL 逐位，并在验证划分上评估，禁止用 `sample_00` 求任何尺度。
2. **哈希语义显式化（加固，非缺陷）**：manifest 的 `mfcc_sha256`/`encoder_sha256` 是张量内容哈希（`array.tobytes()`），当前 12/12 一致，**无需改值**；建议在 `manifest.json` 增加 `"hash_kind": "tensor-content"` 字段并在校验脚本入口打印，避免下游再拿 `.npy` 文件哈希去比对而误报。（本任务不改 `software/` 内任何文件。）
3. **隔离测试划分**：删除/迁移 `quant_samples.npz` 的 testing 行，或改名并在 `.build` 层面禁止其进入标定；在脚本入口把 `partitions` 含 `testing` 直接拒绝（`export_ffn1_quantized.py` 已有此检查，`quantize_ffn1.py` 尚未强制）。

## 8. 未证事项

1. 本报告只做**数据来源/划分/重叠/泄漏**审计（软件参考层级）；不证明 RTL 逐位、E203 SoC、综合、官方 600 条验收。
2. manifest 只记录**张量内容哈希**，不记录 `.npy` **容器哈希**；因此“容器字节是否与生成时逐字节相同”无法校验（本次已把容器哈希登记入 `logs/audit.json`，但无外部期望值可比）。容器差异与张量内容一致性无关，不影响任何结论。
3. 60 条真实音频的**说话人身份**无法在本目录验证是否与训练说话人集合分离；审计只复现官方文件名哈希规则。
4. `_silence_` 合成静音是否与组委会官方静音生成完全一致未证（题目规定按官方评估流程生成 1 秒全零波形）。
5. 未下载/比对 Speech Commands V2 归档的官方发布 SHA-256（题目未提供期望值）。
6. 未审计 12 层其它层的标定（现有 RTL 只标定第 0 层），第 1–11 层的尺度/阈值来源仍待逐层核查。
