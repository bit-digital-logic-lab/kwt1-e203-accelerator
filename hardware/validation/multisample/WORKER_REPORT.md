# WORKER_REPORT：A12W10 多验证样本 E203 NICE-like 抽检

- 分支：`codex/kwt-multisample`（基于 `fcb4b72` 的 ASCII 路径修复）
- 日期：2026-10-18
- 证据层级：**Icarus 行为仿真下的 NICE-like E203 + 本仓库仿真总线/SRAM 模型**；不是组委会正式 SoC/PDK，不是综合网表，不是 150 MHz 时序、面积、功耗或吞吐结论。
- 工具：Icarus Verilog/vvp 14.0（oss-cad-suite）、RISC-V GCC 11.4.0、Python 3.12.12、NumPy 2.1.3、PyTorch 2.5.1+cu121（向量生成强制 `--device cpu`）。

## 1. 结论

- 选了 **3 条不同类别的验证划分样本**：索引 0（`_unknown_`）、40（`down`）、440（`_silence_`）。每条用冻结 A12W10 参数生成完整 12 层真实向量，共 **36 层 / 144 个 hex 文件**。
- 独立检查器不导入生成器，重新解析全部 hex、核对 manifest/文件 SHA、查 X/W/bias/Y 形状和行数，并用 `a16`、`w16`、`bias32` 重建 `qx @ qw + bias`，与 `expected32` 逐位比较：**36/36 层完全一致，0 个差异元素**。
- E203 实际运行：3 条样本的 `layer01/layer06/layer12 × tile0/tile207` 全部覆盖；另用 12 层轮转把 **layer01–layer12 每层至少抽到一次**。共 **36 条 layer-tile PASS**，全部退出码 0。
- 资源允许时追加了第 4 条样本（索引 80，`go`）的 `layer01` **整层 208 tiles**：**PASS**，`outputs=25344`、`padding=1280`，退出码 0。
- 本次 E203 合计 **244 条 sample-layer-tile 实测 PASS 记录**（36 + 208），有效输出 **28,512**、填充 **2,720**；没有把未运行的组合写成 PASS。
- 没有启动 `3×12×208` 全矩阵大仿真。

## 2. 统计口径（严格分开，不混写）

| 证据类别 | 样本数 | 层覆盖 | 本次实际运行范围 | 输出/结论 |
| --- | ---: | --- | --- | --- |
| **本次 E203 NICE-like** | 3 条主样本 + 1 条追加样本 | 主样本各 12 层向量；E203 覆盖 12 层轮转 | 36 条 layer-tile + 1 条整层（208 tiles） | 244 条 PASS；有效输出 28,512；填充 2,720 |
| 已有 RTL 240 样本汇总 | 240 条验证样本 | 12 层 | **本次未重跑** | 已有 `quantized_validation240_a12w10_rtl_summary.json`：每层 6,082,560 输出，12 层合计 72,990,720 |
| 软件样本缓存 | 960 条（480 training + 480 validation） | 12 层 | 本次只取 4 条 validation（索引 0/40/440/80） | 不把 480 验证集说成已跑 E203 |
| 官方 600 条评测 | 600 条 | 12 层 | **未运行** | 无结论 |

软件缓存的划分来自 `quant_samples_12labels_40.npz` 的 `partitions` 字段；参数元数据明确 `validation_used_for_calibration: false`。本次没有用验证样本调参。

## 3. 样本与 SHA-256

三条主样本均来自 validation 划分；`sample_feature_sha256` 是对该样本 `features[98,40]` 原始字节的 SHA-256。

| 验证索引 | 标签 | 样本路径 | feature SHA-256 | manifest SHA-256 | 48 个 hex 合并 SHA-256 |
| ---: | --- | --- | --- | --- | --- |
| 0 | `_unknown_` | `./four/750e3e75_nohash_0.wav` | `8bd524752fcc7bd99e17b4634e089361f612be06891f0eeb62434546a3dad3d1` | `465fef9af8e2b94cc797e26b816344eb744a529d3bfd2c392c55608a81998284` | `8bec2f7c67cfd9abd4dd655952c71f160ecf760f3e85df12bd20b0262bb46e36` |
| 40 | `down` | `./down/c2d15ea5_nohash_0.wav` | `6ae1009912d0b5892c57a51917207b86e52ba9affed759f0374871fde60dd415` | `cb26ce6bba56f6b3fa4ed6d710b1e31a5b13077016508cc3e6dc4d1326275db9` | `22e0f67305aa59e6009c92487627f3506634c343090de8182fd2cf03fb354b55` |
| 440 | `_silence_` | `_silence_/validation_zero_000.wav` | `22bbd8bea1c4bc6822f8f644494b6bda7c67f30cf5472894d7c4ce421b63b2b4` | `ef6f97b22c5923c86eee485682cdcbf7a698536699d483033d0bb3124d4adc22` | `66fcb0ec3d821a2427be423155c9439f1aa15e13915a05a3b34c544ae0b9ff35` |
| 80（追加） | `go` | `./go/3de53d23_nohash_0.wav` | `2bd748ea32c5db4167907a3fca468dfc329caf99440c534b5e4bb7afcbfed9a1` | `6523be7bbb03ab790e18cc938585c2b672ca2571d7a722ab69b5dbca5af1f85b` | 仅 layer01（`expected32`=`13c5fdd602a3b21577d8d9785b07dcc143408f2c6f3a4c73008ad1cdc36ce300`） |

> 索引 80 是追加整层样本，只生成并运行了 `layer01`；它的 feature SHA 由同一条缓存计算，未参与三条主样本的 12 层独立检查统计。

冻结共享输入的 SHA-256（只读，未修改）：

| 输入 | SHA-256 |
| --- | --- |
| `software/.build/quant_samples_12labels_40.npz` | `43d639fb64ea8d907db967b071010e41e68827368532880ba8a99b92769bbae8` |
| `software/results/ffn1_a12w10_static_sym_12labels_40.npz` | `10c481c7a79d0ee8063ad807f8c0f685769c2c8e1dbcdb5f33079919dfeef367` |
| `software/results/ffn1_a12w10_static_sym_12labels_40.json` | `99cfe56106530814a84215f558d0cfe55b93035d83e765dd351bc6a38c2eb039`（由 manifest 记录并复核） |
| `software/weights/kwt1_fp32.npz` | `8f359abf7adbe287b8f89fca98af46194f98f90f2146784731bff551a155825a` |

逐文件、逐层 SHA 已保存在 `results/vector_checks.json`；每条样本 48 个 hex 的完整列表也在其中，报告不重复粘贴。“48 个 hex 合并 SHA” 的定义是：按文件名排序后，把每行 `文件名:sha256` 拼接再取 SHA-256。

## 4. 向量独立检查结果

检查脚本：`hardware/validation/multisample/check_vectors.py`。它不导入 `prepare_real_e203_vectors.py`，从 hex 重新构造数据。

| 检查项 | 结果 |
| --- | --- |
| manifest SHA / 源归档 SHA | 3/3 样本一致 |
| hex 文件 SHA 与 manifest 对照 | 144/144 一致 |
| 行数 | a16 6,656；w16 16,384；bias32 256；expected32 25,344，全部一致 |
| X / W / bias / Y 形状 | 99×64 / 64×256 / 256 / 99×256，3/3 一致 |
| 最后一组 A tile 的 5 行零填充 | 36/36 层通过 |
| A12/W10 范围 | 36/36 层未越界 |
| 独立重建 `qx @ qw + bias` 对 `expected32` | **36/36 层逐位一致，0 个差异元素** |
| 每样本输出量 | 12×25,344 = **304,128** 个有效输出 |
| INT32 溢出 | 无；本次最大 `max_abs_expected_int32` 为 4,593,325（索引 0），保守界最大 67,780,536 |

生成侧 NRMSE（仅软件 FP64 解码参考，不是本次 E203 实测误差）：

| 验证索引 | 标签 | 12 层 NRMSE 范围 | 最差层 |
| ---: | --- | --- | --- |
| 0 | `_unknown_` | 0.101708% – 0.134930% | layer11，0.134930% |
| 40 | `down` | 0.103312% – 0.146047% | layer12，0.146047% |
| 440 | `_silence_` | 0.123314% – 0.176703% | layer11，0.176703% |

## 5. 12 层轮转抽检矩阵

轮转设计（每条样本都必须包含 `layer01/06/12`）：

- 索引 0：主轮转 `1,4,7,10`，加必测 `6,12` → 实际 `1,4,6,7,10,12`
- 索引 40：主轮转 `2,5,8,11`，加必测 `1,6,12` → 实际 `1,2,5,6,8,11,12`
- 索引 440：主轮转 `3,6,9,12`，加必测 `1` → 实际 `1,3,6,9,12`

这样 layer01–layer12 每层至少被一条样本覆盖，同时每条样本都覆盖 layer01/06/12 的 tile0 与 tile207。

| 验证索引 | 标签 | 层 | 主轮转 | tile0 | tile207 |
| ---: | --- | ---: | :---: | :---: | :---: |
| 0 | `_unknown_` | 1 | true | PASS | PASS |
| 0 | `_unknown_` | 2 | false | not_run | not_run |
| 0 | `_unknown_` | 3 | false | not_run | not_run |
| 0 | `_unknown_` | 4 | true | PASS | PASS |
| 0 | `_unknown_` | 5 | false | not_run | not_run |
| 0 | `_unknown_` | 6 | false | PASS | PASS |
| 0 | `_unknown_` | 7 | true | PASS | PASS |
| 0 | `_unknown_` | 8 | false | not_run | not_run |
| 0 | `_unknown_` | 9 | false | not_run | not_run |
| 0 | `_unknown_` | 10 | true | PASS | PASS |
| 0 | `_unknown_` | 11 | false | not_run | not_run |
| 0 | `_unknown_` | 12 | false | PASS | PASS |
| 40 | `down` | 1 | false | PASS | PASS |
| 40 | `down` | 2 | true | PASS | PASS |
| 40 | `down` | 3 | false | not_run | not_run |
| 40 | `down` | 4 | false | not_run | not_run |
| 40 | `down` | 5 | true | PASS | PASS |
| 40 | `down` | 6 | false | PASS | PASS |
| 40 | `down` | 7 | false | not_run | not_run |
| 40 | `down` | 8 | true | PASS | PASS |
| 40 | `down` | 9 | false | not_run | not_run |
| 40 | `down` | 10 | false | not_run | not_run |
| 40 | `down` | 11 | true | PASS | PASS |
| 40 | `down` | 12 | false | PASS | PASS |
| 440 | `_silence_` | 1 | false | PASS | PASS |
| 440 | `_silence_` | 2 | false | not_run | not_run |
| 440 | `_silence_` | 3 | true | PASS | PASS |
| 440 | `_silence_` | 4 | false | not_run | not_run |
| 440 | `_silence_` | 5 | false | not_run | not_run |
| 440 | `_silence_` | 6 | true | PASS | PASS |
| 440 | `_silence_` | 7 | false | not_run | not_run |
| 440 | `_silence_` | 8 | false | not_run | not_run |
| 440 | `_silence_` | 9 | true | PASS | PASS |
| 440 | `_silence_` | 10 | false | not_run | not_run |
| 440 | `_silence_` | 11 | false | not_run | not_run |
| 440 | `_silence_` | 12 | true | PASS | PASS |

`not_run` 明确表示未运行，不是 PASS。12 层中每层的至少一个 tile 组合已实际运行。

## 6. 实际 E203 运行与退出码

每条样本把轮转选中的层做成只含这些 `layer??_*.hex` 的 ASCII 临时 CASE_ROOT；`run_e203.sh --real-layers <dir> <tile> 1` 一次编译、顺序跑该样本选中的层。未修改 `run_e203.sh`、数值核心或共享样本。

| 运行 | 样本 | tile | 层数 | 退出码 | PASS 行 | 有效输出 | 填充 | 任务周期 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 6 | **0** | 6 | 768 | 0 | 4,025/层 |
| 2 | 0 | 207 | 6 | **0** | 6 | 288 | 480 | 4,025/层 |
| 3 | 40 | 0 | 7 | **0** | 7 | 896 | 0 | 4,025/层 |
| 4 | 40 | 207 | 7 | **0** | 7 | 336 | 560 | 4,025/层 |
| 5 | 440 | 0 | 5 | **0** | 5 | 640 | 0 | 4,025/层 |
| 6 | 440 | 207 | 5 | **0** | 5 | 240 | 400 | 4,025/层 |
| 7（追加） | 80，layer01 | 0..207 | 1 层 | **0** | 1 | 25,344 | 1,280 | 838,028 |

- tile0 每次 `outputs=128, padding=0`；tile207 每次 `outputs=48, padding=80`（最后 5 行×16 列填充），与 TB 的 `check_tile` 期望一致。
- 每条 PASS 行均来自 DUT 写回后对 `expected32` 的逐 32-bit 比较；日志中没有 `FATAL`、`mismatch` 或 X 输出。
- 追加整层 208 tiles 的日志末尾为：
  `PASS ... start=0 count=208 outputs=25344 padding=1280 task_cycles=838028 total_cycles=838041`。
- 首次失败即停止的逻辑在调度器中实现；本次没有数值失败。曾有一次调度器因日志正则缺少 `MULTILINE` 误报 0/6 PASS，E203 进程本身退出码均为 0；修正解析后重跑，最终证据以上表为准。

## 7. 与已有 validation0 ASCII 结果的对照（重要）

已有主目录结果 `real_e203_validation0_vectors` 覆盖同一验证索引 0、同一 12 层。本次重新生成的 48 个 hex 中：

- **42/48 文件字节一致**；
- `layer06/08/09` 的 `a16` 和 `expected32` 不同；每层仅 **1 个 qx 元素相差 1 LSB**，导致 255–256 个输出相差 ±511；
- 在同一环境立即重跑生成，48/48 完全一致，说明本次生成在本地是确定性的；差异来自已有向量的生成环境/版本（同一脚本、同一归档与参数 SHA 均一致），属于量化阈值附近的 tie 敏感，而不是共享输入被改动。

因此本报告**不声称**与已有 validation0 向量逐字节复现；本次 E203 比较的是本次重新生成、且被独立 `a16/w16/bias` 重建验证过的 `expected32`。已有 validation0 的 12/12 PASS 仍只对它自己的向量成立，两个结果不混写。

## 8. 未证事项与边界

- 只对 tile0、tile207 做了跨层抽检；除追加样本 `layer01` 外，没有对 3 条主样本的所有 208 tiles 做全层仿真。
- 没有运行 `3×12×208` 全矩阵，也没有跑 12 层 × 3 样本的所有 tile。
- 没有在组委会正式 SoC/PDK、真实 NICE 接口或板卡上运行；RTL 是 Icarus 下的 NICE-like 模型，共享 RAM/总线/DMA 为仿真模型。
- 没有综合、布局布线、150 MHz 时序、面积、功耗、吞吐或能耗结论；`task_cycles` 只是单 tile/单层仿真观察值。
- 追加样本 80 的 `layer01` 未跑独立 `check_vectors.py` 的 GEMM 重建；它是冻结脚本生成并由 E203 对自己的 `expected32` 逐位比较。主统计的 36 层已做独立重建。
- NRMSE 来自生成侧软件解码，不是本次 E203 输出重新对 FP32 计算的误差；本次 E203 的误差是 INT32 逐位零误差。
- 未在本任务中执行 C 驱动或 RV32 驱动联调（属 driver 分支范围）。
- 官方 600 条评测未运行，不作任何精度或排名结论。

## 9. 复现与产物

脚本（只读共享输入，临时向量写 `/tmp/opencode/multisample_validation`）：

- `hardware/validation/multisample/run_multisample_e203.py`：生成 3×12 层向量、调用独立检查、按轮转矩阵调度 E203、解析逐条退出码/PASS；可选 `--extra-full-layer`。
- `hardware/validation/multisample/check_vectors.py`：独立 SHA/形状/行数/INT32 重建检查。
- `hardware/validation/multisample/results/commands.log`：实际命令与退出码记录。
- `hardware/validation/multisample/results/e203_results.json`：机器可读的样本、PASS 行、tile、退出码、日志 SHA。
- `hardware/validation/multisample/results/e203_matrix.tsv`：12 层轮转矩阵。
- `hardware/validation/multisample/results/vector_checks.json`：36 层逐文件 SHA 与独立重建结果。
- `hardware/validation/multisample/results/*.log`：生成、检查与 E203 原始日志（含日志 SHA 记录）。

复现入口：

```bash
/home/yang/Desktop/ic设计竞赛/software/.venv/bin/python \
  hardware/validation/multisample/run_multisample_e203.py \
  --skip-generate --extra-full-layer
```

首次生成时去掉 `--skip-generate`；不追加整层时去掉 `--extra-full-layer`（主矩阵约 97 s，追加整层约 1003 s）。

工作树只新增 `hardware/validation/multisample/` 下的脚本、报告和结果摘要；没有修改 `run_e203.sh`、RTL/数值核心、共享样本或主目录；没有 push。
