# 研究生组分阶段验证基础设施（graduate/verification）

本目录提供研究生组任务 4–7（注意力、单层编码器、十二层编码器、E203/SoC、综合）
的可复现、分阶段验证入口。它**只读**被测设计，本身不修改 RTL；同一套 runner 通过
`--design-root` 可指向本 worktree 或任何未来独立分支/worktree，便于逐分支验收。

> 判据底线：**文件存在不算 PASS**。只有证据命令真实执行、退出码为 0，且输出中出现
> 真正的比较结果（契约行或带计数的通过标记）才判 PASS；否则 FAIL。设计模块缺失的
> stage 判 SKIP，并列出缺失路径。

## 运行

```bash
# 列出 stage
python3 graduate/verification/run_stages.py --list

# 在“当前 worktree”跑全部 stage，证据写入 results/latest/
python3 graduate/verification/run_stages.py --design-root "$PWD"

# 只跑某几个 stage
python3 graduate/verification/run_stages.py --design-root /path/to/other-worktree \
    --stage attention --stage synth

# 单元测试 + 故障注入测试 + 全量 stage
bash graduate/verification/run_tests.sh            # SKIP 不视为失败
bash graduate/verification/run_tests.sh --strict   # SKIP 也返回非 0
```

出口码：有 FAIL → 1；`--strict` 且有 SKIP → 2；否则 0。

## 证据分级

| 级别 | 含义 | 对应 stage |
| --- | --- | --- |
| `software` | 软件数值模型的误差（NRMSE / 最大绝对误差），不是硬件证据 | `smoke_software_golden` |
| `rtl` | RTL 与**独立 golden** 的逐位/带容差比较 | `attention`、`encoder1`、`encoder12`、`smoke_rtl_vector`、`smoke_golden_faults` |
| `soc` | E203 CPU、总线、DMA 的集成 | `e203` |
| `synth` | 综合（elaboration、结构计数、示例库面积） | `synth` |

四类证据**不互相替代**：软件误差合格不能证明 RTL 逐位正确；综合网表面积不是时序签核。

## 阶段与契约

| stage | 类别 | 被判为 PASS 的条件 |
| --- | --- | --- |
| `attention` | rtl | `hardware/rtl/attention/tb/run.sh` 退出 0，输出 `KWT-GRAD-VERIFY attention checked=N failed=0`（N≥1），且无 `FAIL`/`MISMATCH` |
| `encoder1` | rtl | `hardware/rtl/encoder/tb/run_single_layer.sh`，契约同上（`encoder1`） |
| `encoder12` | rtl | `hardware/rtl/encoder/tb/run_twelve_layers.sh`，契约同上（`encoder12`） |
| `e203` | soc | `hardware/rtl/control/tb/run_e203.sh` 退出 0，输出 `PASS real E203 NICE: requests=… responses=… outputs=… cycles=…`，requests/responses 非零 |
| `synth` | synth | `synth/yosys_evidence.py` 退出 0 且输出 `KWT-GRAD-VERIFY synth checked=N failed=0` |
| `smoke_rtl_vector` | rtl | `hardware/testbenchplus/common/run_vectors.py` 输出 `result: N/N passed`（N≥1）且无 `FAIL`/`MISMATCH`/`NaN` |
| `smoke_golden_faults` | rtl | `negative_injection_check.py` 证明被污染的 golden 被检出（`detected with exit=非0`） |
| `smoke_software_golden` | software | `golden/check_encoder_input.py` 输出 `KWT-GRAD-VERIFY encoder_input checked=N failed=0` |

### 分支 harness 契约

未来在独立分支补齐 `attention` / `encoder1` / `encoder12` 时，harness 必须：

1. 从 `$KWT_GRAD_DESIGN_ROOT` 解析设计路径；
2. 用**自己独立生成的 golden**（不得从 DUT 输出反推）与被测 RTL 比较；
3. 最后打印一行 `KWT-GRAD-VERIFY <stage> checked=<比较项数> failed=<失败数>`；
4. 任何不通过都必须打印 `FAIL` 或 `MISMATCH` 并返回非零退出码。

模块存在但 harness 缺失，runner 判 **FAIL**（不是 SKIP）；模块与 harness 都缺失才判
SKIP。每个 stage 的输出日志保存在 `results/latest/<stage>.log`。

## 每项保留的证据

`results/latest/summary.json` 与 `summary.md` 为每个 stage 记录：实际命令、退出码、
耗时、输入文件 SHA-256、样本身份、比较方式、golden 来源、工具版本与输出尾部。
不使用“文件是否存在”作为通过条件。

## SKIP 语义（当前集成基线）

当前 worktree **尚未包含**注意力与编码器 RTL，因此：

* `attention`、`encoder1`、`encoder12` → SKIP，原因列出缺失模块路径；
* 若环境缺少可用 Icarus（需 ≥12；Icarus 11 无法正确展开本 RTL 的 unpacked array）、
  Yosys（需 ≥0.20）、`riscv64-linux-gnu-gcc` 或 `资料/e203_hbirdv2_full`，
  相应 stage → SKIP 并说明。

## 真实故障注入

* `faults/test_harness_faults.py`：证明“退出 0 但无输出”“`checked=0`”“`failed>0`”
  “比率不满”“出现 FAIL/MISMATCH”“模块在但 harness 缺失”都不会被误判为 PASS。
* `faults/test_golden_faults.py`：
  * 纯 Python 与 NumPy int64 两种独立整数 GEMM 在随机数据上必须逐位一致，并用手算
    字面向量固定算术；
  * 真实写盘故障：`check_encoder_input.py --corrupt-golden` 必须失败、
    `yosys_evidence.py --corrupt-mul-count` 必须失败；
  * 正常输入必须通过。

## 目录

```text
graduate/verification/
├── run_stages.py              # 分阶段 runner（--design-root / --stage / --out）
├── run_tests.sh               # 单元+故障注入测试，随后跑全部 stage
├── synth/yosys_evidence.py    # 综合证据（elaboration + 结构计数 + 可选示例库面积）
├── golden/
│   ├── ffn1_reference.py      # 独立整数 GEMM / NRMSE 参考
│   └── check_encoder_input.py # 软件 golden：design vs 独立 numpy vs 存储 TF golden
├── faults/
│   ├── test_harness_faults.py # runner 判定逻辑故障注入
│   └── test_golden_faults.py  # golden 与写盘故障注入
└── results/latest/            # 最近一次真实运行的证据（summary.json / summary.md / *.log）
```
