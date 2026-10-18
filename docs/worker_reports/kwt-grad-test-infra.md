# WORKER_REPORT — codex/kwt-grad-test-infra（研究生组分阶段验证基础设施）

- 分支：`codex/kwt-grad-test-infra`（worktree `/home/yang/kwt-worktrees/grad-test-infra`）
- 基线：`3d9ae98`（继承 `codex/kwt-integration`；`main@1494597` 的整合结果）
- 范围：研究生组任务 4–7 的**可复现分阶段验证 runner**，位于 `graduate/verification/`；
  支持 `--design-root` 指向其它独立分支/worktree；每项保留实际命令、退出码、耗时、
  输入 SHA-256、样本身份、比较方式与 golden 来源；显式 PASS/SKIP/FAIL，**不以文件存在判 PASS**。
- 边界：只改 `graduate/verification/` 与根 `WORKER_REPORT.md`；未开子智能体；未碰主目录或
  他人分支；未 `merge`/`rebase`/`cherry-pick`/`push`；未改任何 RTL/数值模型。

## 交付

| 文件 | 作用 |
| --- | --- |
| `graduate/verification/run_stages.py` | 分阶段 runner：`--design-root` / `--stage` / `--out` / `--list` / `--strict` |
| `graduate/verification/run_tests.sh` | 先跑 17 个单元+故障注入测试，再跑全部 stage |
| `graduate/verification/golden/ffn1_reference.py` | 独立整数 GEMM（纯 Python 与 NumPy int64 两实现）+ NRMSE |
| `graduate/verification/golden/check_encoder_input.py` | 软件 golden：design convert vs 独立 numpy vs 存储 TF golden |
| `graduate/verification/synth/yosys_evidence.py` | 综合证据：elaboration + `check -assert` + 结构计数 + 可选示例库面积 |
| `graduate/verification/faults/test_harness_faults.py` | 判定逻辑故障注入（10 例） |
| `graduate/verification/faults/test_golden_faults.py` | golden 与写盘故障注入（7 例） |
| `graduate/verification/README.md` | 用法、stage 契约、证据分级、SKIP 语义 |
| `graduate/verification/results/latest/` | 本次真实运行证据：`summary.json` / `summary.md` / 各 stage `.log` |

## 判定口径

- **PASS 的必要条件**：证据命令真实执行、退出码 0，且输出含有真正的比较结果
  （分支 harness 的 `KWT-GRAD-VERIFY <stage> checked=N failed=0` 契约行，或带计数的通过标记），
  并保证 `checked ≥ 1`。退出 0 但无比较输出 → FAIL。
- **SKIP**：设计模块缺失（列出缺失路径），或环境缺工具（Icarus ≥12 / Yosys ≥0.20 /
  `riscv64-linux-gnu-gcc` / `资料/e203_hbirdv2_full`）。
- **FAIL**：模块存在但缺 harness；或证据命令非 0 / 出现 `FAIL`、`MISMATCH` / `checked=0` / `failed>0`。

## 实跑结果（本次，设计 commit `3d9ae98`）

命令：`bash graduate/verification/run_tests.sh` → 退出码 **0**，`PASS=5 SKIP=3 FAIL=0`。

| stage | 类别 | 判定 | 实际命令 | 退出码 | 耗时 s |
| --- | --- | --- | --- | ---: | ---: |
| attention | rtl | **SKIP** | 期望 `hardware/rtl/attention/tb/run.sh` | — | — |
| encoder1 | rtl | **SKIP** | 期望 `hardware/rtl/encoder/tb/run_single_layer.sh` | — | — |
| encoder12 | rtl | **SKIP** | 期望 `hardware/rtl/encoder/tb/run_twelve_layers.sh` | — | — |
| e203 | soc | **PASS** | `bash hardware/rtl/control/tb/run_e203.sh` | 0 | 0.955 |
| synth | synth | **PASS** | `python3 graduate/verification/synth/yosys_evidence.py --design-root …` | 0 | 9.282 |
| smoke_rtl_vector | rtl | **PASS** | `python3 hardware/testbenchplus/common/run_vectors.py` | 0 | 1.602 |
| smoke_golden_faults | rtl | **PASS** | `python3 hardware/testbenchplus/common/negative_injection_check.py` | 0 | 3.301 |
| smoke_software_golden | software | **PASS** | `python3 graduate/verification/golden/check_encoder_input.py --design-root …` | 0 | 0.109 |

工具（记录在 `summary.json`）：Icarus `14.0`（oss-cad-suite，**非**系统 Icarus 11）；
Yosys `0.63+97`；`riscv64-linux-gnu-gcc`；Python `/usr/bin/python3`。

### 关键实跑数值

- `e203`：`PASS real E203 NICE: requests=1 responses=1 outputs=128 cycles=1055`（简化 SoC 模型，
  128 个输出由 TB 对照冻结 tile golden）。
- `synth`：`$mul=8 $mem_v2=8 $add=9`，`check -assert` 0 problems；`kwt_cacc` 用 Nangate45
  示例库映射 `Chip area=47861.114`（**仅为示例库相对面积，不是签核**）。
- `smoke_rtl_vector`：`layer0_int16` 逐位 `6/6`，`layer0_fp32` NRMSE `0.00000000%`，`result: 2/2 passed`。
- `smoke_software_golden`：`converted_vs_independent=0`；`converted_vs_golden=4.7684e-07`；
  `independent_vs_golden=4.7684e-07`；NRMSE `7.84e-06%`；`checked=19008`（99×64×3 项比较）。

## 独立 golden 与故障注入（真实执行）

- 单元+故障注入：`python3 -m unittest discover -s graduate/verification/faults` → **17 tests OK**（退出码 0）。
- 判定逻辑注入：退出 0 空输出、`checked=0`、`failed>0`、错误 stage 名、`1/2 passed`、出现
  `FAIL`/`MISMATCH`、模块在但 harness 缺失 —— 均被正确判为 FAIL（真契约判 PASS）。
- 写盘故障注入：
  - `check_encoder_input.py --corrupt-golden`（存储 golden +1.0）→ 退出码 **1**，`FAIL`；
  - `yosys_evidence.py --corrupt-mul-count`（要求乘法器数远超实际）→ 退出码 **1**，`FAIL`；
  - 对应正常输入 → 退出码 0。
- 独立整数 golden：纯 Python GEMM 与 NumPy int64 GEMM 在 25 组随机矩阵上逐位一致；
  手算向量 `[[1,2,3],[4,5,6]]×[[1,0],[0,1],[1,1]]+[10,-10]=[[14,-5],[20,1]]` 固定。

## SKIP 原因（如实）

1. 本 worktree 尚无注意力/编码器 RTL：`hardware/rtl/attention/**`、`hardware/rtl/encoder/**`
   不存在 → `attention`、`encoder1`、`encoder12` SKIP 并列出缺失路径。
2. 这些分支应提供 `tb/run.sh` harness，并在末尾打印
   `KWT-GRAD-VERIFY <stage> checked=N failed=0`；模块存在而 harness 缺失时 runner 判 FAIL。
3. 环境边界：系统自带的 Icarus 11 会错误展开本 RTL 的 unpacked array（int16 向量出现假失配、
   fp32 挂起）；runner 自动选用 oss-cad-suite 的 Icarus ≥12，否则 SKIP 并说明。

## 未证事项 / 诚实说明

1. `e203` 是**简化 SoC 模型**（行为级总线/存储），不是组委会专用平台；不能当 150 MHz、
   面积或功耗证据。
2. `synth` 是**粗粒度结构综合 + 示例库面积**，无 SDC、无目标工艺库、无 STA，不能当签核。
3. `smoke_rtl_vector` 的向量是打通路径的小样例，**不是**真实 KWT-1 12 层数据，也不代表量化精度。
4. `smoke_software_golden` 只是 FP32 软件误差（首层编码器输入），与 RTL 逐位一致是两回事。
5. 注意力/单层/十二层编码器、E203 驱动的整层与 12 层换参在**本分支未被验证**（SKIP），
   需要相应独立分支补齐模块与 harness 后再跑同一 runner。
6. 未 push；未合并进 `main`；`main` 的既有状态未改动。
