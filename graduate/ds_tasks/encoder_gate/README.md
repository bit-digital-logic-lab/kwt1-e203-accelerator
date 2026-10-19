# ds_tasks / encoder_gate — 把 NRMSE 从“只打印”改成真门禁

## 1. 问题

`graduate/encoder/model/verify_encoder.py` 原本在第 3 项检查里对每个阶段计算 FP32
NRMSE，并算出 `worst`，**但从未用 `worst` 做判定**：

```python
worst = max(worst, e)
print(f"  worst-stage NRMSE {worst*100:.4f}%  (threshold 3%)")
...
if not ok:          # 只判逐位一致
    print("VERIFY_FAIL"); raise SystemExit(1)
print("VERIFY_PASS")
```

因此只要 6 个阶段逐位一致，即使最差阶段 NRMSE 超过 3%，脚本仍打印
`VERIFY_PASS` 并 `exit 0`——**门禁是空的（no-op）**。

## 2. 修复（只改 `verify_encoder.py`）

| 改动 | 内容 |
| --- | --- |
| 门禁常量 | `NRMSE_THRESHOLD = 0.03`（题目 3% 规格常数，**不由任何样本拟合**） |
| 判定谓词 | 新增 `nrmse_ok(worst, threshold=NRMSE_THRESHOLD)`：`worst <= threshold` 为过，**严格超过 3% 才失败** |
| 真正失败 | 最差阶段超过阈值时打印 `GATE ... nrmse=FAIL`、`VERIFY_FAIL reason=nrmse` 并 `exit 1` |
| 机器可读 | 每阶段超限标 `<-- OVER THRESHOLD`；打印 `GATE bit_exact=.. nrmse=.. rows=.. stages=.. worst=..% threshold=..%` |
| 负例注入 | `--negative-inject {none,bitexact,nrmse}`；`--nrmse-threshold` 可覆盖阈值 |

退出码：**逐位全部一致 且 最差阶段 NRMSE ≤ 3%** 才为 0。

负例注入设计（只在测试时使用，默认关闭）：

* `bitexact`：把读回的 RTL 阵列 `y[0,0] += 1`（只污染逐位路径）→ 触发 `reason=bit_exact`；
* `nrmse`：只对**反量化路径**乘 `1.10`（不动整数阵列）→ 逐位仍一致，但 `reason=nrmse`。

## 3. 证据层级

| 层级 | 内容 | 本任务状态 |
| --- | --- | --- |
| **E0 软件参考** | NumPy 独立整数模型 + 研究生 FP32 golden | ✅ 已用（`verify_encoder.py` 的 oracle 与 NRMSE 参考） |
| **E1 RTL 逐位** | Verilator 5.030 功能仿真，6 阶段 × 99 行逐位一致 | ✅ 已跑（`run.sh`、`logs/01_run_all.log`） |
| **E2 E203** | 真实 E203 指令流 / System Bus / NICE / SRAM 宏 | ❌ 未覆盖（仅 E203-like 事务模型，见 `SOFTWARE_BOUNDARY.md`） |
| **E3 综合** | 逻辑综合 / 150 MHz STA / 面积 / 功耗 | ❌ 未覆盖（本任务不跑） |
| **E4 官方验收** | 官方 600 条评测集 NRMSE / 准确率 | ❌ 未覆盖（无官方评测集/脚本） |

**门禁本身只到 E1**：它证明“RTL 功能仿真逐位一致 + 反量化 NRMSE ≤ 3%”，
不证明 E2/E3/E4。

## 4. 划分策略（训练标定 / 验证评估）

官方 `manifest.json` 的说话人划分（6 条真实样本）：

* **训练**：`sample_01`(go)、`sample_03`(no)、`sample_04`(yes) → 只用于**无梯度标定**；
* **验证**：`sample_00`(_unknown_)、`sample_02`(no)、`sample_05`(yes) → **仅评估**。

`split_calibration.py` 用训练划分重算 per-tensor 静态尺度并断言：

* 标定只读训练样本（`assert used <= train_files`）；
* `NRMSE_THRESHOLD` 是常数 3%，不由验证样本拟合。

> **已发现的先前遗留问题（本任务文件范围内无法修复）**：当前
> `vectors/meta.json` 与 RTL 激励里 baked 的尺度来自 `kwt_postnorm_ref.calibrate()`，
> 它硬编码 `SAMPLE = sample_00`（**验证样本**）。训练划分重算显示
> `s_x` 相对偏差 **+44.6%**、`s_a` **+19.7%**、`s_out` **+4.8%**，即标定确实样本相关。
> 该尺度计算在 `kwt_postnorm_ref.py`（本任务无权修改），故列为未证事项；
> 门禁阈值本身不受影响（是常数）。

## 5. 复现

```bash
bash graduate/ds_tasks/encoder_gate/run_gate.sh      # 7 个用例，任一不符即 exit 1
```

日志与结果：

* `logs/01_run_all.log` 全链正向（向量→Verilator→校验）
* `logs/02_verify_positive.log` 带门禁的正向（exit 0）
* `logs/02b_clean_tight_threshold.log` 干净数据 + 过紧阈值 0.1%（exit 1，证明门禁随实测值动作）
* `logs/03_neg_bitexact.log` 逐位注入（exit 1，`reason=bit_exact`）
* `logs/04_neg_nrmse.log` 反量化注入（exit 1，`reason=nrmse`）
* `logs/05_gate_selfcheck.log` 谓词边界单测（12 项）
* `logs/06_split_calibration.log` 划分来源
* `summary.json` 汇总（命令/退出码/层级/样本数）

## 6. 实测结论（2026-10-19，Verilator 5.030）

* 正向：`GATE bit_exact=PASS nrmse=PASS rows=99 stages=6 worst=1.3987% threshold=3.00%`，`VERIFY_PASS`，exit 0。
* 逐位：6 阶段 × 99 行 = 57,024 个整数项全部一致。
* 负例：逐位注入 exit 1（`bit_exact`）；反量化注入 `y` NRMSE 9.9819% → exit 1（`nrmse`）。
* 驱动：`PASS=7 FAIL=0`，`GATE_DRIVER_PASS`，exit 0。
