# WORKER_REPORT — attn_scale：用训练划分冻结注意力尺度，独立验证多样本

- 分支 / worktree：`codex/kwt-ds-grad-attn_scale` @ `/home/yang/kwt-worktrees/ds-grad-attn_scale`
- 范围：只改 `graduate/encoder/model/attn_int_model.py`，新增 `graduate/ds_tasks/attn_scale/`。
- 边界：只在本 worktree 操作；未开子智能体；未改其他 worktree；未 `merge`/`rebase`/`push`；
  未提交大数据/密钥/缓存。数据只读来自官方划分（`software/validation/real_input_chain_20261017`）。

## 1. 结果

**问题**：复制自 grad-attention 的注意力整数模型用**当前验证样本**求输入尺度与所有
`fit_shift`（`sx, RQ/RK/RV/RC/RO, KC/SH_KC, bias`），即“用测试集标定”。

**修复**：把模型拆成两相——
- `calibrate()`（训练划分，无梯度）：从 manifest 读 `split=="training"` 的 `sample_01/03/04`，
  校验样本数组 SHA-256 与 manifest `encoder_sha256` 一致；冻结
  `scale_x=1.961475124e-03, RQ/RK/RV=6/7/7, KC=6986, SH_KC=37, RC=14, RO=11,
  scale_o=3.579879541e-05`，写入 `attn_calib_train.json`（SHA-256 `9a535f04…fa267f`）；
- `integer_forward()`（验证划分，只读常量）：输入只用冻结 `scale_x` 量化，不再重拟合；
  softmax 的逐行最大值是算法内运行时操作，非冻结阈值。
- `load_calibration()` 断言标定来自训练划分、溯源不含验证样本；缺标定文件直接报错。

**多验证样本结果（L1 软件参考，独立实现与模型逐位互校）**

| 验证样本 | 冻结尺度 NRMSE | max_abs | 逐位(driver==model) | 泄漏基线(诊断) |
| --- | ---: | ---: | :---: | ---: |
| `sample_00` | 0.1642% | 1.34e-03 | 是 | 0.1589% |
| `sample_02` | 0.2321% | 2.02e-03 | 是 | 0.2124% |
| `sample_05` | 0.1748% | 2.55e-03 | 是 | 0.1707% |

三条验证样本全部 **< 3% 门槛**。泄漏基线是“按单条验证样本重标定”（修复前行为）的
**诊断对照**，不是交付结果；冻结尺度略差正说明未偷看测试集。

**L2 注意力 RTL 逐位**（只读 RTL 副本 + 重新生成的 `attn_params.svh`，Verilator）：
每条验证样本 `q/k/v/p/c/o` = **41481/41481 逐位一致**，FP32 NRMSE 0.16419%/0.23209%/0.17484%，
TB 退出码 0。

**泄漏审计（全部 PASS，`run_attn_scale.py` 步骤 [5]）**：标定文件评估前后不变；
标定声明 training；溯源无验证样本；冻结 `scale_x` 与每条验证样本的自标定尺度均不同；
独立驱动与模型逐位一致；三条验证 NRMSE<3%。标定重复两次逐位相同；同样本重复评估哈希一致。

**证据层级**：L1 软件参考 = 已证；L2 注意力 RTL 逐位 = 已证（仅注意力、Verilator）；
E203 = 未证；综合/时序/面积 = 未证；官方 600 条验收 = 未证。

## 2. 复现命令与退出码（本次实跑，2026-10-19）

| 命令（仓库根执行） | 退出码 | 结果 |
| --- | ---: | --- |
| `bash graduate/ds_tasks/attn_scale/repro.sh` | **0** | RECALIBRATE=0 / RUN=0 / RTL_CHECK=0 |
| `python3 graduate/encoder/model/attn_int_model.py --recalibrate --no-write-vectors` | **0** | `CALIB_OK split=training samples=3 (validation=3 never used)` |
| `python3 graduate/ds_tasks/attn_scale/run_attn_scale.py` | **0** | `SCALE_AUDIT_PASS`，6 项审计全 PASS |
| `bash graduate/ds_tasks/attn_scale/rtl_check.sh` | **0** | 3 样本各 41481/41481，`RESULT: PASS` |
| `python3 graduate/encoder/vectors/gen_vectors.py` | **0** | 与新冻结常量一致（`GEN_OK M_A=8649`） |
| `python3 graduate/encoder/model/attn_int_model.py --sample 0/1/2` | **0** | 验证划分样本 00/02/05 |

样本数：训练 3（标定）、验证 3（评估）。（`--sample N` 为验证划分下标。）
日志与退出码：`graduate/ds_tasks/attn_scale/logs/`（`repro_console.log`、
`run.log`、`rtl_check.log`、`rtl_val_sample0/1/2.log` 等）。

## 3. 未证事项

1. **样本量小**：仅 6 条真实 encoder 输入（训练 3 / 验证 3），能证“训练冻结、验证独立”，
   不代表官方 600 条评测集分布。
2. **仅第 0 层**：只标定 `enc_layers/0` 注意力，第 1–11 层未重标。
3. **A12 有效位轻微越界**：冻结尺度下 `sample_05` 有 2/6336 个输入量化值 `|Xi|=2095>2047`，
   仍在 int16 容器内、无饱和；未重标定。
4. **跨任务常量未同步（重要）**：全编码器 TB `graduate/encoder/tb/run.sh` 仍用
   `third_party/attn_params.svh`（旧 KC=6680/SH_KC=38），且 `kwt_postnorm_ref.py` 的输入尺度
   仍按验证样本 `sample_00` 标定——二者都在本任务允许文件之外，故**未重跑**该脚本；把新注意力
   尺度接进整链需同步更新第三方常量并让 postnorm 也用同一训练冻结尺度（后续任务）。
5. **RTL 逐位仅注意力**：整层/12 层 RTL 逐位、E203 集成/总线/NICE、综合与 150 MHz STA、
   面积/功耗、官方 600 条验收均未证明。
6. 不得宣称“研究生 4 项注意力验收通过”或“整链 E203 通过”。
