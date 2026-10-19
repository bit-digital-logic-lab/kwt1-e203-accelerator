# attn_scale — 用训练划分冻结注意力尺度，独立验证多样本

任务：修复注意力整数参考模型从**验证样本**求 `sx` / `fit_shift` 的标定泄漏，改为
**只用训练划分**冻结全部数据相关常量，**验证划分只做评估**。

- 范围：只改 `graduate/encoder/model/attn_int_model.py`，新增 `graduate/ds_tasks/attn_scale/`。
- 只在本 worktree 操作；未开子智能体；未改其他 worktree；未 `merge`/`rebase`/`push`。
- 数据：只读使用官方 manifest 标注的 train / validation 划分（6 条真实 encoder 输入，
  每条数组 SHA-256 与 `software/validation/real_input_chain_20261017/manifest.json` 一致）。

## 1. 问题（修复前）

复制自 grad-attention 的 `attn_int_model.py` 在 `main()` 里：

```python
Xi, sx = symmetrize(Xf, ACT_BITS)     # X 尺度取自“当前评估样本”
RQ = fit_shift(Qa); RK = fit_shift(Ka); RV = fit_shift(Va)
RC = fit_shift(ctx); RO = fit_shift(Oa)
```

`Xf` 正是被评估的验证样本，于是 `sx`、`RQ/RK/RV/RC/RO`、`KC/SH_KC`、输出
`bias` 全部隐含地由验证数据决定——这是**把测试集用于标定**的泄漏（硬件常量本应在
量产前冻结）。逐样本 `fit_shift` 还把“编译期常量”变成了“每条样本各算一次”。

## 2. 修复（修复后）

`attn_int_model.py` 拆成两相：

- **标定（训练划分，无梯度）** `calibrate()`：
  - 从 manifest 读取 `split == "training"` 的样本（`sample_01/03/04`），并断言
    每个样本的数组 SHA-256 等于 manifest 的 `encoder_sha256`；
  - `scale_x = max|X_train| / 2047`（对称 A12，零点 0）；
  - `RQ/RK/RV = max_train fit_shift(...)`；
  - `KC/SH_KC` 由冻结的 `scale_q·scale_k` 推出；`RC` 由训练上下文拟合；
    输出 `bias_int` 与 `RO` 由冻结尺度推出；`scale_o = scale_ct·scale_wo·2^RO`。
  - 常量写入 `attn_calib_train.json`（含训练样本哈希、权重哈希、`calibration_split`）。
- **评估（验证划分，只读常量）** `integer_forward()`：
  - 用冻结 `scale_x` 量化输入（`integer_forward()` 不重新拟合任何尺度）；
  - 行内 softmax 最大值 `mmax` 是算法内的**运行时**操作，不是冻结阈值；
  - 权重尺度只来自 checkpoint（与划分无关）。

`load_calibration()` 断言标定文件来自训练划分，且标定溯源里不含任何验证样本。

## 3. 样本与冻结常量

| 划分 | 样本（数组 SHA-256 见 manifest） | 条数 |
| --- | --- | ---: |
| training（标定） | `sample_01`(go)、`sample_03`(no)、`sample_04`(yes) | 3 |
| validation（只评估） | `sample_00`(_unknown_)、`sample_02`(no)、`sample_05`(yes) | 3 |

冻结常量（`attn_calib_train.json`，SHA-256 `9a535f04…fa267f`）：

| 常量 | 值 | 常量 | 值 |
| --- | ---: | --- | ---: |
| `scale_x` | 1.961475124e-03 | `RQ/RK/RV` | 6 / 7 / 7 |
| `KC` | 6986 | `SH_KC` | 37 |
| `RC` | 14 | `RO` | 11 |
| `scale_o` | 3.579879541e-05 | `scale_ct` | 4.447924715e-05 |

> 修复前后 `RQ/RK/RV/RC/RO` 恰好相同（`fit_shift` 阈值较粗），但 `scale_x`
> （2.776/2047→**4.015/2047**）、`KC`（6680→**6986**）、`SH_KC`（38→**37**）、
> `scale_o` 都变了：旧值确实带了验证样本 `sample_00` 的信息。

## 4. 结果

### L1 软件参考（`run_attn_scale.py`，独立实现与模型逐位互校）

| 样本 | 冻结尺度 NRMSE | max_abs | 逐位(driver==model) | 泄漏基线 NRMSE |
| --- | ---: | ---: | :---: | ---: |
| validation `sample_00` | **0.1642%** | 1.339e-03 | 是 | 0.1589% |
| validation `sample_02` | **0.2321%** | 2.021e-03 | 是 | 0.2124% |
| validation `sample_05` | **0.1748%** | 2.552e-03 | 是 | 0.1707% |
| training `sample_01` | 0.2034% | — | — | — |
| training `sample_03` | 0.2190% | — | — | — |
| training `sample_04` | 0.2726% | — | — | — |

- 三条验证样本全部 **< 3% 门槛**；“泄漏基线”列是**诊断**（按单条验证样本重标定，
  即修复前行为），仅用于对比，**不是交付结果**。冻结尺度比泄漏基线略差
  （0.16→0.23%），正是“不偷看测试集”的代价，方向正确。
- P 行和 `32708..32729`（理想 32768），与 grad-attention 一致。
- 标定重复两次结果逐位一致；同一验证样本重复评估 `params.json` 哈希一致。

### L2 注意力 RTL 逐位（`rtl_check.sh`）

用本仓库只读 RTL 副本 `graduate/rtl/encoder/third_party/attn_single_head.sv` +
**重新生成的** `attn_params.svh`（KC=6986, SH_KC=37）+ 逐样本向量，Verilator 逐位对照：

| 样本 | 逐位 q/k/v/p/c/o | FP32 NRMSE | TB 退出码 |
| --- | ---: | ---: | ---: |
| validation `sample_00` | 41481/41481 | 0.16419% | 0 |
| validation `sample_02` | 41481/41481 | 0.23209% | 0 |
| validation `sample_05` | 41481/41481 | 0.17484% | 0 |

TB 只装载/回读/比较，不做任何乘加或 softmax（见 `tb/tb_attn.cpp` 头注）。

### 证据层级

| 层级 | 状态 | 证据 |
| --- | --- | --- |
| L1 软件参考 | **已证** | `run_attn_scale.py`、`results.json`、`attn_int_model.py`；独立实现逐位互校 |
| L2 注意力 RTL 逐位 | **已证** | `rtl_check.sh`，3×41481/41481；`logs/rtl_val_sample*.log` |
| E203 SoC | **未证** | 本任务未接入真实 E203/总线/NICE |
| 综合/时序/面积 | **未证** | 未跑综合、STA、面积 |
| 官方验收（600 条） | **未证** | 无官方评测脚本与 600 条数据 |

## 5. 泄漏审计（`run_attn_scale.py` 步骤 [5]，全部 PASS）

- `calibration_file_unchanged_by_eval`：评估前后标定文件 SHA-256 不变；
- `calibration_split_is_training`：标定文件声明为 training；
- `no_validation_sample_in_calibration_provenance`：标定溯源不含任何验证样本；
- `frozen_sx_differs_from_each_validation_sx`：冻结 `scale_x` 与每条验证样本各自的
  “样本内”标定尺度都不同（证明未使用验证样本）；
- `driver_matches_model_bit_exact`：独立驱动与模型逐位一致；
- `all_validation_nrmse_below_limit`：三条验证样本 NRMSE 均 < 3%。

## 6. 复现

```bash
bash graduate/ds_tasks/attn_scale/repro.sh
```

分步：

```bash
python3 graduate/encoder/model/attn_int_model.py --recalibrate --no-write-vectors
python3 graduate/ds_tasks/attn_scale/run_attn_scale.py
bash    graduate/ds_tasks/attn_scale/rtl_check.sh
python3 graduate/encoder/model/attn_int_model.py --sample 2   # 单样本（0/1/2 = 验证划分）
```

`attn_int_model.py --sample N` 的 `N` 是**验证划分**下标（0→`sample_00`，1→`sample_02`，
2→`sample_05`）；缺省加载冻结标定文件，找不到即报错，不会退回“按样本标定”。

## 7. 未证 / 影响（诚实边界）

1. **样本量小**：本仓库仅 6 条真实 encoder 输入（训练 3 / 验证 3）。规模足以证明
   “训练冻结、验证独立”，但不足以代表官方 600 条评测集的整体分布。
2. **仅第 0 层**：只标定 `enc_layers/0` 注意力；第 1–11 层未重标。
3. **A12 有效位轻微越界**：冻结尺度下验证样本 `sample_05` 有 2/6336 个输入量化值
   `|Xi|=2095>2047`，仍在 int16 容器内、无饱和（`x_int_sat_int16=0`）。这是冻结训练
   尺度对验证样本覆盖略不足的正常泛化现象，未做重标定。
4. **跨任务常量未同步**：全编码器 TB `graduate/encoder/tb/run.sh` 使用
   `third_party/attn_params.svh`（仍是旧的 KC=6680/SH_KC=38），且
   `kwt_postnorm_ref.py` 的输入尺度 `s_x` 仍按验证样本 `sample_00` 标定；两者都在本任务
   允许修改的文件之外，**未重跑**该脚本。要把新注意力尺度接进整链，需要同步更新
   `third_party/attn_params.svh` 并让 postnorm 也使用训练冻结的同一输入尺度
   （属于后续任务/其他 worker 的范围）。
5. **RTL 逐位仅覆盖注意力**：整层/12 层 RTL 逐位、E203 集成、综合、官方验收均未证明。
6. `attn_calib_train.json` 是训练推导产物，已冻结提交；重新标定可复现同一文件。
