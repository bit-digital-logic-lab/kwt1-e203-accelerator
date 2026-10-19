# postnorm_scale — PostNorm 激活尺度/LUT 改由训练集冻结，验证集只评估

对应任务 `postnorm_scale`：修复 `graduate/encoder/model/kwt_postnorm_ref.py` 的
`calibrate()` **直接从验证样本 `sample_00` 反推中间张量尺度**的问题。

## 1. 问题（校准泄漏）

原 `calibrate()` 用评估样本 `sample_00_encoder.npy` 直接求
`s_x, s_a, s_g, s_r1, s_r2, s_out`：

```python
s_x = max(abs(x0)) / CLAMP_A           # x0 = sample_00（官方 validation 划分）
s_a = max(abs(ref["attn_out"])) / CLAMP_A
sg  = max(abs(ref["g"])) / CLAMP_A
sr1 = max(abs(ref["r1"])) / CLAMP_A
sr2 = max(abs(ref["r2"])) / CLAMP_A
s_out = max(abs(ref["y"])) / CLAMP_A
```

`sample_00` 在 [`manifest.json`](../../../software/validation/real_input_chain_20261017/manifest.json)
中 `split == "validation"`。题目 [`题目/640.jpg 7`](../../../题目/640.jpg%207) 第 3 条规定：
训练划分用于无梯度标定，验证划分用于功能和精度验证，**不得**用验证样本确定尺度/阈值。

## 2. 修复

`calibrate()` 现在只从**训练集冻结文件**读取尺度与 GELU LUT，验证样本仅用于
量化/评估（`compute xq/aq -> integer 前向 -> 与 FP32 对照`），不再反推任何尺度：

- 冻结文件：`graduate/ds_tasks/postnorm_scale/frozen_postnorm_calib.json`
  （`calibration_split: "training"`，含 257 点 GELU LUT 与来源哈希）；
- 生成脚本：`graduate/ds_tasks/postnorm_scale/run_postnorm_scale.py`；
- 标定划分：`software/.build/quant_samples_12labels_20.npz` 中 `split=="training"`
  的 **240** 条（与冻结 A12W10 FFN1 包 `ffn1_a12w10_static_sym.npz` 同一划分、同一
  `sample_sha256`），验证 **240** 条全程只评估；
- 编码器输入：`class_emb + patch_proj(mfcc) + pos_emb`（与 `offline_encoder_input.py`
  逐位一致，最大误差 `4.77e-07`）；
- 无梯度：只有最大值统计与整数舍入，无任何梯度/训练。

## 3. 实测结果（训练冻结尺度 vs FP32 golden）

| 张量 | 冻结尺度 | 训练集最大值 |
| --- | ---: | ---: |
| `s_x` | 6.737881e-03 | 13.792443 |
| `s_a` | 1.536083e-03 | 3.144361 |
| `s_g` | 1.984456e-03 | 4.062182 |
| `s_r1` | 6.694600e-03 | 13.703847 |
| `s_r2` | 2.569834e-03 | 5.260450 |
| `s_out` | 2.140992e-03 | 4.382611 |

整数 PostNorm 路径（冻结尺度）反量化 vs FP32，**验证集 240 条**：

| 阶段 | 平均 NRMSE | 最差样本 NRMSE |
| --- | ---: | ---: |
| `r1` | 0.3465% | - |
| `g` | 1.1766% | - |
| `r2` | 0.8728% | - |
| `y` | 0.7847% | **1.6898%**（`./on/e11fbc6e_nohash_2.wav`） |

三条真实 validation WAV 派生的编码器输入：`sample_00` y=0.7170%（worst 1.5634%）、
`sample_02` y=0.7374%（worst 1.1836%）、`sample_05` y=0.7745%（worst 1.0534%），
全部 < 3%。

## 4. 泄漏审计（全部 PASS）

| 检查 | 结果 |
| --- | --- |
| 校准划分 = training | True |
| `validation_used_for_calibration` | False |
| 评估后校准文件字节不变 | True（sha256 不变） |
| 6 个尺度的取最大值样本都属于训练集 | True（无 validation 路径） |
| 冻结尺度 ≠ `sample_00` 自身 per-sample 尺度 | True（6/6 项都不同） |
| 全部验证 NRMSE < 3% | True |
| 验证样本偶然共享同一冻结最大值 | 24/240，全部是 `_silence_` 片段（训练/验证共用同一静音波形，非泄漏） |

## 5. 证据层级（本任务实际达到）

| 层级 | 状态 | 依据 |
| --- | --- | --- |
| 软件参考 | **是** | 本目录 `run_postnorm_scale.py` + `frozen_postnorm_calib.json`（训练冻结尺度，验证只评估） |
| RTL 逐位 | **否（本任务未达成）** | 组装 RTL 的常数（`third_party/kwt_postnorm_cfg.svh`、`tb_encoder.sv` 的 `M_A`）仍冻结在**旧的逐样本标定**上；且注意力级 `scale_x/scale_o` 仍由 `sample_00` 决定（属 `attn_scale` 任务）。8 个模型常数仅 1 个与 RTL 头一致。见 `results.json.rtl_coupling` 与 `logs/rtl_run.log`（`VERIFY_FAIL`，6330+ 失配） |
| E203 | 否 | 未接官方 SoC/NICE/总线 |
| 综合 | 否 | 未做目标 PDK 综合/STA/面积/功耗 |
| 官方验收 | 否 | 未跑官方 600 条评测集 NRMSE/准确率 |

## 6. 复现命令与退出码

```bash
# 主证据（软件参考）：训练冻结 + 验证评估 + 泄漏审计
python3 graduate/ds_tasks/postnorm_scale/run_postnorm_scale.py
#   退出码 0，输出 POSTNORM_SCALE_PASS；日志 logs/run.log，汇总 results.json
#   样本数：训练 240，验证（数据集）240，真实 validation 3（sample_00/02/05）

# 组装 RTL 现状（对照，退出码 1）
bash graduate/encoder/tb/run.sh
#   退出码 1：TB_PASS 但 VERIFY_FAIL（RTL 常数停留在旧逐样本标定）
```

冻结/汇总文件哈希（sha256）：

| 文件 | sha256 |
| --- | --- |
| `frozen_postnorm_calib.json` | `3a1cf51b2f00716b0dc6a182d80e6d4cdf5886fa196affad268a47167bbf53fb` |
| `results.json` | `53c93b6549d6fb40cdb1c02f72d7c3885e29af598b177e82ed1f3f71e30250f2` |
| `run_postnorm_scale.py` | `72831b8d2671cb883d431b973894ca1b8eaa38a56e10919218d0a65bdbcb1bee` |
| `model/kwt_postnorm_ref.py` | `1a6e316604fedddbe187015879e8d0c414b9dd14c5e71b0cbadb7c51e11cb80e` |

## 7. 未证事项 / 边界

1. 生成冻结文件需要主工程 gitignored 数据 `software/.build/quant_samples_12labels_20.npz`
   （本 worktree 只读引用；已提交的冻结文件使其在干净副本中无需该数据即可复算）。
2. **RTL 逐位未达成**：本次修复会改变组装 RTL 的常数，而 RTL 常数头与测试台 `M_A`
   仍冻结在旧的逐样本标定上（本任务只允许改 `kwt_postnorm_ref.py` 与 `ds_tasks/`）。
   需要集成阶段用新的冻结尺度重生 `kwt_postnorm_cfg.svh`、`tb_encoder.sv` 的 `M_A`，
   并同步 `attn_scale` 任务把注意力级 `scale_x/scale_o` 也改为训练冻结，才能恢复整链逐位。
3. 仅第 0 层的 PostNorm 尺度被冻结；第 1–11 层未标定。
4. 未做 E203、综合、官方 600 条验收。
