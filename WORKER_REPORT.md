# WORKER_REPORT — layer_params（12 层冻结整数权重/偏置/尺度 manifest）

- 分支 / worktree：`codex/kwt-ds-grad-layer_params` @ `/home/yang/kwt-worktrees/ds-grad-layer_params`
- 任务：为 12 层 FFN1 第一线性（`99×64 @ 64×256 + bias → 99×256`）生成冻结整数
  权重/偏置/尺度 manifest；训练集标定、验证集只评估；不改原模型。
- 产物目录：`graduate/ds_tasks/layer_params/`（详见其 `REPORT.md`、`README.md`）。

## 1. 结果

- 冻结方案 **A12W10**（激活 12 位 / 权重 10 位，有符号零点 0），
  `layer_manifest_A12W10.json` + `layer_manifest_A12W10.npz`（SHA-256 `41fb8d28…994ed5`）。
- 训练 / 验证 = **480 / 480**（官方缓存 `quant_samples_12labels_40.npz`，路径无交叉、无测试划分）；
  官方权重 `software/weights/kwt1_fp32.npz`（`8f359abf…25825a`）**未修改**。
- 验证集最差层 = 第 11 层 = **0.147047% < 3%**（`validation_pass=true`）；
  逐层 NRMSE 0.1065%–0.1470%。
- 独立重写量化/整数 GEMM/NRMSE，并与既有参考函数逐位对照：
  A11W10 `0.181954%`、A12W10 `0.147047%`（均第 11 层）与已发布控制点完全一致，
  权重/偏置整数编码 `weight_identical/bias_identical = true`。
- 独立门禁 `verify_manifest.py` 从冻结 npz 重建：尺度来源（训练最大值）、整数量化、INT32 上界、
  验证 NRMSE 全部通过（`VERIFY_PASS samples=480/480 worst=0.147047% layer=11`）。
- 证据层级：**L1 软件参考已证**；L2 RTL 逐位未产生、L3 E203、L4 综合、L5 官方验收均 **MISSING**。

## 1b. 验收修正（2026-10-19，浮点 NRMSE 门限）

- 主验收 `verify_manifest.py --device auto` 曾退出 1：哈希/训练来源/整数权重偏置/INT32/`<3%` 全过，
  仅 11 层 `abs(measured-recorded) < 1e-9` 失败。
- 诊断：参考项 `val_x @ W + b` 为 FP32 CPU BLAS 归约，求和顺序随 BLAS/OMP 线程数（记录值=4 线程，
  验收默认=12）与设备（CUDA 冻结→CPU 复验）变化。实测同设备线程最大 `|d|=2.0e-8` 百分点、
  跨设备最大 `|d|=1.5e-6` 百分点（相对 `1.4e-5`）；`1e-9` 绝对门限不成立。
- 修正仅限 `verify_manifest.py`，**未改任何数值/npz/json**：整数与哈希保持逐位相等；
  浮点 NRMSE 用 `1e-3` 相对 + `1e-6` 百分点下限，激活尺度来源用 `1e-5` 相对容忍
  （仍拒绝验证集标定，差距 ≥1.14e-2）；`<3%` 门限与最差层独立断言。
- 负控制：把某层记录值仅抬高 0.5% 相对仍判 FAIL，容差非空。依据与限制见
  `graduate/ds_tasks/layer_params/logs/nrmse_diagnosis.txt` 与 `REPORT.md` §1b。

## 2. 复现命令与退出码（本任务，2026-10-19）

```bash
VENV=/home/yang/Desktop/ic设计竞赛/software/.venv/bin/python
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 $VENV graduate/ds_tasks/layer_params/build_layer_manifest.py --device auto
# 退出码 0：A12W10 worst=0.147047% layer=11；控制点 A11W10=0.181954%；写出 json+npz
$VENV graduate/ds_tasks/layer_params/verify_manifest.py --device auto   # 主验收口径，退出码 0
$VENV graduate/ds_tasks/layer_params/verify_manifest.py --device cpu    # 跨设备复验，退出码 0
# 退出码 0：VERIFY_PASS samples=480/480 worst=0.147047% layer=11
```

| 命令 | 退出码 | 关键输出 |
| --- | ---: | --- |
| `build_layer_manifest.py --device auto` | **0** | A12W10 `0.147047%` / A11W10 `0.181954%`（layer 11） |
| `verify_manifest.py --device auto`（无线程 env） | **0** | `VERIFY_PASS … max_dev=1.99e-08pp` |
| `verify_manifest.py --device cpu` | **0** | `VERIFY_PASS … max_dev=1.45e-06pp` |
| `build_layer_manifest.py --device cpu --output-dir /tmp/…` | **0** | CPU worst `0.147048%`（设备敏感性） |

日志：`graduate/ds_tasks/layer_params/logs/build.log`、`logs/verify.log`（首次）、
`logs/verify_auto.log`、`logs/verify_cpu.log`、`logs/nrmse_diagnosis.txt`。
（manifest 冻结于 `device=cuda`；CPU 复跑激活尺度最大差 `6.99e-10`、偏置 49/3072 项差 1 LSB，
故哈希设备绑定，复现须同设备。）

## 3. 未证事项

1. 无 RTL 逐位：manifest 未灌入 RTL；既有 layer-0 逐位证据（`graduate/encoder/tb/run.sh`）用的是
   240/240 冻结包，与本 manifest 非同一次标定。
2. 无 E203 SoC / 驱动 / 总线/NICE 协议验证。
3. 无综合：无 PDK / SRAM 宏 / SDC，未做 150 MHz STA、面积、功耗签核。
4. 非官方验收：未跑官方 600 条评测集；NRMSE 分母为本设计口径，未核对官方脚本展平/RMS 细节。
5. 范围仅 FFN1 第一线性；注意力、LayerNorm、GELU、FFN2、分类头不在本 manifest。
6. 激活尺度为训练集最大值，随样本数变化（240→480 时最大约 7%），哈希对数据缓存与设备敏感。
