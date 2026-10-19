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

## 2. 复现命令与退出码（本任务，2026-10-19）

```bash
VENV=/home/yang/Desktop/ic设计竞赛/software/.venv/bin/python
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 $VENV graduate/ds_tasks/layer_params/build_layer_manifest.py --device auto
# 退出码 0：A12W10 worst=0.147047% layer=11；控制点 A11W10=0.181954%；写出 json+npz
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 $VENV graduate/ds_tasks/layer_params/verify_manifest.py --device auto
# 退出码 0：VERIFY_PASS samples=480/480 worst=0.147047% layer=11
```

| 命令 | 退出码 | 关键输出 |
| --- | ---: | --- |
| `build_layer_manifest.py --device auto` | **0** | A12W10 `0.147047%` / A11W10 `0.181954%`（layer 11） |
| `verify_manifest.py --device auto` | **0** | `VERIFY_PASS samples=480/480 worst=0.147047%` |
| `build_layer_manifest.py --device cpu --output-dir /tmp/…` | **0** | CPU worst `0.147048%`（设备敏感性） |

日志：`graduate/ds_tasks/layer_params/logs/build.log`、`logs/verify.log`。
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
