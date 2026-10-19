# REPORT — layer_params：12 层冻结整数权重/偏置/尺度 manifest

- 分支 / worktree：`codex/kwt-ds-grad-layer_params` @ `/home/yang/kwt-worktrees/ds-grad-layer_params`
- 基线：`main` 系列 `5136ca1`（`codex/kwt-grad-encoder` 组装提交）
- 边界：只在本 worktree 操作；未开子智能体；未改其他 worktree；未 `merge`/`rebase`/`push`；
  未改 `software/`、`hardware/`、`graduate/encoder` 任何既有文件（只**只读**引用）。
- 只写：`graduate/ds_tasks/layer_params/**` 与根 `WORKER_REPORT.md`。

## 1. 结果

**产出**：A12W10 冻结 manifest，覆盖 12 层 FFN1 第一线性（`99×64 @ 64×256 + bias`），
训练标定、验证只评估。

- 训练 / 验证样本：**480 / 480**（12 类各 40；Speech Commands V2，官方缓存
  `quant_samples_12labels_40.npz`，训练/验证路径无交叉，无测试划分）。
- 官方权重：`software/weights/kwt1_fp32.npz`（SHA-256 `8f359abf…25825a`），**未修改**。
- 冻结产物：`layer_manifest_A12W10.json` + `layer_manifest_A12W10.npz`
  （npz SHA-256 `41fb8d28…994ed5`）。

**验证划分 NRMSE（A12W10，训练标定）**：

| 层 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NRMSE % | 0.1099 | 0.1065 | 0.1158 | 0.1293 | 0.1237 | 0.1178 | 0.1172 | 0.1200 | 0.1173 | 0.1312 | **0.1470** | 0.1434 |

最差层 = 第 11 层 = **0.147047% ≪ 3%** 门槛（`validation_pass=true`）。

**独立实现与交叉核对**：

1. 量化器、整数 GEMM、NRMSE 均在本脚本**独立重写**；与仓库既有参考函数
   （`software/calibrate_blackbox_ffn1.py`）逐位对照：A11W10 与 A12W10 的权重、偏置编码
   `weight_identical=true, bias_identical=true`。
2. 复现既有已发布控制点（480/480）：**A11W10 = 0.181954%（第 11 层）**、
   **A12W10 = 0.147047%（第 11 层）**——与 `software/results/ffn1_container_headroom_12labels_40.json`
   完全一致，证明口径一致。
3. 与既有冻结包 `software/results/ffn1_a12w10_static_sym.npz`（240/240）比较：
   **整数权重与权重尺度 0 失配**（二者只依赖冻结 FP32 权重）；激活尺度因训练集从 240→480
   样本而最大差 `1.77e-4`（训练最大值随样本数增大，最多约 7%），偏置随之有 1816/3072 项差。
   这量化了“尺度来自训练集最大值”的样本敏感性。
4. `verify_manifest.py` 从冻结 npz 重建：尺度来源（训练最大值）、整数权重/偏置、INT32 上界、
   验证 NRMSE 逐层重算，全部通过（`VERIFY_PASS samples=480/480 worst=0.147047% layer=11`）。

**设备与可复现性**：manifest 冻结于 `device=cuda`（torch 2.5.1+cu121，Python 3.12.12）。
CPU 复跑得最差 `0.147048%`，激活尺度最大差 `6.99e-10`、49/3072 偏置差 1 LSB、权重/尺度一致；
故 **npz/JSON 哈希是设备绑定的**，复现须用同一 `--device`。

**硬件装载映射**：manifest 对应 `kwt_encoder_layers_ctrl.sv` 事务
`sel7 W1(16384×int16)` + `sel8 b1(256×int32)`；每层 staged 装载 33,792 B，
12 层权重容器 393,216 B（10 位未打包），打包后 240 KiB。

## 2. 复现命令与退出码（本任务，2026-10-19）

| 命令（worktree 根执行） | 退出码 | 结果 |
| --- | ---: | --- |
| `…/software/.venv/bin/python graduate/ds_tasks/layer_params/build_layer_manifest.py --device auto` | **0** | A12W10 worst `0.147047%` layer 11；控制点 A11W10 `0.181954%`；`wrote …json/.npz` |
| `…/software/.venv/bin/python graduate/ds_tasks/layer_params/verify_manifest.py --device auto` | **0** | `VERIFY_PASS samples=480/480 worst=0.147047% layer=11` |
| 同上 build 加 `--device cpu --output-dir /tmp/opencode/layer_params_cpu` | **0** | CPU 复跑 worst `0.147048%`（设备敏感性证据） |

日志：`graduate/ds_tasks/layer_params/logs/build.log`、`logs/verify.log`。

## 3. 未证事项（不得当作完整研究生组验收）

1. **无 RTL 逐位**：本任务只做软件整数参考；未把该 manifest 灌入 `kwt_encoder_layer`/RTL 做逐位对照。
   既有 layer-0 逐位证据在 `graduate/encoder/tb/run.sh`，且用的是 240/240 冻结包，**与本 manifest 非同一次标定**。
2. **无 E203 SoC**：未在官方 E203/NICE/System Bus 上运行；无驱动、无总线协议验证。
3. **无综合**：无目标 PDK/SRAM 宏/SDC，未做 150 MHz STA、面积、功耗签核。
4. **非官方验收**：未跑官方 600 条评测集；NRMSE 分母按本设计口径（FP32 输出 RMS）；未核对官方
   评测脚本张量展平/RMS 细节。验证集 480 条只代表本缓存。
5. **范围限于 FFN1 第一线性**：注意力、LayerNorm、GELU、FFN2、分类头不在本 manifest。
6. **样本敏感性**：激活尺度为训练集最大值，随训练样本数变化（240→480 时最大约 7%）；
   manifest 的哈希因此对数据缓存与设备都敏感。
7. 未声明“12 层整链 E203 通过”或“研究生 5–6 项验收通过”。
