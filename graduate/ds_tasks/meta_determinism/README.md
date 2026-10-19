# meta_determinism — grad-encoder `meta.json` 哈希可复现性修复

任务（`graduate/fleet/dispatch.py`）：修复 `grad-encoder` 的
`ACCEPTANCE.json` 中 `graduate/encoder/vectors/meta.json` 的 sha256 在干净树
里不可复现的问题。

- 分支 / worktree：`codex/kwt-ds-grad-meta_determinism` @
  `/home/yang/kwt-worktrees/ds-grad-meta_determinism`
- 只改：`graduate/encoder/vectors/gen_vectors.py`、`ACCEPTANCE.json`、
  本目录 `graduate/ds_tasks/meta_determinism/`。
- 未开子智能体、未改他人 worktree、未 merge/rebase/push。

## 1. 现象（真实失败）

编排者 `verify_worker.py` 从 `git archive HEAD` 导出干净树后复跑，命令退出码
全 0，但 5 个声明产物里只有 `meta.json` 摘要不符：

| 产物 | declared | actual (干净树) | ok |
| --- | --- | --- | --- |
| `tb/build/y.hex` | `7620a347…` | `7620a347…` | ✔ |
| `tb/build/r1.hex` | `87b3fcdb…` | `87b3fcdb…` | ✔ |
| `tb/build/g.hex` | `ef8e1586…` | `ef8e1586…` | ✔ |
| `tb/build/verify.log` | `7fa33efb…` | `7fa33efb…` | ✔ |
| `vectors/meta.json` | `124e5d35…` | **`03c02015…`** | ✘ |

来源：`evidence/orchestrator_verify_before_fix.json`（
`/home/yang/kwt-worktrees/orchestration/status/grad-encoder.verify.json`）。

## 2. 根因（已在同机复现，非推测）

`gen_vectors.py` 生成 `meta.json` 时直接写入 FP64 诊断量（注意力 NRMSE、
从 BLAS 产出的张量上取的量化尺度）。这些值的最后若干 ULP 取决于 BLAS 实现与
线程数：同一份代码在两种解释器下产生不同字节。

用 HEAD^ 的旧代码 + 干净导出实测（`demo_baseline_nondeterminism.sh`）：

```
interpreter A  python 3.10.12, numpy 1.21.5  (netlib reference BLAS, /usr/bin/python3)
  meta.json sha256 = 124e5d3503bf3539286bdcc0e3e4f3b5093a01ee2963fd67ac0480e70edb40a7   == 声明值
interpreter B  python 3.12.12, numpy 2.x     (scipy-openblas, 项目 software/.venv)
  meta.json sha256 = 03c020155286f84783a11276cdc07bf307143a13862887860e1338150c3fe880   == verify actual
```

唯一差异是最后几个有效位（`evidence/baseline/diff.txt`）：

```
-  "scale_a": 0.00027124510409477423,
+  "scale_a": 0.0002712451040947742,
-  "attn_nrmse_fp32": 0.0015886401079165293
+  "attn_nrmse_fp32": 0.0015886401079165228
```

工作进程当时用 `/usr/bin/python3`（得 `124e5d35…`）记录声明值，而
`verify_worker` 进程环境里 `python3` 解析到项目 venv（得 `03c02015…`），
于是干净树复跑摘要不符。

## 3. 修复

在 `gen_vectors.py` 增加 `canon_float()`：写 `meta.json` 前把每个诊断浮点
四舍五入到 **9 位有效数字**（`f"{x:.9g}"`，再转回 float）。

- 9 位有效数字对观测到的 ULP 噪声（约 4e-15 相对）留 ~10⁶ 倍余量，
  远高于报告所需精度，且使文件对解释器/BLAS 逐字节一致。
- 整数域（`T/D/HID/M_A/Q_A/layer`）与喂给 RTL 的系数（hex、`M_A`）不变。
- `scale_x` 等由精确 `max` 得到的量在 9 位下不变；由 BLAS 张量得到的
  `scale_a/sx/s_out` 也一并规范化。

修复后两种解释器一致：

```
interpreter A (/usr/bin/python3): meta.json sha256 = d63fc090ba420763d133a5206270c53a91c0bf5e4865202135768699fb2a2b34
interpreter B (.venv            ): meta.json sha256 = d63fc090ba420763d133a5206270c53a91c0bf5e4865202135768699fb2a2b34
```

`ACCEPTANCE.json` 中 `meta.json` 摘要更新为 `d63fc090…`；其余 4 个产物摘要
不变（整数域输入未变，逐位结果与 `verify.log` 不受影响）。

## 4. 干净树独立复跑（本任务证据）

`repro_acceptance.sh` 完全照 `verify_worker.py` 的做法：`git archive HEAD`
→ 一次性导出目录 → 按 `ACCEPTANCE.json` 逐条执行并记录真实退出码 → 重算每个
声明产物 sha256。

| 解释器 | HEAD | 命令退出码 | 产物摘要 | 结论 | 证据 |
| --- | --- | --- | --- | --- | --- |
| `/usr/bin/python3` (numpy 1.21.5) | `3deec02` | `0 0` | 5/5 匹配 | `ACCEPTANCE_REPRO_OK` | `evidence/clean_tree_sys/` |
| 项目 `.venv` (numpy 2.x, OpenBLAS) | `3deec02` | `0 0` | 5/5 匹配 | `ACCEPTANCE_REPRO_OK` | `evidence/clean_tree_venv/` |

两次 `summary.json` 除 `generated_at` 外完全一致；`meta.json` 两次均为
`d63fc090…`。

复跑命令：

```bash
bash graduate/ds_tasks/meta_determinism/repro_acceptance.sh --out /tmp/opencode/repro-sys
bash graduate/ds_tasks/meta_determinism/repro_acceptance.sh \
     --python /home/yang/miniconda3/bin/python3 --out /tmp/opencode/repro-venv
bash graduate/ds_tasks/meta_determinism/demo_baseline_nondeterminism.sh
```

## 5. 证据层级

| 层级 | 本任务状态 | 依据 |
| --- | --- | --- |
| 软件参考（独立整数模型） | 覆盖 | `verify_encoder.py` 逐位比对 + FP32 NRMSE |
| RTL 逐位 | 覆盖（`a/r1/g/f/r2/y` 6 阶段 99 行全一致） | `evidence/clean_tree_*/cmd1.log`：`TB_PASS`/`VERIFY_PASS`、`sat=0/0` |
| E203 / NICE / 系统总线 | **未覆盖** | 仍是 E203-like 事务模型，见 `graduate/encoder/SOFTWARE_BOUNDARY.md` |
| 综合 / STA / 面积 / 功耗 | **未覆盖** | 无 PDK/SDC，仅 Verilator 功能仿真 |
| 官方验收（编排者 `verify_worker`） | 本任务不在其内，仅复刻其流程 | 本目录 `repro_acceptance.sh`；真正的 `verified_ok` 由编排者裁定 |

## 6. 数据划分说明

本修复只改 `meta.json` 的序列化精度，不新增任何标定：`gen_vectors.py` 仍
沿用既有 `attn_int_model.py` / `kwt_postnorm_ref.calibrate()`，其尺度来自
验证样本 `sample_00_encoder.npy`。该“验证样本求尺度”是**既有**泄漏问题，属于
`attn_scale` / `postnorm_scale` 两个任务的范围；本任务未触碰、也未据此计算任何
阈值。9 位有效数字只影响报告文本，不影响 `M_A`、hex 或 RTL 行为。

## 7. 未证事项

1. 本任务只证明 `ACCEPTANCE.json` 的 5 个声明产物在两种解释器/干净树下摘要
   一致，不证明 RTL 数值正确性或 12 层数值正确性（后者是 `encoder_12numeric`
   等任务范围）。
2. 9 位有效数字消除的是观测到的 ULP 级噪声；若未来 BLAS 在更低有效位上
   产生更大偏差，理论上仍可能改写第 9 位（余量约 10⁶ 倍，实际风险可忽略，
   但非形式化证明）。
3. `verify.log` 的摘要稳定依赖其中 NRMSE 只打印到 4 位小数百分数/3 位有效数字
   科学计数；若未来把 `verify_encoder.py` 的打印精度提高到接近 15 位，需要
   同样做规范化。
4. 未重跑官方 600 条评测集；未做综合/STA/功耗；未在真实 E203 SoC 上运行。
