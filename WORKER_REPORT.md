# WORKER_REPORT — meta_determinism（修复 grad-encoder `meta.json` 哈希不可复现）

- 分支 / worktree：`codex/kwt-ds-grad-meta_determinism` @
  `/home/yang/kwt-worktrees/ds-grad-meta_determinism`
- 范围：只改 `graduate/encoder/vectors/gen_vectors.py`、`ACCEPTANCE.json`、
  `graduate/ds_tasks/meta_determinism/`。未开子智能体、未改他人 worktree、
  未 merge/rebase/push。
- 详细根因、证据层级与脚本见 `graduate/ds_tasks/meta_determinism/README.md`。
  （原 `grad-encoder` 报告已被本任务报告取代，内容保存在 Git 历史
  `5136ca1` 及其 `graduate/encoder/README.md`、`PROVENANCE.md` 中。）

## 1. 结果

编排者 `verify_worker.py` 的干净树复跑里，`grad-encoder` 两条命令退出码均为
0，但 5 个声明产物中 `meta.json` 摘要不符（declared `124e5d35…` vs actual
`03c02015…`），整体 `verified_ok=false`。

**根因（同机复现，非推测）**：`gen_vectors.py` 把 FP64 诊断量（注意力 NRMSE、
从 BLAS 输出张量取的量化尺度）原样写入 `meta.json`。它们的最后若干 ULP 依赖
BLAS 实现/线程数，同一份代码在不同解释器下产生不同字节：

- `/usr/bin/python3`（numpy 1.21.5, netlib BLAS）→ `124e5d35…`（即声明的值）
- 项目 `.venv`（numpy 2.x, scipy-openblas）→ `03c02015…`（即 verify 实际值）

**修复**：`gen_vectors.py` 新增 `canon_float()`，写 `meta.json` 前把每个诊断
浮点四舍五入到 9 位有效数字。整数域（`T/D/HID/M_A/Q_A/layer`）和喂给 RTL 的
系数（hex、`M_A`）不变。修复后两种解释器得到同一摘要：

```
meta.json sha256 = d63fc090ba420763d133a5206270c53a91c0bf5e4865202135768699fb2a2b34
```

`ACCEPTANCE.json` 只更新了 `meta.json` 摘要；`y.hex`/`r1.hex`/`g.hex`/
`verify.log` 四个摘要不变。

**干净树独立复跑**（照 `verify_worker.py`：`git archive HEAD` → 一次性导出 →
逐条执行 → 重算产物摘要）：在 `/usr/bin/python3` 与项目 `.venv` 两种解释器下
均 `ACCEPTANCE_REPRO_OK`，命令退出码 `0 0`，5/5 产物摘要匹配。RTL 仍 6 阶段
99 行逐位一致（`TB_PASS`/`VERIFY_PASS`，最差阶段 GELU NRMSE 1.3987% < 3%）。

证据层级：软件参考（独立整数模型）与 RTL 逐位已覆盖；E203/NICE/总线、
综合/STA/面积/功耗、官方 600 条评测**均未覆盖**；真正的官方验收由编排者
`verify_worker` 裁定，本任务只复刻其流程。

## 2. 复现命令与退出码

| 命令 | 退出码 | 结果 |
| --- | ---: | --- |
| `bash graduate/ds_tasks/meta_determinism/repro_acceptance.sh --out /tmp/opencode/repro-sys` | **0** | `ACCEPTANCE_REPRO_OK`；命令 `0 0`；5/5 摘要匹配（`/usr/bin/python3`） |
| `bash graduate/ds_tasks/meta_determinism/repro_acceptance.sh --python /home/yang/miniconda3/bin/python3 --out /tmp/opencode/repro-venv` | **0** | `ACCEPTANCE_REPRO_OK`；命令 `0 0`；5/5 摘要匹配（项目 `.venv`） |
| `bash graduate/ds_tasks/meta_determinism/demo_baseline_nondeterminism.sh` | **0** | `BASELINE_NONDETERMINISM_REPRODUCED`；旧代码 sys=`124e5d35…` / venv=`03c02015…` |

产物摘要（干净树，两次一致）：

| 产物 | sha256 |
| --- | --- |
| `tb/build/y.hex` | `7620a3470ce5d5910c185f86f7704a88a9ef9eb67d21de476709dd859c62b0be` |
| `tb/build/r1.hex` | `87b3fcdb64149bb24e4f5ea5c0a7e3c18aeb287084c3ae5d5afc4f9536473d60` |
| `tb/build/g.hex` | `ef8e15863a8a65f211b724c8762d3ef7dee58856c9066bb9aaba0c8ade058737` |
| `tb/build/verify.log` | `7fa33efbad31073841365a9d738c5648d55e4e1bca3179e5fe671f9f9c2339a8` |
| `vectors/meta.json` | `d63fc090ba420763d133a5206270c53a91c0bf5e4865202135768699fb2a2b34` |

原始日志/摘要/基线对照在 `graduate/ds_tasks/meta_determinism/evidence/`。

## 3. 未证事项

1. 只证明 5 个声明产物在两种解释器/干净树下摘要一致，**不证明 RTL 数值或 12 层
   数值正确性**（属 `encoder_12numeric` 等任务）。
2. 9 位有效数字消除的是观测到的约 4e-15 相对 ULP 噪声，留 ~10⁶ 倍余量；非形式化
   证明，若未来 BLAS 产生更大偏差理论上仍可能改写第 9 位。
3. `meta.json` 尺度仍取自验证样本 `sample_00_encoder.npy`（**既有**泄漏，属
   `attn_scale`/`postnorm_scale` 任务）；本任务未新增标定、未用验证样本求任何阈值。
4. `verify.log` 摘要稳定依赖其只打印 4 位小数百分数/3 位有效数字，提高打印精度需
   同样规范化。
5. 未做 E203/NICE/总线真实接口、综合/STA/面积/功耗签核、官方 600 条评测。
