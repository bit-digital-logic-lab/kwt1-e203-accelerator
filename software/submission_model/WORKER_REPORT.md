# WORKER_REPORT（分支 codex/kwt-frozen-model，FFN1 A12W10 数值协议冻结）

## 任务

为本科必做第一前馈线性层 `X[99,64] @ W[64,256] + bias[256] → Y[99,256]`
冻结一份可提交的 **A12W10** 数值协议与独立软件参考入口，固定算法与配置 ID，
训练划分标定、验证划分只评估。未追 A14W12；未用测试划分调参数；未改原量化核心、
RTL 或其它工作者文件；未 push。

## 交付物（全部在 `software/submission_model/`）

- `PROTOCOL.md`：冻结协议全文——张量/形状、端序、有符号性、缩放、零点、舍入、
  INT32 溢出环绕、输出字节序、A/W tile 填充规则、NRMSE 口径（分母下限 1e-8）。
- `ffn1_reference.py`：可运行 CLI，四个子命令 `config` / `emit` / `verify-vectors` / `selftest`。
- `test_submission_model.py`：可执行验收脚本，写出 `results/selftest_summary.json`。
- `config/KWT1-FFN1-A12W10-SYM-v1.json`：配置 ID + 算法 + 三份 SHA-256。
- `results/selftest_summary.json`：本轮测试摘要（小文件，入库）。
- 大文件（`expected32`、manifest、缓存）只写 gitignore 的 `software/.build/`，不入库；
  未新增任何权重或样本到提交。

## 冻结配置

| 项 | 值 |
| --- | --- |
| 配置 ID | `KWT1-FFN1-A12W10-SYM-v1` |
| 整数参数包 | `software/results/ffn1_a12w10_static_sym.npz`，SHA `a77f60fb...be1918e` |
| FP32 权重包 | `software/weights/kwt1_fp32.npz`，SHA `8f359abf...a155825a` |
| 标定样本包 | `.build/quant_samples_12labels_20.npz`，SHA `ee04e849...66c8c149` |

## 实际执行的命令与退出码

```bash
PY=/home/yang/Desktop/ic设计竞赛/software/.venv/bin/python   # numpy 2.1.3 / torch 2.5.1+cu121
V=/home/yang/Desktop/ic设计竞赛/hardware/architecture/results/real_e203_validation0_vectors
```

1. RTL 向量逐位交叉（退出码 0）：

```
$PY software/submission_model/ffn1_reference.py verify-vectors --vectors "$V" --out software/.build/submission_model
layer=01 bit_exact=True ... layer=12 bit_exact=True weights_match_config=True
PASS verify-vectors all_bit_exact=True
VERIFY_EXIT=0
```

2. 负向注入（退出码 0）：

```
wrong_weight_detected / wrong_bias_detected / wrong_rounding_detected / ties_to_even_used 全 PASS
SELFTEST_EXIT=0
```

3. 两条不同类真样本 × 12 层（退出码 0）：

```
$PY software/submission_model/ffn1_reference.py emit --sample-count 2 --device cuda --out software/.build/submission_model/emit
layer=01 NRMSE=0.102498% ... layer=11 NRMSE=0.140042% (最差) ... layer=12 NRMSE=0.136042%
PASS emit config=KWT1-FFN1-A12W10-SYM-v1 worst=0.140042%
EMIT_EXIT=0
```

4. 验收脚本（退出码 0）：

```
$PY software/submission_model/test_submission_model.py
PASS config_shas
PASS rtl_vectors_bit_exact   {"all_bit_exact": true}
PASS negative_injection_detected
PASS two_samples_12_layers   {"worst_nrmse_pct": 0.14004218815242878, "labels": ["_unknown_", "down"]}
PASS test_submission_model
TEST_EXIT=0
```

5. 生成冻结配置记录（退出码 0）：`ffn1_reference.py config` → `config/KWT1-FFN1-A12W10-SYM-v1.json`。

## 结论（证据层级 L1：本轮软件整数路径实测）

- **协议可复算**：独立 NumPy INT64 GEMM + INT32 环绕，对既有 E203 全层 golden 向量
  12 层共 **12 × 25,344 = 304,128 个输出逐位一致**；且 12 层 `w16` 与本配置 `qw`
  完全相同（权重量化只依赖冻结 FP32 权重）。
- **负向注入有效**：权重/偏置各改 1、舍入由 ties-to-even 改为截断，均被检出。
- **真样本精度**：2 条不同类验证样本（`_unknown_`、`down`）× 12 层，最差层 11
  NRMSE **0.140042%**，远低于 3% 门槛；分母已用 `max(ref_rms,1e-8)`。
- **配置冻结**：算法、A12W10 位宽、零点 0、舍入、INT32 环绕、tile 填充、输出端序
  与 NRMSE 公式写死在 `PROTOCOL.md` 与 `config/*.json`，并锁定三份 SHA-256。

## 未证事项 / 限制

- 真实样本证据是本地训练/验证划分，**不是官方隐藏 600 条**；两样本子集 0.140042%
  与既有 240 条汇总的 0.147047% 不可互相替换。
- RTL 逐位交叉用的是既有 E203 golden 向量（40 样本 A12W10 参数包，参数 SHA
  `10c481c7...`），与本配置参数包 SHA 不同、协议相同；`verify-vectors` 校验向量文件
  SHA 后逐位通过。既有 RTL 路径为 NICE-like 仿真模型，非组委会正式 SoC/PDK。
- 若评测环境无法拿到官方 FP32 模型/数据资源，只能按配置哈希核对，**不得伪造 NRMSE**。
- 未综合、未 STA/功耗、未跑 E203 全仿真（本轮为数值协议冻结任务）。

## 合规

- 仅新增 `software/submission_model/**`；未改 `export_ffn1_quantized.py`、`quantize_ffn1.py`、
  `calibrate_blackbox_ffn1.py`、RTL、驱动、综合或主目录文件。
- 未提交大型权重/样本；`.build/` 与 `__pycache__/` 被忽略。
- 训练划分只用于既有参数包的标定（本轮未重新标定）；验证划分只评估；未使用测试划分。
