# verilator_extended — FFN1 12 层 Verilator 差分/边界回归

本目录只新增验证工程，不改动核心 RTL、现有脚本，也不涉及 16 行 E203 集成。
目标是给本科必做对象——**12 套独立 FFN1 线性层（每层 `99×64 @ 64×256 + bias`）**
——提供可复现的 Verilator 差分与边界回归，**期望值由独立整数模型给出，绝不从
DUT 输出反造**。

## 被测对象

- DUT：`hardware/rtl/pulse_array/kwt_pingpong_cacc_engine.sv`（8×16 tile，
  INT16 A/W、INT32 累加/偏置/输出，乒乓 A/B 页 + 行 bank CACC）。
- Testbench：`tb_kwt_extended_regression.sv`，单次仿真只跑一个 `+MODE` 用例，
  只负责加载外部激励并转储原始 32 bit 输出。
- 参考模型：`run_extended_verilator.py` 用 NumPy 整数 GEMM 独立计算
  `Y[r,c] = Σ_k a[k,r]·b[k,c] + bias[c]`，再按 `row_mask/col_mask/relu` 掩码。

冻结参数来自已入库的 `software/results/ffn1_a12w10_static_sym.npz`
（12 层 `layer01..layer12` 的 10 bit 权重与 INT32 偏置）。激活是脚本生成的
**确定性合成激励**（12 bit 有符号范围内），不是真实音频激活；这一点在证据层级中
明确标注。整数公式与既有 `run_quantized_ffn1_batch_rtl.py` 一致。

## 覆盖

| 类别 | 用例 | 说明 |
| --- | --- | --- |
| 正常 tile | `L01_first`…`L12_first`, `*_middle` | 12 层各取首 tile 与中间 tile，满掩码 |
| 末 tile / 输出填充 | `*_last` | 行 96..98 有效、99..103 填充为 0；列 240..255 |
| K 边界 | `B_k1` | `k_len=1` 最小归约 |
| 列填充 | `B_colpad` | `col_mask=0x000F`，列 4..15 必须为 0 |
| ReLU 掩码 | `B_relu` | 负结果必须被清零 |
| reset 边界 | `B_reset` | 计算中拉低 `rst_n`，复位期间 `done/busy` 必须为低，复位后重跑需逐位正确 |
| 握手/反压 | `B_pingpong` | 忙时写活跃页必须被 `load_ready` 拒绝；空闲页可预装，第二个 tile 复用同一期望值 |

共 **41 个用例、5376 个逐位比较输出**（其中 4107 个期望值非零，防止空激励）。

## 运行

```bash
bash hardware/verification/verilator_extended/run_extended.sh /tmp/kwt-verilator-extended
```

等价的直接命令：

```bash
python3 -u hardware/verification/verilator_extended/run_extended_verilator.py \
  --out /tmp/kwt-verilator-extended
```

脚本用 Verilator 编译一次，然后逐用例启动仿真、转储 `out.hex`、比较并写
`summary.json`。

## 负例注入（证明检测器会失败）

1. 脚本内部的期望值污染：把首个用例某个期望元素 +1，比较器必须报出
   1 处失配（`negative_injection.detected=true`）。
2. DUT 侧外部注入：

```bash
python3 -u hardware/verification/verilator_extended/run_extended_verilator.py \
  --out /tmp/kwt-verilator-inject --inject-output L12_last 0 0
```

该命令在比较前把 `L12_last` 的第 (0,0) 个 DUT 输出 +1，脚本以退出码 1 结束并
打印 `MISMATCH (0, 0, 2165842, 2165841)`。

## 证据层级与限制

- 这是 **tile 引擎级 Verilator 仿真**：不含 E203、DMA、共享总线、SRAM 宏、
  综合、时序、面积或功耗；不是组委会 SoC 或综合后网表结果。
- 只覆盖每层 3 个 tile，不等价于 99×256 整层结果，也没有遍历 12 层整层。
- 激活为合成模式；权重/偏置为冻结 A12W10 参数。
- `summary.json` 记录了 Verilator 版本与参数包 SHA256。
