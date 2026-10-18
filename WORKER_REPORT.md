# WORKER_REPORT — FFN1 12 层 Verilator 差分/边界回归

工作树：`/home/yang/kwt-worktrees/verilator-extended`（分支 `codex/kwt-verilator-extended`）
范围：只在**新目录** `hardware/verification/verilator_extended/` 增加验证工程；不改
核心 RTL、现有脚本、主目录或其他代理负责的 16 行 E203 集成。

## 新增文件

- `run_extended_verilator.py`：编译一次 `kwt_pingpong_cacc_engine`，逐用例驱动
  Verilator，并用 NumPy 整数 GEMM 独立计算期望值后逐位比较；含负例注入。
- `tb_kwt_extended_regression.sv`：单用例 testbench，只加载外部激励、转储原始
  32 bit 输出，不产生期望值。
- `run_extended.sh`：入口包装脚本。
- `README.md`：覆盖项、运行方式、证据层级与限制。

## 对象与覆盖

本科必做对象为 **12 套独立 FFN1 线性层，每层 `99×64 @ 64×256 + bias`**。
冻结权重/偏置取自已入库的 `software/results/ffn1_a12w10_static_sym.npz`
（`layer01..layer12`，SHA256
`a77f60fb7c8560adbca73db3cf64dcd5501ec0108b96a1131a00aa722be1918e`）。

- `L01..L12`：每层首 tile、中间 tile、末 tile（行 96..98 有效、99..103 输出填充为 0）= 36 例。
- 边界：`K=1` 最小归约、`col_mask` 列填充、`relu` 掩码、计算中注入 reset、乒乓
  反压/握手 = 5 例。
- 合计 **41 个用例、5376 个逐位比较输出**，其中 4107 个期望值非零。
- 期望值由独立整数模型 `Y = Σ_k a·b + bias` 按掩码计算，**不从 DUT 输出反造**。

## 实际命令、退出码与结果

| 命令 | 退出码 | 结果 |
| --- | ---: | --- |
| `bash hardware/verification/verilator_extended/run_extended.sh /tmp/kwt-verilator-extended` | 0 | `PASS 41 cases, 5376 outputs, 0 mismatches`；内部期望值污染被检出 1 处 |
| `python3 hardware/verification/verilator_extended/run_extended_verilator.py --out /tmp/kwt-verilator-inject --inject-output L12_last 0 0` | 1 | 负例注入成功：`MISMATCH (0, 0, 2165842, 2165841)` |

工具版本：`Verilator 5.030 2024-10-27 rev v5.030`。参数包与 Verilator 版本写入
`summary.json`。

非空激励抽检：`L01_last` 行 0..2 非零 48 项、行 3..7 全为 0；`B_colpad` 列 4..15
全为 0；`B_relu` 无负输出。

## 证据层级与未证事项

- 证据为 **tile 引擎级 Verilator 仿真**；不含 E203、DMA、共享总线、SRAM 宏、
  综合、时序、面积或功耗，非组委会平台或综合后网表结果。
- 每层只跑 3 个 tile，不等价于 99×256 整层，也未遍历 12 层整层（有意避免长仿真）。
- 激活为确定性合成模式，不是真实音频激活；权重/偏置为冻结 A12W10 参数。
- `--inject-output` 验证“检测器能对错误 DUT 输出报错”，不是对硬件的故障注入。
