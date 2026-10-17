# FFN1 架构实验：可复用仿真与综合

这个目录以赛题的 `99×64 × 64×256 + bias` 为任务边界，比较现有 8×16 乘积阵列与“8 行 MAC + 一个 B 字广播 + 8 个部分和 bank”的方向。题目约束、原设计的重复搬运和假设见 [数据流审视](ffn1_dataflow_review.md)。这里统一使用 INT16 A/W、INT32 环绕累加和偏置；它只验证这一路整数运算，不代表十二层的量化精度。

若只讨论 MAC 流水与每拍吞吐，先看[不依赖存储/总线模型的连续计算实验](compute_only_pipeline.md)。后文的任务周期含测试台搬运假设，不能用于决定目标平台吞吐。

## 运行

在仓库根目录：

```bash
python3 hardware/architecture/run_row_broadcast.py \
  --out /tmp/kwt-row-architecture-full
bash hardware/architecture/run_compare_synth.sh
```

仿真驱动编译一次 Verilator，然后跑小尺寸、非整 tile（附带 valid 间隙）、有符号溢出压力测试、完整赛题尺寸。逐项与 Python 独立计算的 INT32 结果比较，还检查补零位置与访存计数。`--quick` 跳过完整赛题尺寸；`--trace` 生成 VCD；`--case path/to/case.json` 读外部向量，字段为 `format: "int16"`、`a[M][K]`、`b[K][N]`，可选 `bias[N]`、`expected[M][N]`、`gaps`。已有 `hardware/testbenchplus/vectors/layer0_int16/case.json` 可作为外部向量示例。日志和逐项指标保留在 `--out` 路径。

综合脚本先保留 `$mem_v2` 统计乘法器、加法器和部分和 bank；再用默认 Nangate45 Liberty 将这些小存储映射成触发器，取得一个**仅供同库相对比较**的标准单元面积。可以用 `LIBERTY=/path/to/library.lib` 指定其他库，`SYNTH_OUT=/path/to/output` 指定输出目录，`--coarse-only` 跳过单元映射。两个综合顶层都只含乘积生成、部分和累加及输出偏置加法；不含 A/W 缓存、DMA、任务控制或输出暂存。

## 2026-10-16 实测

| Verilator 向量 | 结果 | 拍数 | 共享数据字节 | 有效输出 |
| --- | --- | ---: | ---: | ---: |
| 3×4 × 4×5 | PASS | 451 | 144 | 15 |
| 10×7 × 7×19，插入 valid 间隙 | PASS | 2,160 | 1,242 | 190 |
| 8×64 × 64×16，含 INT32 环绕溢出 | PASS | 3,180 | 3,648 | 128 |
| 99×64 × 64×256 | PASS | 327,920 | 147,840 | 25,344 |

完整尺寸中，初次装载 7,872 拍，计算及 A 预取 215,696 拍，串行读出 CACC 与写回 103,936 拍，其余为 tile 清空和边界。`147,840 B` 包含 A、W、bias 各读一次和有效输出各写一次。当前测试台的 **A/W/bias 装载、W 双页预装和输出写回都是按 16 位一字一拍的定时模型**，不是已实现的 DMA、共享总线或 E203 控制 RTL；327,920 拍不能用于赛题性能申报，也不应直接与现有 E203 系统模型的拍数作速度比值。

| 计算核综合顶层 | `$mul` | `$add` | `$mem_v2` bank | Nangate45 标准单元面积（库单位） |
| --- | ---: | ---: | ---: | ---: |
| 现有 8×16 阵列 + CACC + bias | 128 | 9 | 8 | 355,178.628 |
| 8 行广播 + CACC + bias | 8 | 9 | 8 | 60,358.060 |

Yosys 两种映射均 `check -assert` 通过。后一项计算核标准单元面积约为前一项的 17%，主要原因是每拍只生成 8 个乘积。**这不是整机面积**：部分和 bank 在这次映射中被做成触发器，未使用题目规定的 SRAM 宏，也没有时钟约束、STA、功耗或物理实现证据。8 个 32 位 bank 同拍各更新一个元素；若以 16 位宽 SRAM 宏实现，每个 bank 的读写带宽和拼宽都需要落实，不能只按 512 B 容量估算宏数量。

## 下一步架构判断

现有定时模型里，串行导出与写回占完整任务的约 31.7%。这给了“**先把 CACC 结果搬到输出暂存，再在下一 tile 计算时由 DMA 写回**”明确的优化目标。单页 CACC 仍需花时间导出 128 个结果；若每拍读一个约为 128 拍/tile，之后便可清空并计算下一 tile。写回 256 个 16 位字可在下一 tile 的约 1,024 拍计算期内进行，因此先评估一个独立输出暂存和总线仲裁，再考虑复制整套 CACC。W 的下一列组预装也占用共享总线，须与输出 DMA 排时隙。上述是由实测阶段计数推得的**设计假设**，尚需真实控制 RTL 与相同任务仿真验证。

原 8×16 引擎的回归命令是 `bash hardware/rtl/pulse_array/tb/run_pingpong_cacc_verilator.sh`；它已验证单 tile 乒乓与 CACC 功能，但与这里的完整任务定时模型不是同一比较边界。

## 冻结量化参数驱动 RTL：速度下界与一次真实输入

先运行 `python3 hardware/architecture/speed_estimate.py`。每层有 1,622,016 个有效 MAC。若 128 个乘法器真的每拍都能拿到操作数，纯计算下界是 **12,672 拍（0.0845 ms，150 MHz）**；这要求每拍维持 128 个乘积及相应 CACC 输入带宽，现有单口缓存供数不满足。按当前“一个 B 字/拍广播”方式，计算下界为：

| 行并行数 | 含补零的计算拍 | 150 MHz 纯计算时间/层 | A 预取最低读字/拍 | 同拍部分和更新 bank |
| ---: | ---: | ---: | ---: | ---: |
| 8（已实现） | 212,992 | 1.420 ms | 1 | 8 |
| 16（计算核已实现） | 114,688 | 0.765 ms | 1 | 16 |
| 32（待实现） | 65,536 | 0.437 ms | 2 | 32 |

这些数没有加初次装载、输出、CACC 导出、控制、总线等待或流水间隙。按未紧密打包的 INT16 A/W、INT32 bias/output，每层至少搬运 147,840 B；若整个共享数据路径只有一个 16 位字/拍，仅搬运就需 73,920 拍，理想 128 位/拍则需 9,240 拍。完美紧密打包 A11W10 可降至 131,592 B、单 16 位字/拍下界 65,796 拍，但还需解包、对齐和仲裁。实际赛题总线宽度和仲裁时序未提供，这些条件都不是平台实测。所有时间还以 **150 MHz 达标** 为前提，目前没有目标 SRAM 宏和完整 STA 证明。

`run_quantized_ffn1_rtl.py` 从已冻结的 A11W10 参数包取 12 层权重/偏置，选一条验证语音经 FP32 模型生成真实 FFN1 输入，编码后送入 **Verilator 执行的 `kwt_row_broadcast_tile` RTL**。测试台核对每个 INT32 输出，另将 RTL 输出文件读回，与整数参考逐项比较，再解码与 FP32 参考求 NRMSE。运行：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python3 hardware/architecture/run_quantized_ffn1_rtl.py \
  --out /tmp/kwt-quantized-rtl
```

本次第 0 条验证样本（`_unknown_`）的 **12 层、304,128 个输出全部逐位一致**；单层 327,920 拍，12 层合计 3,935,040 拍。若能闭合 150 MHz，按当前测试台调度相当于 **2.186 ms/层、26.234 ms/12 层**；最差层 NRMSE **0.17041%**。逐层结果见 [`quantized_validation0_12layers.json`](quantized_validation0_12layers.json)。它只是一条验证样本的 RTL 数值检查；240 条验证样本的 0.18216% 汇总值仍来自软件整数模型。当前 DMA、共享总线、E203 调度和缓存端口冲突均未进入这条 RTL 定时结果，因此它不是赛题正式性能或实物时钟测量。

多样本 RTL 精度复测使用 `run_quantized_ffn1_batch_rtl.py`：只编译一次 Verilator，
然后让选定的验证样本逐个经过 12 层 `99×64 × 64×256 + bias`。每个 INT32
输出与独立整数 GEMM 逐位比对；按层汇总所有样本的平方误差和 FP32 参考
平方和，最后取 12 层中的最大 NRMSE。默认从 12 个类别各选一条验证样本，
`--per-label 0` 可遍历本地 240 条验证样本；输出目录保存报告和校验值，
默认删除逐任务 hex 与输出文本，避免长时间运行占满磁盘。运行：

```bash
python3 hardware/architecture/run_quantized_ffn1_batch_rtl.py \
  --per-label 1 --rows 16 --staged --out /tmp/kwt-ffn1-batch-rtl
```

这一路仍用测试台模拟输入/权重装载与共享总线；不能把它当作真实 E203/DMA
全链路、正式 SRAM 宏时序或官方 600 条评测结果。
2026-10-17 的本地全量复测覆盖 240 条验证样本和 12 层，共
**72,990,720 个 RTL 输出逐位一致**；第 11 层最差 NRMSE 为
**0.18216320%**，与软件标定结果相符。详见[精度复测记录](quantized_validation240_rtl.md)
和[逐层摘要](quantized_validation240_12layers_rtl_summary.json)。

同一测试台用 `--rows 16 --out /tmp/kwt-quantized-rtl-row16` 运行 16 行参数化计算核，也逐位核对了同一条语音的 12 层。单层 **233,168 拍**，12 层 **2,798,016 拍**；条件性 150 MHz 换算为 **1.554 ms/层、18.653 ms/12 层**，比 8 行总任务模型快约 29%。阶段计数显示计算 117,040 拍、串行导出 108,032 拍，所以乘法并行度翻倍后导出几乎与计算一样长。逐层结果见 [`quantized_validation0_12layers_row16.json`](quantized_validation0_12layers_row16.json)。Nangate45 示例库将 16 行核的部分和存储展开成触发器后，得到计算核面积 120,434.692 库单位，对应 8 行核的 60,358.060；两者均不含 A/W 存储、DMA、输出暂存或 SRAM 宏，不能直接判断是否满足 2.25 mm²。

## 16 行 CACC 输出暂存：硬件模块与任务级复测

新增可综合 [`kwt_tile_output_stage.sv`](../rtl/pulse_array/kwt_tile_output_stage.sv)：逐拍读出一个 16×16 CACC tile，叠加同步读取的 INT32 bias，将有效结果写入一个 256×32 bit（1 KiB）暂存区，再经 `out_valid/out_ready` 输出。下游反压时地址和数据保持；暂存区仍在写回时，下一 tile 可先计算，但下一次 CACC 导出要等暂存区清空。`tb_kwt_tile_output_stage.sv` 检查了完整 tile、3×5 边界 tile 和反压稳定性。Yosys `memory_collect` 识别一个 `$mem_v2`，`check -assert` 通过。

运行包含暂存的完整任务模型：

```bash
bash hardware/architecture/run_output_stage.sh
python3 hardware/architecture/run_row_broadcast.py --rows 16 --staged \
  --out /tmp/kwt-staged-row16
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  python3 hardware/architecture/run_quantized_ffn1_rtl.py --rows 16 --staged \
  --out /tmp/kwt-quantized-staged-row16
ROW_BROADCAST_ROWS=16 SYNTH_OUT=/tmp/kwt-synth-row16 \
  bash hardware/architecture/run_compare_synth.sh --coarse-only
```

2026-10-16 复测结果：16 行完整任务 **154,240 拍/层**，较原串行写回模型 233,168 拍少 **78,928 拍（33.85%）**；按条件性的 150 MHz 是 1.028 ms/层。12 层同一条真实验证语音共 **1,850,880 拍、304,128 个输出逐位一致**，最差层 NRMSE 仍为 **0.17041%**；报告见 [`量化复测`](quantized_validation0_12layers_row16_staged.json)和[`任务阶段计数`](staged_task_row16_summary.json)。完整任务中 24,297 个结果在下一 tile `in_valid` 时完成下游握手，说明写回与计算确实在此模型中重叠。W 下一页预装和输出写回不共享同一拍的单字总线时隙；行/权重/偏置的初装、A 预取、双页 W 控制、总线应答及 E203 发令仍由测试台建模，并非可综合任务控制器。

暂存区增加了明确的面积代价：若按 Nangate45 示例库**把 1 KiB 暂存器全部展开成触发器**，此模块单独面积为 **66,224.956 库单位**，相对于同库 16 行计算核的 120,434.692 不可忽略；这些数不能简单当作最终整机面积。赛题指定的是 16 bit 宽 SRAM 宏，256×32 bit 暂存需落实拼宽/深度/bank/端口和读写仲裁，随后用目标宏重新评估。以上拍数也仍未包含正式共享总线等待、目标 SRAM 延迟、软件驱动和目标时序，不能直接申报赛题吞吐。

16 行的 A 值按一个 K 保持 16 个寄存器，在随后 16 个列拍中复用；这 16 拍内每拍从单口 A 存储预取下一 K 的一行字。故稳态只需 **A 一字/拍、B 一字/拍**，而非从 SRAM 每拍读 16 个 A 字。测试台已经调度并计数这种预取，**尚无可综合 A 缓存控制 RTL 或目标宏端口验证**。输出暂存已在当前抽象任务模型中证明可与下一 tile 计算重叠；后续仍须用实际共享总线、宏端口和任务控制 RTL 复测，确认 W 预装与写回仲裁后仍有收益。

外部 Claude 只读评审及对其关键判断的逐项核实见 [`claude_review_check.md`](claude_review_check.md)。
