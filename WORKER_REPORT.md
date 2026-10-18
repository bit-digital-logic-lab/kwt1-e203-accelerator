# WORKER_REPORT — codex/kwt-grad-nonlinear（研究生单层编码器：GELU/FFN2/残差/LayerNorm）

- 分支：`codex/kwt-grad-nonlinear`（worktree `/home/yang/kwt-worktrees/grad-nonlinear`）
- 只在本 worktree 内操作；未开子智能体；未改主目录；未碰任何已有 RTL（新增目录之外零改动）；
  未 merge/rebase/cherry-pick/push。
- 新增文件全部在 `graduate/rtl/postnorm_ffn/`：**RTL 原型 + 独立整数/浮点参考**，
  对应 [`题目/640.jpg 9`](题目/640.jpg%209) 研究生必做第 5 项中的
  GELU、FFN2 256→64、两处残差、两处 LayerNorm（PostNorm）。
- 根目录新增 `ACCEPTANCE.json`（一键复现契约）。
- 本文件覆盖了此前的整合分支报告；其内容已归档在 `docs/worker_reports/kwt-integration.md`，未丢失。

## 1. 交付内容

| 文件 | 说明 |
| --- | --- |
| `SPEC.md` | 冻结数值规范：PostNorm 顺序、全部量化尺度、整数数据通路、舍入/饱和规则 |
| `rtl/kwt_postnorm_pkg.sv` | `mul_sr`（round-half-up 有符号乘移）、整数 `isqrt`、对称饱和 |
| `rtl/kwt_residual_add.sv` | 残差加法（逐通道尺度对齐 + 饱和 + `out_sat`） |
| `rtl/kwt_layernorm.sv` | `d_model=64` 整数 LayerNorm（均值/总体方差/isqrt/1/√var/仿射） |
| `rtl/kwt_gelu.sv` | Q8 归一 + 257 节点插值 GELU LUT + 反量化 |
| `rtl/kwt_ffn2.sv` | 256→64、INT32 累加、权重/bias 同步装载端口 |
| `rtl/kwt_postnorm_ffn.sv` | 顶层：残差1→LN1→(FFN1)→GELU→FFN2→残差2→LN2，valid/ready 行级流 |
| `rtl/kwt_postnorm_cfg.svh` | 由 `gen_config.py` 生成的系数与 GELU LUT（已提交，可再生成且逐字节一致） |
| `model/gen_config.py` | 用真实 checkpoint + 真实样本标定，导出系数与测试向量 |
| `model/kwt_postnorm_ref.py` | **独立整数参考模型** + FP32 参考（NumPy 复刻 `software/run.py`） |
| `model/verify.py` | RTL dump 与整数模型逐位比对 + FP32 误差单列 + 方差覆盖断言 |
| `tb/tb_kwt_postnorm_ffn.sv`、`tb/run.sh` | 测试台（109 行）与驱动（Verilator / Icarus 二选一） |
| `tb/run_yosys.sh` | Yosys 结构检查（`check -assert`） |
| `ACCEPTANCE.json` | 编排者独立复跑契约（Verilator 仿真 + Yosys + 产物哈希） |

接口：`x`/`a`（64×INT16，层输入/注意力输出）、`h`（INT32，每拍一个隐通道，256 拍/行，
直接接现有 FFN1 输出）、`o1`（64×INT16，LN1 输出→FFN1 装载）、`y`（64×INT16，编码器输出）、
`w_we/w_addr/w_wdata` + `b_we/...`（FFN2 权重/bias 同步装载）。张量存储：GELU 缓冲 256×INT16、
FFN2 权重 16384×INT16（32 KiB 行为 SRAM）、bias 64×INT32。
尺度：A12W10（对称、零点 0、INT16 容器、INT32 累加）；激活按张量、权重按输出通道；
FFN1 参数直接取冻结包 `software/results/ffn1_a12w10_static_sym.npz`（层 1）。

## 2. 实跑命令与退出码（2026-10-19 本次重跑，基于当前工作树）

| 命令 | 退出码 | 结果 |
| --- | ---: | --- |
| `python3 model/gen_config.py` | **0** | 109 行向量 + 系数（`sx=0.00243742`、`sg=0.000956496`、`s_r1=0.00144993`、`s_r2=0.00207154`、`s_out=0.00183158`） |
| `SIM=verilator BUILD_DIR=build_accept bash tb/run.sh` | **0** | `TB_PASS rows=109 cycles=73913 o1_sat=1 y_sat=1`；`VERIFY_PASS`（6 个阶段 109/109 行逐位一致；方差覆盖断言通过） |
| `SYNTH_OUT=build_accept/yosys bash tb/run_yosys.sh` | **0** | `YOSYS_PASS`；`check -assert` 报告 0 problem、未推断 latch；5900 cells、835 `$mul`、2 `$div`、2 memories / 264192 memory bits |
| `SIM=iverilog BUILD_DIR=build_iv bash tb/run.sh` | 见 §5.8 | 独立第二仿真器复核；本机 Icarus 11 解释执行明显慢于早期估计，本次以后台启动，**以最终日志为准，未完成即记为未完成** |

Verilator 5.030 本次 0.265 s（73913 拍）。`build_accept/` 为 §2 命令的产物目录（见 `ACCEPTANCE.json`）。

## 3. 覆盖

- **真实形状 99×64**：99 行来自真实冻结样本 `sample_00_encoder.npy` → 真实注意力（NumPy 复刻
  `run.py`，`heads=1`、`scale=1/sqrt(64)`）→ 真实 A12W10 FFN1 → 本模块。`h` 由 **LN1 输出** 经
  FFN1 整数模型产生（不是层输入），与硬件顺序一致。
- **10 行极值/退化**：`x=+2047`、`x=-2047`、`x=+2047 & a=-2047`、常数行（方差恰为 0）、
  **近零方差**（单通道偏 24 LSB，`r1` 的 `var_q=8`，远小于 `eps_q1≈3.1e4` 与满量程 ~4e6）、
  零行、`h=INT32_MAX`、`h=INT32_MIN`、交替 ±2047、`x=a=+32767`（触发残差饱和）。
  合计 **109 行 × (64/256) 通道 = 6976 个 64 通道批次**。
- **复位**：第 98 行之后拉低 `rst_n` 4 拍，断言期间输出/各阶段快照全为 0 且状态机回到 IDLE
  （否则 `$fatal`），再继续第 99 行起；复位后 10 行仍逐位正确。
- **反压**：`o1_ready`/`y_ready` 由 32-bit LFSR 随机拉低，所有 109 行仍在反压下逐位正确。
- **饱和**：整数模型统计 6976 通道中 r1 饱和 192 个、out1 饱和 1 个、r2 饱和 15 个、y 饱和 1 个；
  RTL 的 `o1_sat_count`/`y_sat_count` = 1/1，与模型一致，无 X。`out_sat`/`y_sat` 计数逐行一致即
  证明饱和路径被真实触发且行为与整数模型相同。

## 4. 逐位对照与 FP32 误差（单列）

`model/verify.py` 逐位比对（109/109 行）：`r1`、`out1`、`g`、`f`、`r2`、`y` **全部逐位一致**；
系数由 verify 独立重算后与烧进 RTL 的配置**逐项相等**；FFN1 隐向量由 verify 从 LN1 输出重算后
与喂给 RTL 的向量**相等**。整数参考答案由 RTL 之外的独立实现给出（自带 restoring `isqrt`、
自带 ties-to-even、自带 LUT 求值），与 RTL 无共享代码。

FP32 误差（整数结果反量化 vs 官方 PostNorm FP32，仅 99 真实行）：

| 阶段 | NRMSE | max_abs |
| --- | ---: | ---: |
| r1（残差1） | 0.0971% | 2.06e-3 |
| out1（LN1） | 0.2894% | 1.95e-2 |
| g（GELU） | 1.3993% | 1.29e-2 |
| f（FFN2） | 1.4211% | 1.57e-2 |
| r2（残差2） | 0.6991% | 4.73e-2 |
| y（LN2） | 0.7288% | 3.87e-2 |

近似误差归因（单列）：`gelu_approx_error`（对同一量化 `h` 的精确 GELU）= **1.32% NRMSE**；
纯 LUT 插值误差（[-8,8)、257 节点、步长 1/16）= **max 5.34e-3 / rms 1.37e-3**。
即 GELU 段误差以 LUT 插值为主，其余来自 `s_g` 反量化。

> 过程纠错（本报告如实记录）：初版有两处错误，均由逐位/FP32 对照暴露并已修复——
> (a) LayerNorm 的 `beta` 系数误乘 `2^GOUT_FRAC`（导致 LN 输出全部饱和、FP32 误差 >500%）；
> (b) `r1`/`r2` 调试快照比行晚一拍（dump 取值早一个周期）。修复后 6 阶段全部逐位一致。
> 另：初版 GELU 重标定系数 `QG=16` 时，最小通道尺度 `s_h=8.9e-8` 使 `mg=1`（相对误差 33%），
> 提高 `QG=28` 后 GELU NRMSE 由 3.03% 降到 1.40%。以上均记录在 `SPEC.md`。
> 本次新增 `var_tiny` 行时发现：偏差 2 LSB 经残差缩放后 `var_q` 仍被 floor 归零，改为偏 24 LSB
> 后才得到 `0 < var_q=8`，并由 `verify.py` 的覆盖断言强制守住（`var_zero==0`、`var_tiny∈(0,64)`）。

## 5. 未证 / 缺口（不得当作已通过）

1. **无综合/时序/面积/功耗签核**：无目标 PDK、SRAM 宏、SDC；`run_yosys.sh` 只证明可被
   通用流程读入、无 latch/无 `check -assert` 问题，**不代表 150 MHz 或面积**。
   Yosys 会把 `comb`/`g_buf`/`w2` 等行为数组展开成寄存器（14 条 "Replacing memory" 提示），
   这是行为建模，不是最终 SRAM 映射。
2. **未接入 SoC**：未接 E203/NICE/DMA；无逐层换参时序；无跨行流水（当前约 680 拍/行，
   73913/109≈678）。
3. **FFN2 权重带宽是行为模型**：每拍并行读 64 个权重，真实宏需 64 bank 或时分复用；
   `g_buf`/`w2` 等为行为数组。
4. **仅标定第 1 层**：其他 11 层尺度不同，未标定。
5. **FP32 参考是 NumPy 复刻**（本机未安装 PyTorch），非独立框架运行；未与官方 Keras 交叉。
6. **官方 600 条评测集缺失** → 无官方 NRMSE/准确率结论。
7. **注意力 RTL 不在本次范围**（只作为激励在 NumPy 中复刻）；FFN1 用现有硬件/冻结包，本次未改。
8. **Icarus 第二仿真器复核未完成**：本次以 `SIM=iverilog BUILD_DIR=build_iv` 后台启动全量 109 行，
   在本报告成文时尚未结束；**不据此声明通过**。Verilator 全量结论仅以 `build_accept/sim.log`、
   `build_accept/verify.log` 为准。Icarus 若最终打印 `TB_PASS`/`VERIFY_PASS` 可作为附加证据，
   未打印则视为未完成。
9. 本报告未声明“单层编码器整体通过”：注意力、FFN1 的 RTL 证据不在本目录。

## 6. 复现

```bash
cd graduate/rtl/postnorm_ffn
SIM=verilator BUILD_DIR=build_accept bash tb/run.sh      # 生成向量 → Verilator 仿真 → 逐位/FP32 对照
SIM=iverilog  BUILD_DIR=build_iv     bash tb/run.sh      # 同上的 Icarus 复核（慢）
SYNTH_OUT=build_accept/yosys bash tb/run_yosys.sh        # Yosys check -assert
```

根目录 `ACCEPTANCE.json` 给出编排者独立复跑所需的命令与产物 `sha256`（在 `git archive HEAD`
干净副本中重跑；本 report 不作为通过依据）。
