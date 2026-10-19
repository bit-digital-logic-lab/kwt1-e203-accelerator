# 研究生组 5–6 项：单层 PostNorm 编码器组装 + 十二层参数更新/搬运控制

本目录（`graduate/encoder/`）与 `graduate/rtl/encoder/` 是把 **grad-attention（第 4 项）**、
**grad-nonlinear（第 5 项）**、**grad-golden（第 4–6 项 FP32 golden）** 的成果**组装**成
一条可运行的编码器链，并补上缺失的 **FFN1 行引擎**、**层间编排**与**十二层参数更新/搬运
控制器**，明确 E203 软件边界与分类头软件边界。全部数值计算在 RTL，TB 只搬运与记录。

对应题目 [`题目/640.jpg 9`](../../题目/640.jpg%209)：第 5 项（单层编码器：注意力 + 两处残差 +
两处 LayerNorm + 64→256→64 前馈 + GELU，PostNorm）与第 6 项（顺序执行 12 层、层间数据传递、
逐层参数更新；最终编码器输出送统一软件分类头）。

## 目录

```
graduate/rtl/encoder/
  kwt_encoder_layer.sv          单层 PostNorm 编码器数据通路（组装）
  kwt_ffn1_row.sv               FFN1 行引擎 64->256（INT48 累加，per-channel scale）
  kwt_encoder_layers_ctrl.sv    12 层参数更新 / 层间搬运 / 分类头边界控制器
  kwt_encoder_top.sv            layer + ctrl 顶层（E203 事务边界）
  third_party/                  只读复制（见 PROVENANCE.md）
graduate/encoder/
  README.md  SOFTWARE_BOUNDARY.md  PROVENANCE.md
  model/attn_int_model.py       复制自 grad-attention（路径适配）
  model/kwt_postnorm_ref.py     复制自 grad-nonlinear（路径适配）
  model/verify_encoder.py       逐级对照（**本次新增**：组合独立整数模型 + FP32 golden）
  reference/kwt1_fp32_golden.py 复制自 grad-golden（路径适配）
  vectors/gen_vectors.py        **本次新增**：生成全部 RTL 激励/参数 hex
  tb/tb_encoder.sv              **本次新增**：单层逐级对照 TB（Verilator）
  tb/tb_layers_smoke.sv         **本次新增**：12 层调度 smoke TB
  tb/run.sh  tb/run_smoke.sh    一键复现
```

## 组装后的数据流（单层，PostNorm，顺序取自官方 `transformer_utils.py`）

```
X[99,64] --attn_single_head--> o  --requant(scale_o->scale_a)--> a
 r1 = x + a                               (kwt_postnorm_ffn 残差1, RTL)
 out1 = LN1(r1)                           (kwt_postnorm_ffn LayerNorm1, RTL; A12W10 scale sx)
 h = out1 @ W1 + b1                       (kwt_ffn1_row, RTL; per-channel scale s_h[c])
 g = gelu(h)                              (kwt_postnorm_ffn GELU LUT, RTL)
 f = g @ W2 + b2                          (kwt_postnorm_ffn FFN2, RTL)
 r2 = out1 + f                            (残差2, RTL)
 y  = LN2(r2)                             (LayerNorm2, RTL)  -> 下一层输入 / 分类头
```

`kwt_encoder_layer` 用统一装载端口把 X、注意力 Wq/Wk/Wv/Wo/bias/LUT、FFN1 W1/b1、
FFN2 W2/b2 灌入各子模块；行流为“注意力一次算完 99×64 → 逐行 residual1→LN1→FFN1→
GELU→FFN2→residual2→LN2”。

## 复现与实测（实际退出码）

```bash
bash graduate/encoder/tb/run.sh         # 单层逐级对照：向量→Verilator→逐位+FP32 校验
bash graduate/encoder/tb/run_smoke.sh   # 12 层调度 smoke
```

| 命令 | 退出码 | 结果 |
| --- | ---: | --- |
| `bash graduate/encoder/tb/run.sh` | 0 | `TB_PASS` + `VERIFY_PASS`；6 个阶段 99 行全 **逐位一致** |
| `bash graduate/encoder/tb/run_smoke.sh` | 0 | `SMOKE_PASS`；12 次启动、layer 0..11、1188 行输出、每层 49,600 项参数重载、head 边界 1 次 |

单层逐级对照（真实样本 `sample_00`，第 0 层，99 行）：

| 阶段 | 逐位对照（vs 组合独立整数模型） | FP32 NRMSE（反量化 vs 研究生 FP32 golden） |
| --- | --- | ---: |
| `a`（注意力输出，含 requant） | 99×64 bit-exact | 0.1591% |
| `r1`（残差1） | 99×64 bit-exact | 0.1082% |
| `g`（GELU） | 99×256 bit-exact | **1.3987%** |
| `f`（FFN2） | 99×64 bit-exact | 1.3968% |
| `r2`（残差2） | 99×64 bit-exact | 0.7000% |
| `y`（LN2，编码器输出） | 99×64 bit-exact | 0.7289% |

最差阶段 `g` = **1.40% NRMSE < 3%**（误差以 GELU LUT 插值为主，与 grad-nonlinear 一致）。

**关键量化尺度**：`scale_x=0.00135625`（X 与注意力同一张量）、`scale_o=2.475293e-05`
（注意力输出）、`scale_a=2.712451e-04`（postnorm `a` 域，= max|attn_out|/2047）、
requant 系数 `M_A=5981, Q_A=16`。FFN1 参数直接取冻结 A12W10 包
`software/results/ffn1_a12w10_static_sym.npz` 的 `layer01_*`，FFN2 参数由
`kwt_postnorm_ref.calibrate()` 按同一规则重算，保证 `h` 与现有 FFN1 整数输出一致。

## 十二层调度 smoke（`SMOKE_PASS`）

`kwt_encoder_layers_ctrl` 每层按固定事务顺序发起：`sel1 Wq(4096)、sel2 Wk(4096)、sel3 Wv(4096)、
sel4 Wo(4096)、sel5 bias(64)、sel6 exp_lut(64)、sel7 W1(16384)、sel8 b1(256)、sel9 W2(16384)、
sel10 b2(64)`，再 `sel0 X(99×64)` 写回，然后 `start`；`done` 后把 y 抓回 x 缓冲，作为下一层输入；
第 11 层后拉起 `head_valid/head_token`。实测：`n_start=12, max_layer=11, yrows=1188,
n_ld_par=595200 (=12×49600), head=1`。

> 该 smoke 只声明**调度正确性**（命令顺序、逐层参数重载、层间数据搬运、分类头边界）；
> 第 1–11 层用的是第 0 层的尺度与权重 ROM，**不声明 12 层数值正确性**（其余层尺度未在此标定）。
> 详见 `SOFTWARE_BOUNDARY.md`。

## 未证事项（不得当作完整研究生组验收）

1. **无官方 SoC**：没有官方 E203 + System Bus / NICE + SRAM 宏 + PDK + SDC。本设计只做
   **E203-like** 命令/搬运模型（见 `SOFTWARE_BOUNDARY.md`），未在真实 E203 指令流中执行。
2. 未做目标工艺综合与 150 MHz STA、面积、功耗；只做了 Verilator 功能仿真。
3. 仅第 0 层的尺度被标定；12 层数值正确性、官方 600 条评测集 NRMSE/准确率均未测。
4. FFN2/注意力的大数组与组合读端口未映射到题目规定的 SRAM 宏，仍是行为存储。
5. 分类头本身是**软件**阶段（题目规定）；硬件只提供 `head_token`（类 token 行）与
   `head_valid` 边界，未做 12 路 Dense 的 RTL。
6. `a` 阶段的逐位一致依赖注意力整数模型 `exp_o`；注意力 RTL 自身已由 grad-attention
   独立验证（41,481/41,481），本目录不重跑其 full TB。
