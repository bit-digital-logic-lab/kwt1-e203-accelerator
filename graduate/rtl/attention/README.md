# 研究生必做第 4 项：单头注意力独立可综合 RTL 原型

KWT-1 第 0 层单头自注意力的独立原型：`Q/K/V 64→64` 投影、`99×99` 的
`QKᵀ/√64`、逐行 Softmax、概率加权 V、输出投影。范围与验收对齐题目
[`题目/640.jpg 9`](../../../题目/640.jpg%209) 第 4 项。

- 只使用**真实**权重与真实输入：`software/weights/kwt1_fp32.npz` 的
  `enc_layers/0/att/{query,key,value}_dense/kernel`、`combine_heads/{kernel,bias}`，
  输入 `software/validation/real_input_chain_20261017/sample_00_encoder.npy`
  （`99×64`，即第 0 层编码器输入 = 该层注意力输入）。
- 张量顺序取自官方实现 `资料/ARM_KWT-1_237e4097/kws_streaming/models/transformer_utils.py`：
  Dense kernel 为 `[in, out]`，`score = Q @ Kᵀ`，`scaled = score/√dim_key`，
  `softmax(axis=-1)`，`out = weights @ V`，`combine_heads = out @ Wo + bias`。
  单头 `projection_dim = 64`，`separate_heads/transpose` 为恒等。
- **乘法、QKᵀ、Softmax（含 exp、除法）、加权累加全部在 RTL 内**；testbench 只装载
  输入/参数、读回结果、逐位比较并另算 FP32 误差，不做任何乘加或 Softmax。

## 目录

```
graduate/rtl/attention/
├── rtl/attn_single_head.sv    可综合 RTL（顶层 FSM + 数据通路）
├── rtl/attn_params.svh        由模型生成的常量（移位、Softmax 常数）
├── model/attn_int_model.py    独立整数数值模型 + 向量/常量生成
├── tb/tb_attn.cpp             Verilator C++ testbench（逐位对照 + FP32 误差）
├── tb/run.sh                  模型→Verilator 构建→运行 一键脚本
└── vectors/                   冻结的输入/期望/FP32 参考（文本）
```

## 接口

`attn_single_head`

| 端口 | 方向 | 位宽 | 说明 |
| --- | --- | --- | --- |
| `clk,rst_n` | in | 1 | 同步复位，低有效 |
| `start` | in | 1 | 单拍脉冲启动 |
| `busy,done` | out | 1 | 运行 / 完成 |
| `ld_sel` | in | 3 | 0=X 1=Wq 2=Wk 3=Wv 4=Wo 5=bias 6=exp LUT |
| `ld_row,ld_col` | in | 7 | 装载索引 |
| `ld_data` | in | 32 | 装载数据（16 位有符号按符号扩展，bias 为 INT32） |
| `ld_we` | in | 1 | 装载写使能 |
| `rd_sel` | in | 3 | 0=x 1=q 2=k 3=v 4=p 5=c 6=o 7=bias（**仅验证用**） |
| `rd_data` | out | 32 | 读数据（16 位按符号扩展） |

写端口=输入/参数装载；读端口=纯存储访问，供 TB 逐位核对，不含运算。

## 数值格式（尺度、舍入、溢出、Softmax 近似）

对称定点，零点 0，张量级静态标定（离线，允许）：

| 量 | 容器 | 有效位 | 尺度 | 备注 |
| --- | --- | --- | --- | --- |
| `X` | int16 | 12 | `s_x = max|X|/2047` | 激活 |
| `Wq/Wk/Wv/Wo` | int16 | 10 | `s_w = max|W|/511` | 权重 |
| `Q,K,V` | int16 | ≈15 | `s_q=s_x·s_wq·2^RQ` 等 | 投影后重定标 |
| `P` | int16 | Q15 | `p = P/2^15` | Softmax 概率 |
| `C` | int16 | | `s_ct = s_v·2^(RC-15)` | 上下文 |
| `O` | int16 | | `s_o = s_ct·s_wo·2^RO` | 输出 |

本次冻结样本：`RQ=6, RK=7, RV=7, RC=14, RO=11`，
`KC=6680, SH_KC=38, F=6, M_LUT=14, G=20`（见 `vectors/params.json`）。

- **重定标/舍入**：`rq(v,s) = (v + 2^(s-1)) >> s`（算术右移，负数同式），
  与模型逐位一致；结果按需 `clip16` 到 `[-32768,32767]`。
- **溢出**：累加器按保守上界选宽（投影 32b 足够但 RTL 统一用 64b；分数
  `|S|max≈1.44e10`、上下文 `|ctx|max≈` Q15·int16·99 量级），模型内 assert 上界，
  重定标前不溢出；重定标后显式饱和，不会回绕。累加上界写入本文件与模型注释。
- **Softmax 近似**：
  1. 行内减最大值 `m`，只算 `exp(d)=2^(d·log2e)`，`d≤0`；
  2. `d` 经整数乘 `KC` 右移 `SH_KC` 得 `A≤0`（Q(F)），整数部 `A>>F`、小数部查
     64 项 `2^(f/64)·2^14` LUT，整数部用移位，`-ka≥32` 记 0（下溢近似）；
  3. 行和 `ΣE` 后**每行一次**倒数 `R=(2^(15+G)+ΣE/2)/ΣE`，`P=(E·R)>>G` 得 Q15。
  误差来源：LUT 分段线性近似、倒数的整数截断与 Q15 量化。概率行和
  `∈[32710,32726]`（理想 32768），`|P−P_fp32|` 最大 7.6 LSB（Q15）。

## 分块 / 转置 / 缓存复用

- **转置访问**：`S[i][j]=Σ_d Q[i][d]·K[j][d]` 直接以 `K` 的**行**作被点积向量，
  等价于访问 `Kᵀ` 的第 `j` 列，**不显式物化 `Kᵀ`**，省一次转置与一份存储。
- **Q 复用**：第 `i` 行 `Q` 在计算该行全部 99 个分数时反复使用（行驻留）。
- **V 复用**：上下文阶段 `C[i]=Σ_j P[i][j]·V[j]`，`V` 每行在 99 个 query 行中复用。
- **X 复用**：投影阶段每个输出列共享同一输入行 `X[i]`，`X[i]` 被 64 个输出列复用。
- **C 复用**：输出投影按行复用 `C[i]`（64 列）。
- **融合**：Softmax 与概率加权 V **逐行融合**——产出 `P[i][j]` 的同时累加
  `C[i][d]+=P[i][j]·V[j][d]`，避免第二遍读 99×99 概率矩阵。
- `P` 矩阵在原型中保留仅为验证（可综合 SRAM）；量产可去掉，省约 19.6 KB。
- 暴露性缺口：读端口是组合随机访问，综合会展开为大型多路选择（非 SRAM 宏），
  面积不具代表性，需在接入正式 SRAM 宏时改为 banked 顺序访问。

## 存储 / 周期预算

| 存储 | 字节 | 说明 |
| --- | ---: | --- |
| X | 12,672 | 99×64×16b |
| Wq/Wk/Wv/Wo | 32,768 | 4×64×64×16b |
| bias | 256 | 64×32b |
| Q,K,V | 38,016 | 3×99×64×16b |
| P | 19,602 | 99×99×16b（验证用） |
| C,O | 25,344 | 2×99×64×16b |
| **合计** | **≈128.7 KB** | int16 容器，未打包 |

周期（实测 `cycles=2,957,728`，Verilator）：

| 阶段 | 周期（模型推算，与实测一致） |
| --- | ---: |
| Q/K/V 投影 | 99·64·3·(64+1) = 1,235,520 |
| 分数 QKᵀ | 99·99·(64+1) = 637,065 |
| Softmax（exp+行内归约） | ≈ 99·(99·2 + 99 + 1) ≈ 29,503 |
| 概率加权 V（融合） | 99·99·(1+64) = 637,065 |
| 输出投影 | 99·64·(64+1) = 411,840 |
| **合计** | **≈2,950,993（实测 2,957,728）** |

每拍 1 次乘加（串行核，P=1）。150 MHz 下 ≈19.7 ms/注意力模块（模型推算，非平台实测）。
并行度提高（多 lane/行广播）与 `1/√64` 折入 `KC` 已在设计中体现（`KC` 内含 `/8`）。

## 复现

```bash
cd graduate/rtl/attention
bash tb/run.sh                 # 默认 --sample 0
python3 model/attn_int_model.py --sample 3 && \
  verilator --cc --exe --build -O2 -Wno-fatal --top-module attn_single_head \
    -Irtl rtl/attn_single_head.sv tb/tb_attn.cpp --Mdir obj_dir -o tb_attn && \
  ./obj_dir/tb_attn vectors    # 换真实样本（会重算每个张量的移位）
```

退出码：模型 0 / 构建 0 / TB 0 即通过。

## 验证结果（本次实跑）

在**两个真实样本**上端到端逐位对照，两个样本的量化移位不同（证明 RTL 与模型对
移位配置一致）：

| 样本 | 移位 RQ/RK/RV/RC/RO | KC/SH_KC | 逐位一致 | FP32 NRMSE |
| --- | --- | --- | --- | ---: |
| `sample_00`（默认） | 6/7/7/14/11 | 6680/38 | 41481/41481 | **0.15886%** |
| `sample_03` | 6/7/**6**/**15**/11 | **4815/37** | 41481/41481 | 0.21102% |

- 逐位：每样本 `q,k,v,c,o` 各 6336/6336、`p`（Softmax 全 99×99）9801/9801，
  合计 **41481/41481，0 失配**。
- 覆盖：QKV 全部；分数/Softmax 全 99×99（`p`）；首行/末行单列 NRMSE 单列如下
  （sample_00）。

FP32（float64 参考，另列，sample_00）：

| 指标 | 值 |
| --- | ---: |
| NRMSE（全体） | 0.15886% |
| NRMSE 首行 | 0.15766% |
| NRMSE 末行 | 0.15077% |
| max abs err | 0.00124016 |
| 参考 RMS | 0.21848 |

- 前端可综合性：`yosys 0.63 read_verilog -sv; hierarchy -check; proc; check` 退出码 0。

## 未做 / 缺口（不得宣称整项通过）

1. 未接入 E203 / NICE / System Bus，无 SoC 集成与驱动，无复位/连续任务/边界系统级覆盖。
2. 未做目标 PDK 综合与 150 MHz STA；组合除法器、64b 乘法器、组合读端口未做时序优化，
   面积/功耗未测。
3. 组合读端口与大型数组未映射到题目规定的 SRAM 宏；`P` 存储未去掉。
4. 仅 1 个真实样本（第 0 层）。未验证 12 层、多样本、量化标定泛化。
5. Softmax 近似误差只在本样本统计；不同层/样本的尺度需重新标定并重跑。
