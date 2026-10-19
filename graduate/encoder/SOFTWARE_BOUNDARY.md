# E203 软件边界与分类头边界（单层编码器 + 12 层调度）

本文件冻结“加速器内部算什么、E203 软件/搬运做什么”的边界。因为**官方 SoC（E203 +
System Bus / NICE + 指定 SRAM 宏 + PDK + SDC）缺失**，本目录只实现一个 **E203-like**
的命令/搬运模型，并在每一处标注“E203-like，未接官方 SoC”。

## 1. 系统分层

```
[E203-like 软件/搬运模型]  ← 本目录的 controller/TB 扮演
      │  命令、参数装载、X 写回、启动、状态查询
      ▼
[kwt_encoder_top]  =  kwt_encoder_layers_ctrl  +  kwt_encoder_layer
      │  数值计算：注意力、requant、残差、LayerNorm、FFN1、GELU、FFN2
      ▼
[head_token + head_valid]  →  [软件分类头]（题目规定为软件阶段）
```

- **数值计算 100% 在 RTL**（本目录 `graduate/rtl/encoder/` 与 `third_party/`）。
- TB 只做：(a) 参数/激励字节装载；(b) 读回结果；(c) 计数与日志。**不做任何乘加、
  Softmax、GELU、LayerNorm、requant**。

## 2. 命令 / 寄存器边界（E203-like）

`kwt_encoder_layers_ctrl` 暴露的硬件接口（E203-like；真实 E203 下应映射为 NICE 指令或
System Bus 寄存器，本实现未接真实总线）：

| 信号 | 方向 | 含义 |
| --- | --- | --- |
| `start` | in | 启动整段 12 层推理（单拍脉冲） |
| `busy` / `done` | out | 运行 / 完成 |
| `o_ld_sel[3:0]` `o_ld_addr[13:0]` `o_ld_we` | out | **参数/数据装载事务**：控制器发出的读请求地址 |
| `p_rdata[31:0]` | in | 事务数据（来自 E203 侧参数/数据区） |
| `layer_id[3:0]` | out | 当前层号 0..11 |
| `head_valid` / `head_ready` | 握手 | 第 11 层完成、分类头边界就绪 |
| `head_token[64×16]` | out | 编码器输出第 0 行（类 token） |

命令序列（每层，由控制器自动产生，软件只需按此地址窗口提供数据）：

```
sel1 Wq(4096) sel2 Wk(4096) sel3 Wv(4096) sel4 Wo(4096)
sel5 attn_bias(64) sel6 exp_lut(64)
sel7 W1(16384) sel8 b1(256) sel9 W2(16384) sel10 b2(64)
sel0 X(99×64)         # 第 0 层为外部输入；第 L>0 层为上一层 y
start ──► busy ──► done
```

- `exp_lut`（`sel6`）是**与数据无关的常量 ROM**（`2^(f/64)·2^14` 系数），由模型生成后灌入；
  装载它是“常数注入”，不是计算。
- 地址编码：注意力存储 `sel0..4` 用 `addr = {row[6:0], col[6:0]}`（128 步长）；`sel5/6` 用
  低位列/行索引；`sel7..10` 用线性索引（W1 `c*64+j`、W2 `k*256+j`）。

## 3. 参数更新边界

- 每个**层**拥有独立参数集（Wq/Wk/Wv/Wo/bias/LUT、W1/b1、W2/b2）。控制器在每层计算前
  **整体重载** `49,600` 项；层与层之间不复用激活参数。
- 参数按层加载（题目“参数按层加载，明确容量、端口、bank、冲突和面积”），本目录只给出
  **容量与事务量**：W1/W2 各 16384×16b = 32 KiB；注意力四矩阵 4×4096×16b = 32 KiB；
  bias 256×32b+64×32b；LUT 64×16b。见 README 存储估算。
- **虽然控制器逐层重新装载参数，但本目录只标定第 0 层尺度**。第 1–11 层的
  量化尺度/EPS/GELU LUT 尚未逐层重算，因此 smoke 的逐层重载是**调度证据**，
  不是“12 层数值已核对”。

## 4. 层间输入/输出边界

- 层输出 `y`（编码器输出，scale `s_out`）在控制器内抓取为下一层的输入缓冲 `x_buf`，
  再通过 `sel0` 写回加速器输入存储。**数据搬运在 RTL 控制器内完成**（E203-like），
  不经过 TB 的算术。
- 层间接口是整行 `99×64`（逐行 64 通道 valid/ready）。真实 SoC 下这里应接
  DMA/共享 SRAM（题目规定 512 KB 系统共享 SRAM 与 16b 宏），本实现用行为存储。

## 5. 分类头边界（软件阶段）

题目规定“最终编码器输出送入**统一软件分类头**”。因此：

- 硬件边界 = 第 11 层完成后的 `head_valid` + `head_token`（编码器输出**类 token 行**，
  即 `x[:,0]`，64 维，scale `s_out`）。
- 分类头 `Dense(64→12)` 由**软件**执行（`reference/kwt1_fp32_golden.py` 的 `linear(x[0], HEAD)`
  给出 FP32 参考），**不在 RTL 本目录范围**。
- 未把分类头移入硬件，也未把任何必做计算移入软件/适配脚本。

## 6. 明确标注为“未接入官方 SoC”

1. 没有真实 E203 指令流、NICE/System Bus 协议、DMA 或 SRAM 宏；`o_ld_*`/`p_rdata` 是
   E203-like 的事务抽象。
2. 没有目标 PDK、SDC、150 MHz 时序与面积/功耗签核。
3. 没有官方 600 条评测集与官方评测脚本，未测 NRMSE/准确率。
