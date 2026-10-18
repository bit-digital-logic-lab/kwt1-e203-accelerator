# FFN1 数值协议（冻结）— KWT1-FFN1-A12W10-SYM-v1

本文件冻结本科必做第 2 项的**唯一**数值协议：12 个相互独立的线性任务

```
Y[99,256] = X[99,64] @ W[64,256] + bias[256]
```

GELU、第二线性层、注意力不属于本科必做，协议在此处停止。
`X[layer]`、`W[layer]`、`bias[layer]` 由官方 FP32 KWT-1 在电脑端离线提取，
硬件只做上式的整数乘累加与加偏置。软件参考入口为
[`ffn1_reference.py`](ffn1_reference.py)，配置 ID 与哈希见
[`config/KWT1-FFN1-A12W10-SYM-v1.json`](config/KWT1-FFN1-A12W10-SYM-v1.json)。

## 0. 冻结配置与来源

| 项 | 值 |
| --- | --- |
| 配置 ID | `KWT1-FFN1-A12W10-SYM-v1` |
| 模型 | ARM `keyword-transformer` `237e4097`，`models_data_v2_12_labels/kwt1/best_weights` |
| FP32 权重包 | `software/weights/kwt1_fp32.npz`，SHA-256 `8f359abf7adbe287b8f89fca98af46194f98f90f2146784731bff551a155825a` |
| 冻结整数参数包（A12W10） | `software/results/ffn1_a12w10_static_sym.npz`，SHA-256 `a77f60fb7c8560adbca73db3cf64dcd5501ec0108b96a1131a00aa722be1918e` |
| 标定样本包 | `software/.build/quant_samples_12labels_20.npz`，SHA-256 `ee04e849bc407149618e4ecd5ff99575687586d4e57c0ade6a0b3f7b66c8c149`（训练 240 / 验证 240，12 类各 20） |
| 风格 | A12W10 有符号对称、零点 0；激活逐层一个尺度，权重逐输出通道一个尺度 |
| 累加器 | 物理有符号 INT32，按二补码环绕；偏置 INT32 |

标定只用 `training` 划分；`validation` 划分只评估，绝不参与任何尺度搜索。
仓库禁止 `testing/` 或其它划分进入本脚本。

## 1. 张量、形状与端序

| 张量 | 形状 | 逻辑元素类型 | 说明 |
| --- | --- | --- | --- |
| `X` | `[99,64]` | FP32（离线输入）/ 编码后 INT12 有效 | 行 = token 时间步，列 = 特征维 K=64 |
| `W` | `[64,256]` | FP32 / 编码后 INT10 有效 | 行 = K=64，列 = 输出通道 N=256（**不转置存储**） |
| `bias` | `[256]` | FP32 / 编码后 INT32 | 每个输出通道一个偏置 |
| `Y` | `[99,256]` | INT32 | 行主序，行 = token，列 = 输出通道 |

- 权重按 `[K,N]` 存放，点积是 `sum_k X[m,k]*W[k,n]`；**不要**假定 `W` 以 `[N,K]` 存储。
- 所有整数一律**二补码有符号**。十六进制导出时每元素一个 token：
  16 位操作数 4 位十六进制、32 位量 8 位十六进制，小端 16 位 lane。
- 每个 16 位 lane 内部按小端序解释；32 位输出按小端序写回（低 16 位先、高 16 位后）。

## 2. 量化（编码）

设 `qmaxA = 2^(12-1)-1 = 2047`，`qmaxW = 2^(10-1)-1 = 511`。

- 激活（逐层一个静态尺度，取训练集该层最大绝对值）：
  `sx = max|x_train| / qmaxA`（下限 `1e-12`），
  `qx = clip(rint(x / sx), -qmaxA, +qmaxA)`。
- 权重（逐输出通道 `n`，取冻结 FP32 权重该列最大绝对值）：
  `sw[n] = max_k |W[k,n]| / qmaxW`（下限 `1e-12`），
  `qw[k,n] = clip(rint(W[k,n] / sw[n]), -qmaxW, +qmaxW)`。
- 偏置（量化到同一整数域）：`qb[n] = rint(b[n] / (sx * sw[n]))`，必须落在有符号 INT32 内，否则报错。
- **零点恒为 0**（对称量化），无偏移项。
- 舍入统一为 **round-to-nearest, ties-to-even**（NumPy `np.rint` / C `rint`）。
  禁止截断、禁止 ties-away、禁止先把尺度并进梯度。
- 超范围先乘后饱和：编码时显式 `clip` 到 `[-qmax,+qmax]`。

## 3. 整数乘累加与 INT32 溢出环绕

对每个输出 `Y[m,n]`：

```
acc = Σ_{k=0..63} qx[m,k] * qw[k,n] + qb[n]
Y[m,n] = wrap_int32(acc)
```

- 乘积与累加在整数域完成，K=64 项。硬件累加器为**有符号 INT32**。
- 若精确值超出 `[-2^31, 2^31-1]`，按**二补码环绕**取低 32 位：
  `wrap_int32(v) = ((v + 2^31) mod 2^32) - 2^31`。参考模型显式执行该环绕，
  并在 manifest 记录实际环绕次数。
- 保守上界（与样本无关）：`64 * qmaxA * qmaxW + max|qb|`，A12W10 下每层约
  `6.7e7`，占 INT32 的约 3%。本冻结参数包 12 层均不触发环绕
  （软件 `emit` manifest 的 `int32_wrap_events` 为 0）。
- 偏置在整数域与累加同时加入，**不**在反量化后加。RTL 也必须在 INT32 累加器里加偏置。

## 4. 反量化与输出字节序

- 评测用解码（仅在主机/评测软件，RTL 内不含该乘法）：
  `y[m,n] = Y[m,n] * (sx * sw[n])`，在 FP64 中计算。
- RTL 端口输出为 INT32，小端字节序：地址递增时先写 `Y[m,n]` 的低 16 位，再写高 16 位
  （与既有 `kwt_dma` 每 beat 拆两个 16 位字的约定一致）。
- 输出只含 **99 个有效行**，无填充行。行主序展平长度 `99*256 = 25344`。

## 5. 存储布局与填充规则

硬件把 `X`/`W` 按 tile 放入共享 SRAM。填充只影响存储与搬运，不影响有效输出。

- **A（激活）**：行 tile `13`（`ceil(99/8)`）、每 tile 8 行 lane、K=64。
  文件展平顺序 `rt,k,r`（`index=(rt*64+k)*8+r`），元素是量化的 `qx`。
  有效行 `row = rt*8+r < 99`；**最后一个 row tile 的 r=3..7 五行必须补 0**。
  文件长度 `13*64*8 = 6656` 个 16 位元素。
- **W（权重）**：列 tile `16`、每 tile 16 个通道 lane、K=64。
  展平顺序 `ct,k,c`（`index=(ct*64+k)*16+c`），列 `n = ct*16+c`。
  一个 128 位物理字 = 8 个连续 `c` lane。列数 256 恰好整除，无列填充。
  文件长度 `16*64*16 = 16384` 个 16 位元素。
- **bias**：256 个 INT32，按输出通道展平。
- **expected/输出**：`99x256` 行主序。**不用 tile 填充参与比较**，只比较 99 个有效行。
- `a16`/`w16` 是 16 位容器；A12W10 的有效位只占其中 12/10 位，容器高位按二补码符号扩展。

## 6. 误差口径（NRMSE）

对某层全部评测输出展平后：

```
err_rms = sqrt(mean((decode(Y) - FP32_ref)^2))
ref_rms = max(sqrt(mean(FP32_ref^2)), 1e-8)   # 题面分母下限 1e-8
NRMSE   = 100 * err_rms / ref_rms             # 单位 %
```

- 分母为 FP32 浮点参考结果的均方根，**下限 1e-8**。
- 12 层各自独立计算，取 12 层最大值作为该样本集合的误差；门槛 3%。
- 官方评测脚本的展平/汇总口径可能与本地不同，最终以官方脚本为准；本地公式与
  代码在 `ffn1_reference.py:nrmse_pct` 中保留、可复算。

## 7. 参考入口与验证证据

| 命令 | 作用 |
| --- | --- |
| `python ffn1_reference.py config` | 打印冻结配置与三份 SHA-256 |
| `python ffn1_reference.py emit --sample-count 2` | 两条不同类验证样本、12 层：导出逐层 INT32 `expected32`、尺度、manifest（含每层 expected32 SHA）、NRMSE |
| `python ffn1_reference.py verify-vectors --vectors <DIR>` | 用与 DUT 无关的 NumPy INT64 GEMM + INT32 环绕，对既有 E203 全层 golden 向量逐位交叉（12 层） |
| `python ffn1_reference.py selftest --vectors <DIR>` | 负向注入：错误权重、错误偏置、错误舍入必须被检出 |
| `python test_submission_model.py` | 顺序执行以上检查，写出 `results/selftest_summary.json`，退出码 0/1 |

`expected32`、manifest、中间大文件写到 gitignore 的 `software/.build/`，不入库。

## 8. 限制（不得夸大）

- 本协议只在本地训练/验证划分与既有 RTL 向量上验证，**不是官方隐藏 600 条评测**。
- 官方 FP32 模型由本地指定 checkpoint 复现（见 `software/weights/kwt1_fp32.npz`）；
  若评测环境拿不到官方模型/数据资源，只能用配置记录的哈希核对，**不得伪造 NRMSE**。
- RTL 与软件逐位一致的证据来自既有 E203 全层 golden 向量（40 样本 A12W10 参数包，
  参数 SHA `10c481c7...`）；它与本配置的参数包 SHA 不同，但同一协议、同一冻结 FP32
  权重：`verify-vectors` 对 12 层逐位通过，且 12 层 `w16` 与本配置 `qw` 完全相同。
- 本仓库的 E203 路径是 NICE-like 仿真系统，不是组委会正式 SoC/PDK；协议冻结不改变该事实。
