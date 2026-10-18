# PC → RCL 数据传输 ABI（FFN1 第一线性层）

本目录把 **PC 端预处理产物**（离线 MFCC `98×40` 与首层编码器输入 `99×64`）
到**片上加速器消费端**的字节级契约冻结下来，并做逐字节核对。
PC 端只做前处理；片上只执行本科必做的第一线性层
`Y = XW + b`：`99×64 · 64×256 + 256 → 99×256`。

> **术语边界。** 仓库中不存在名为 “RCL” 的模块或文档（`grep -ri '\bRCL\b'` 无命中）。
> 因此本 ABI 不假设任何 “RCL 寄存器/pin 映射”，只冻结仓库里**真实存在**的两段契约：
> (1) 外存/共享 SRAM 的字节流布局；(2) RTL/驱动实际消费的 32 字节 tile 描述符。
> 组委会正式 SoC 的 System Bus 地址图与 NICE 连接在本仓库中缺失
> （`software/driver/README.md`、`kwt_nice_system_model.sv` 均有明示），不在此臆造。

## 1. 证据来源（实际接口，不是设计意图）

| 事实 | 文件:行 |
| --- | --- |
| 外存 512 KiB、总线 128 bit、SRAM 16 bit×1024、三 bank | `hardware/rtl/sram/kwt_memory_defs.svh` |
| 一拍 128-bit 拆成 8 个连续 16-bit 字，低地址=低位字节（小端） | `hardware/rtl/dma/kwt_dma.sv:141-153`（`bus_rdata[lane*16 +: 16]` 顺序写入连续 SRAM 字） |
| 只有 bank 2 可写回；外部地址须 16 字节对齐；越界报错 | `hardware/rtl/dma/kwt_dma.sv:118-127` |
| A 每拍 8 字（`k_len*8`），B 每拍 16 字（`k_len*16`），bias 32 字，out 256 字 | `hardware/rtl/control/kwt_tile_system_model.sv:70-72` |
| A 寻址 `k_index*ROWS+lane`、B 寻址 `k_index*COLS+lane` | `hardware/rtl/pulse_array/kwt_pingpong_cacc_engine.sv:86-87` |
| 输出 INT32 拆成低/高两个 16-bit 字写回，行主序 `r*16+c` | 同文件 `:125-146`、`dump_row/col` 定义 |
| 偏置为 32 位回绕加法、ReLU 可选、**无饱和** | `hardware/rtl/pulse_array/kwt_postprocess.sv`、引擎 `bias_sum` |
| 2 拍描述符字段：beat0=A/B/bias/out 地址；beat1=K/ReLU、行/列掩码 | `hardware/rtl/control/kwt_nice_system_model.sv:106-121` |
| 真实向量 `a16` 展平 `rt,k,r`；`w16` 展平 `ct,k,c`，128-bit 字=8 个 c lane | `hardware/rtl/control/tb/prepare_real_e203_vectors.py:95-107,199-205` |
| 描述符结构、tile 步长（A 1024B/W 2048B/bias 64B/out 512B）、16B 对齐与 512 KiB 检查 | `software/driver/kwt_nice_driver.{h,c}` |

## 2. V1 冻结格式（当前 RTL/驱动实际使用的格式）

- **字节序**：小端。INT16 元素低字节在低地址；一拍 8 个 lane 从低地址到高地址。
- **元素类型**：A、W 为有符号 INT16 容器（A 实际用 12 位、W 实际用 10 位，见 §4）；
  bias、Y 为有符号 INT32。
- **形状 / stride / 对齐**（`A_TILE_BYTES=1024`、`W_TILE_BYTES=2048`、
  `BIAS_TILE_BYTES=64`、`OUT_TILE_BYTES=512`，全部 16 字节对齐）：

| 张量 | 逻辑形状 | 平铺 | 单片字节 | 总字节 | tile 基址 stride |
| --- | --- | ---: | ---: | ---: | --- |
| A | 99×64 | `(rt,k,r)`，rt=13、k=64、r=8 | 1024 | 13312 | `rt*1024` |
| W | 64×256 | `(ct,k,c)`，ct=16、k=64、c=16 | 2048 | 32768 | `ct*2048` |
| bias | 256 | 展平 `ct,c`（INT32） | 64 | 1024 | `ct*64` |
| Y | 99×256 | `(tile,r,c)`，tile=208、r=8、c=16（INT32） | 512 | 106496 | `tile*512` |

- 末行块 rt=12 只有 3 行有效，r∈[3,7] 必须为 0；驱动对应 `row_mask=0x07`。
- 描述符 32 字节、16 字节对齐：字节 0/4/8/12 = A/B/bias/out 地址（小端 u32）；
  字节 16 = `K | ReLU<<7`（K=64 → 0x40）；字节 20 = 行掩码；字节 21-22 = 列掩码。
- 地址边界：每个 tile 起始和总长都必须 16 字节对齐且不越过 512 KiB；A/B/bias/out 五块互不重叠。

## 3. 逐字节核对结果（真实样本，可复现）

命令与退出码：

```bash
python3 software/transport/run_transport_check.py     # EXIT=0，29/29 PASS
```

数据：真实样本 `software/validation/real_input_chain_20261017/sample_00_encoder.npy`
（`99×64` FP32，sha256 `489dc7…c058`），冻结参数 `software/results/ffn1_a12w10_static_sym.npz`
（A12W10，sha256 `a77f60…918e`，与同名 `.json` 的 `parameter_sha256` 一致）。
结果 JSON：`software/transport/results/transport_abi_v1_layer01.json`。

核对结论：

- pack→unpack→repack 对 A/W/bias/Y **逐字节相等**（max|diff|=0）。
- 用**独立 INT64 GEMM**（直接由源数组算，不经打包字节）与逐 tile INT32 累加结果
  **逐位相等**（max|diff|=0，本层 99×256 = 25,344 个输出）。
- 端序验证：`A[0,0]=-195` 编码为字节 `3d ff`（小端）。
- 描述符 208 个往返一致，首个 `control=0x40 / rowmask=0xff / colmask=0xffff`，
  末行块 `rowmask=0x07`。
- 地址：out 区末地址 `0x2a000 < 0x80000`；越界与未对齐（`0xa3001`）均被拒绝。
- FP64 解码 vs 冻结 FP32 权重参考：**NRMSE=0.119933%**（< 3% 门槛）；本样本激活零截断。
- **无饱和**：`INT32_MAX+1` 回绕为 `-2147483648`，容器逐字节保留该回绕值。安全性依赖
  导出期保守上界检查（本层 `67389605 < 2^31`，与 `.json` 一致），不是 RTL 饱和。
- 负例：A 区单字节翻转 → SHA 变化且解出元素改变；A 区截断 1 字节 → `ValueError`；
  V2 载荷末字节翻转 → CRC 拒绝。

blob 哈希（写至 `/tmp/opencode/transport_abi/`，不入库）：
A `b24dca…b538`、W `2ad48c…faa4`、BIAS `a4c720…3497`、Y `3a443a…3154`、V2 `9d324a…859b`。

## 4. 与现状不符 / 现状不支持的字段（只报证据，不改现有预处理与驱动）

1. **W 不是 10 位物理存储。** 冻结 `.npz` 用 `int16` 容器装 W10（`layer01_weight` dtype
   `int16`，每元素 2 字节，`W_BYTES=32768`）；`ffn1_a12w10_static_sym.md` 明说 240 KiB 只是
   紧密 10 位打包的理论容量，“物理打包/解包逻辑还未实现”。两者相差 1.6×。
2. **传输流不自描述。** V1 字节流没有 magic/版本/dtype/尺度/长度/校验。`sx`、`sw`、
   `qb` 的量纲只存在于旁路 `.npz`/`.json`；接收端若无旁路文件无法反量化。
   `.json` 的 `parameter_archive` 还指向 `…/Desktop/…` 的陈旧绝对路径。
3. **无饱和语义。** 见 §3；ABI 层不应假定有 clamp。
4. **SoC 寄存器/pin 级 ABI 未定义。** 只有描述符字段和外存字节流可冻结。

## 5. 候选格式 V2（显式版本化，仅提案，与 V1 隔离）

`transport_abi.StreamV2` 在同一 V1 载荷平铺外用 64 字节头 + 每节 48 字节目录包裹：
`magic='KWT2'`、`version=2`、标志位（小端/含尺度）、层号、节数、总长、**载荷 CRC32**；
每节记录 `name/dtype/elem_count/offset/length/CRC32`，并新增 `SCALEX`、`SCALEW` 浮点尺度节
与可选的**紧密 10 位 `W10` 节**（本章 W10 打包 20480 字节 vs INT16 32768 字节，往返逐位一致）。
V2 是 PC 侧容器提案，**RTL 尚未实现**，不声称是 SoC 寄存器 ABI。

## 6. 未证事项

- X 用的是仓库已提交的真实 **首层编码器输入** `99×64`；真正的“注意力+残差+第一处 LayerNorm
  之后、FFN1 之前”激活需要完整 FP32 前向，本工作树无 torch，未重算。故本层验证的是
  **搬运/布局/整数数值契约**，不是 FFN1 输入语义本身。
- 只覆盖 1 层（layer01）1 条真实样本；未跑 12 层 × 多样本。
- 未接组委会正式 SoC/System Bus/NICE，未综合、未做时序/面积/功耗。
- 未把 V1 字节流写入真实 RTL 仿真（由 `hardware/rtl/control/tb/` 既有向量回归覆盖，
  本目录不重复跑 HDL）。
