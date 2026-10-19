# r2_pack_format — KWT1-PACK v1 字节打包/解包与 roundtrip 校验

第二轮任务。为**12 层参数**与 **99×64 激活**冻结一套可运行的**字节级打包格式**，
给出打包/解包参考实现、逐位 roundtrip 校验、负对照、带宽/端序/对齐口径与证据层级。

- 分支 / worktree：`codex/kwt-ds-grad-r2_pack_format` @ `/home/yang/kwt-worktrees/ds-grad-r2_pack_format`
- 只写：`graduate/ds_tasks/r2_pack_format/**` 与根 `WORKER_REPORT.md`
- 未开子智能体；未改其他 worktree；未 `merge`/`rebase`/`push`；未改 `graduate/encoder`、`software/`、`hardware/` 任何既有文件（只读引用）。
- 大数据只读自主工程 gitignored 路径，未提交大数据/密钥/缓存。

## 结论（一句话）

**KWT1-PACK v1 已可运行且 roundtrip 逐位无损**：12 层 FFN1 冻结整数参数
（24 段，406,720 B）、6 条真实 99×64 激活（76,384 B）、完整 12 层 FP32 参数
（162 段，2,437,072 B）在 Python 参考实现中打包→解包**零失配**；9 项结构/取值负对照
全部被检出；RTL（Verilator）对 little-endian 16/32-bit 字装配与 12-bit PACK12 位流
重建**逐位一致**（`4096/4096`、`2048/2048`、`8192/8192`、`6336/6336`）。10-bit 位流对
激活**不适用**（激活需 12-bit），已作为不兼容对照记录。整套脚本退出码 0。

## 一、格式定义（KWT1-PACK v1）

| 项 | 规定 |
| --- | --- |
| 字节序（端序） | **小端 little-endian**，与 E203/RISC-V 一致；大端解码被负对照拒绝 |
| blob 头 | 64 B，含 magic `KWTP`、version、nseg、payload 偏移/长度 |
| 段目录 | 每段 48 B：name[16] + dtype/bits/rank + 3×u32 shape + u64 offset + u32 nbytes + flags |
| 对齐 | 头部 offset 0；目录 48 B；**payload 与每段起始 16 B 对齐**（与 `SUBMIT` 描述符 16 B 地址契约一致） |
| 元素对齐 | int16 @2 B、int32/fp32 @4 B、位流 @byte |
| 取值顺序 | 行主序（C order），shape 存于目录，解包后索引映射与原整数模型一致（如 FFN1 `W1[c*D+j]`） |
| 位流位序 | PACK10/PACK12：值 *i* 占位 [bits·i, bits·i+bits)；流位 *j* = 字节 *j//8* 的位 *j%8*（**LSB-first**）；负数用补码 |

dtype：

| dtype | 语义 | 每元素 |
| --- | --- | --- |
| `INT16` | 有符号 16-bit 补码，LE | 2 B |
| `INT32` | 有符号 32-bit 补码，LE | 4 B |
| `FP32` | IEEE-754 binary32，LE | 4 B |
| `PACK10` | 有符号 10-bit 补码，4 值/5 B | 1.25 B |
| `PACK12` | 有符号 12-bit 补码，2 值/3 B | 1.5 B |

匹配 A12W10 冻结配置：权重 10-bit（`|w|≤511`）、激活 12-bit（`|q|≤2047`，容器含 `-2048`）。

## 二、打包对象与 roundtrip 结果

| 对象 | 来源（只读） | 段数 | 字节 | roundtrip 失配 |
| --- | --- | ---: | ---: | ---: |
| 12 层 FFN1 参数（W1 int16 + b1 int32），int16 容器 | `software/results/ffn1_a12w10_static_sym.npz` | 24 | 406,720 | **0** |
| 同上，PACK10 权重 | 同上 | 24 | 259,264 | **0** |
| 同上，PACK12 权重 | 同上 | 24 | 308,416 | **0** |
| 6 条真实 99×64 激活，int16 容器 | `software/validation/real_input_chain_20261017/sample_0X_encoder.npy` | 6 | 76,384 | **0** |
| 同上，PACK12 | 同上 | 6 | 57,376 | **0** |
| 同上，PACK10 | 同上 | 6 | 47,872 | 6（**不兼容**：12-bit 激活放不进 10-bit 流） |
| 完整 12 层 FP32 参数（注意力/FFN/LN/投影/嵌入/分类） | `software/weights/kwt1_fp32.npz` | 162 | 2,437,072 | **0** |

- 输入 SHA-256：`ffn1_a12w10_static_sym.npz` = `a77f60fb…918e`；`kwt1_fp32.npz` = `8f359abf…25825a`。
- blob SHA-256：参数容器 `fc3d929a…fa1d`；激活容器 `e8bd93eb…137e`；FP32 包 `f7120f2e…e303`（见 `results.json`）。
- 12 层 FFN1 容器 = 载荷 405,504 B + 头/目录 1,216 B。
- 与设计文档交叉核对一致：W1 单精度容器 12 层 393,216 B、10-bit 紧凑 **240.0 KiB**（=245,760 B）、12-bit 紧凑 288 KiB（=294,912 B）。

### 激活的量化-打包联合校验（划分规则）

- 激活尺度 `s_x = max_{training}|X| / 2047 = 4.015140/2047 = 1.961475e-3`，
  **仅用 training 划分**（`sample_01/03/04`）无梯度求最大；**validation 划分（`sample_00/02/05`）仅用于评估**，
  未参与任何尺度/阈值/格式选择。
- 打包/解包本身逐位无损；数值误差只来自 12-bit 量化。验证划分最差 `0.1400%`（`sample_05`，
  `max|q|=2048`，即训练尺度下验证样本略超 `+2047` 触及 `-2048`，12-bit 容器正好容纳），
  远低于 3% 门槛。

## 三、带宽、端序、对齐

每层 FFN1 与激活（16-bit SRAM 字 / 32-bit NICE 字 / 128-bit 总线拍；取自 `results.json.bandwidth`）：

| 数据 | 字节 | @16-bit 字 | @32-bit 字 | @128-bit 拍 |
| --- | ---: | ---: | ---: | ---: |
| W1 int16 容器 | 32,768 | 16,384 | 8,192 | 2,048 |
| W1 PACK10 | 20,480 | 10,240 | 5,120 | 1,280 |
| W1 PACK12 | 24,576 | 12,288 | 6,144 | 1,536 |
| b1 int32 | 1,024 | 512 | 256 | 64 |
| 分层 staged（W1+b1）容器 / PACK10 | 33,792 / 21,504 | 16,896 / 10,752 | 8,448 / 5,376 | 2,112 / 1,344 |
| X 99×64 int16 / PACK12 | 12,672 / 9,504 | 6,336 / 4,752 | 3,168 / 2,376 | 792 / 594 |
| Y 99×256 int16 | 50,688 | 25,344 | 12,672 | 3,168 |

12 层合计：

| 数据 | 字节 | @128-bit 拍 |
| --- | ---: | ---: |
| 参数容器（int16 W1 + int32 b1） | 405,504 | 25,344 |
| 参数 PACK10 | 258,048 | 16,128 |
| 参数 PACK12 | 307,200 | 19,200 |
| 激活 12×99×64 int16 | 152,064 | 9,504 |

- **端序**：所有多字节量小端；RTL 逐位证明 16/32-bit 小端装配正确。
- **对齐**：payload 与每段 16 B 对齐；int32/fp32 需 4 B；int16 需 2 B；位流 byte 对齐。
  PACK10/PACK12 在 64×256、99×64 上均整除（`16384×10=163,840 bit`、`6336×12=76,032 bit`），无需尾对齐。
- 打包开销：头 64 B + 目录 48 B/段（24 段 = 1,152 B）。

## 四、负对照（checker 必须检出）

| 控制 | 类型 | 结果 |
| --- | --- | --- |
| `truncated_blob` | 结构 | 检出（段超界，ValueError） |
| `bad_magic` | 结构 | 检出 |
| `misaligned_offset` | 结构 | 检出（offset 非 16 B 对齐） |
| `dtype_bits_conflict` | 结构 | 检出（elem_bits 与 dtype 不一致） |
| `dirent_overlap` | 结构 | 检出（目录越入载荷） |
| `big_endian_decode_mismatch` | 取值 | 检出（大端解码与参考不符） |
| `pack10_bit_flip_mismatch` | 取值 | 检出（位流翻转 1 bit 即失配） |
| `dtype_width_confusion_mismatch` | 取值 | 检出（PACK12 流按 PACK10 解码失配） |
| `packed_stream_truncation_detected` | 取值 | 检出（位流截断） |
| 激活 PACK10 | 不兼容 | 6/6 样本失配（`|q|>511`，证明 10-bit 不够） |

## 五、证据层级

| 层级 | 状态 | 依据 |
| --- | --- | --- |
| **L1 软件参考** | **已证** | `run_pack.py`：三类对象 roundtrip 零失配 + 9 项负对照全检出；`logs/l1_run_pack.log`，退出码 0 |
| **L2 RTL 逐位** | **已证（字节/位序）** | Verilator `tb_byte_lane`：LE unpack16 `4096/4096`、unpack32 `2048/2048`、pack16 `8192/8192`、PACK12 `6336/6336`，err=0；`logs/l2_verilator_run.log`，退出码 0 |
| **L3 E203 / SoC** | **未证** | 无官方 E203 指令流 / System Bus / NICE；只有格式与 E203-like 事务口径对齐 |
| **L4 综合** | **部分** | 泛型 yosys 0.63 对三模块 `synth` 通过（0 problem），仅证明**可综合**；无 PDK/SDC/SRAM 宏，**未做 150 MHz STA/面积/功耗** |
| **L5 官方验收** | **未证** | 未跑官方 600 条评测；本格式是传输/存储口径，不改变数值门槛 |

## 六、文件

| 文件 | 作用 |
| --- | --- |
| `pack_format.py` | KWT1-PACK v1 参考 pack/unpack/validate/编解码（含 LE、位流、结构校验） |
| `run_pack.py` | L1：三类对象打包+roundtrip+量化-打包联合校验+9 项负对照+带宽表→`results.json` |
| `emit_rtl_vectors.py` | 把参数包/激活位流再序列化为 Verilator 激励（不选尺度） |
| `rtl/kwt_pack_lane.sv` | 可综合字节-字小端装配/拆解 + PACK12 LSB-first 位流解包 |
| `tb/tb_byte_lane.sv` | RTL 逐位对照 TB（8,192 B→4,096 字 / 8,192 B→2,048 字 / 位流→6,336 值 / 回环 8,192 B） |
| `rtl/`、`tb/` | 只写本任务目录，独立于产品 RTL |
| `run.sh` | 一键复现（L1+L2+L4*），写 `logs/summary.tsv`（含退出码） |
| `results.json` | 机器可读汇总（哈希、段表、带宽、负对照、证据层级） |
| `logs/` | 实跑日志与 `summary.tsv` |

## 七、复现

```bash
bash graduate/ds_tasks/r2_pack_format/run.sh
# exit 0 == R2_PACK_FORMAT_SUITE_PASS（L1/L2/L4* 全 0）
```

构建目录固定在 `/tmp/opencode/r2_pack_format/`，不写其他 worktree。

## 八、范围与未证事项

1. **数值冻结范围**：12 层**整数**参数仅有 FFN1（W1/b1）的冻结包；注意力的逐层整数包、
   FFN2/LN 的逐层整数包不在本任务，格式为其预留同构段。完整 12 层参数另以 FP32 容器打包
   （往返无损，但那是 FP32 原始字节，不是所有段的整数量化产物）。
2. **不是数值精度证据**：打包/解包逐位无损；激活量化误差另计，只做 6 条真实样本的
   训练标定/验证评估，未做 480/480 大样本或 12 层整链误差。
3. **无 SoC/总线**：未在 E203 System Bus/NICE 上跑；未生成 DMA/描述符事务，只对齐 16 B 地址口径。
4. **综合口径**：泛型 yosys `synth`，不是目标 PDK 综合，**无 150 MHz 时序、面积、功耗结论**。
5. **未做官方 600 条评测**；未声明研究生 5–6 项验收通过。
