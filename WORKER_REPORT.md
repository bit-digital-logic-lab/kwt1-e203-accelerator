# WORKER_REPORT — r2_pack_format（12 层参数与 99×64 激活的字节打包/解包与 roundtrip）

- 分支 / worktree：`codex/kwt-ds-grad-r2_pack_format` @ `/home/yang/kwt-worktrees/ds-grad-r2_pack_format`
- 范围：**只写** `graduate/ds_tasks/r2_pack_format/**` 与根 `WORKER_REPORT.md`；未开子智能体；
  未改其他 worktree；未 `merge`/`rebase`/`push`；未改 `graduate/encoder`、`software/`、`hardware/` 既有文件（只读）。
- 只读使用主工程 `/home/yang/Desktop/ic设计竞赛` 的 gitignored 数据；未提交大数据/密钥/缓存；
  构建目录固定在 `/tmp/opencode/r2_pack_format/`。

## 1. 结果

**KWT1-PACK v1 已定义并可运行，roundtrip 逐位无损。**

- **格式**：小端（LE，与 E203/RISC-V 一致）；blob 头 64 B + 48 B/段目录；payload 与每段
  16 B 对齐；int16@2 B、int32/fp32@4 B；值行主序；PACK10/PACK12 为 LSB-first 有符号补码位流。
- **打包与 roundtrip（L1 软件参考，`run_pack.py` → `results.json`）**：

  | 对象 | 段数 | 字节 | 失配 |
  | --- | ---: | ---: | ---: |
  | 12 层 FFN1（W1 int16 + b1 int32）容器 | 24 | 406,720 | **0** |
  | 同上 PACK10 / PACK12 | 24 | 259,264 / 308,416 | **0 / 0** |
  | 6 条真实 99×64 激活 int16 容器 | 6 | 76,384 | **0** |
  | 同上 PACK12 / PACK10 | 6 | 57,376 / 47,872 | **0 / 6（不兼容）** |
  | 完整 12 层 FP32 参数容器 | 162 | 2,437,072 | **0** |

  10-bit 位流对 12-bit 激活不适用（`|q|` 达 2048），作为不兼容对照记录。
- **量化-打包联合校验（划分规则）**：激活尺度 `s_x = max_{training}|X|/2047 = 1.961475e-3`
  仅由 training 划分（`sample_01/03/04`）无梯度求得；validation 划分（`sample_00/02/05`）
  **仅评估**，最差 `0.1400%`（打包无损，误差纯来自 12-bit 量化）。
- **带宽/端序/对齐**已列表（见 `README.md` §三）：单层 staged 容器 33,792 B → PACK10 21,504 B；
  12 层参数 405,504 / 258,048 / 307,200 B；X 12,672 B、Y 50,688 B；均给出 @16/32/128-bit 口径。
- **负对照 9 项全检出**：结构类（截断、magic、错位、dtype 冲突、目录越界）与取值类
  （大端解码、位流翻转、位宽混淆、位流截断），证明 checker 非空转。
- **L2 RTL 逐位（字节/位序）**：Verilator `tb_byte_lane` LE unpack16 `4096/4096`、
  unpack32 `2048/2048`、pack16 `8192/8192`、PACK12 `6336/6336`，err=0。
- **L4 综合（部分）**：泛型 yosys 0.63 对三模块 `synth` 0 problem（仅证可综合，非 PDK/时序）。

## 2. 复现命令与退出码（本任务，2026-10-19）

| 命令（worktree 根执行） | 退出码 | 结果 |
| --- | ---: | --- |
| `bash graduate/ds_tasks/r2_pack_format/run.sh` | **0** | `R2_PACK_FORMAT_SUITE_PASS` |
| ↳ `l1_run_pack`：`python3 graduate/ds_tasks/r2_pack_format/run_pack.py --build-dir /tmp/opencode/r2_pack_format` | **0** | `ALL_BIT_EXACT=True worst_val_quant_nrmse=0.1400%` |
| ↳ `l2_emit_vectors`：`python3 …/emit_rtl_vectors.py /tmp/opencode/r2_pack_format/rtl` | **0** | 生成 8,192 B/8,192 B/9,504 B 激励 |
| ↳ `l2_verilator_build`：`verilator --binary --timing … -Mdir /tmp/opencode/r2_pack_format/rtl/obj_dir` | **0** | 0 error（`-Wno-WIDTHTRUNC` 消 1 处条件位宽提示） |
| ↳ `l2_verilator_run`：`cd /tmp/opencode/r2_pack_format/rtl && ./obj_dir/Vtb_byte_lane` | **0** | `RTL_LANE_PASS`（4 项 got/exp 全等、err=0） |
| ↳ `l4_yosys_synth`：`yosys -p "read_verilog -sv …; synth -top <top>; stat"`（`kwt_le_unpack`/`kwt_le_pack`/`kwt_bit_unpack`） | **0** | 三模块 `Found and reported 0 problems` |

日志与退出码：`graduate/ds_tasks/r2_pack_format/logs/summary.tsv` 及同级 `l1_run_pack.log`、
`l2_emit_vectors.log`、`l2_verilator_build.log`、`l2_verilator_run.log`、`l4_yosys_synth.log`。
输入 SHA-256：`ffn1_a12w10_static_sym.npz = a77f60fb…918e`、`kwt1_fp32.npz = 8f359abf…25825a`。

## 3. 未证事项（不得当作完整研究生组验收）

1. **12 层整数参数仅覆盖 FFN1**：注意力的逐层整数包、FFN2/LN 的逐层整数包不在本任务；
   完整 12 层参数以 **FP32 容器**打包（往返无损，但非整数量化产物）。格式为其预留同构段。
2. **不是数值精度证据**：打包/解包逐位无损；激活量化只做 **6 条真实样本**（3 训练/3 验证）
   的量化-打包联合校验，未做 480/480 或 12 层整链 NRMSE；未跑官方 600 条评测脚本。
3. **无 E203 / SoC**：未在官方 E203 指令流、System Bus 或 NICE 上运行；只对齐 16 B 描述符地址口径，
   未实现 DMA/总线事务。
4. **综合非签核**：泛型 yosys `synth` 只证可综合；**无 PDK/SDC/SRAM 宏，未做 150 MHz STA、面积、功耗**。
5. **未证 bit-packing 的硬件收益**：PACK10/PACK12 的带宽节省是字节口径，未综合紧密解包器，
   未测其面积/时序开销；未对产品 RTL 做逐位整链校验。
6. 本报告不声明“12 层整层 E203 通过”或“研究生 5–6 项验收通过”。
