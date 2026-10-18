# NICE-like 协议边界验证（`nice_protocol`）

## 对象与范围

被测对象是本工作树中的**临时集成模型** `kwt_nice_system_model`（含
`kwt_tile_system_model`、`kwt_dma`、`kwt_ext_memory_model` 与脉冲阵列）。
它的端口与片上替身 `hardware/rtl/control/tb/e203_kwt_nice_core.sv` 一致，但
**不是组委会 SoC、不是官方 E203 NICE 适配器、也不是综合后网表**。

因此本套件证明的是：

- 该模型在其真实端口上满足 valid/ready 握手、响应保持、请求/总线反压、
  复位、非法与越界描述符、DMA error/done 的**行为级**性质；
- 不证明官方 E203 NICE 协议、真实 System Bus、时序收敛或竞赛平台通过。

## 覆盖项（24 个场景 / 25 条命令）

| 场景 | 输入 | 期望 |
| --- | --- | --- |
| S01 上电复位释放 | rst_n 0→1 | `nice_rsp_valid=0`，`nice_req_ready=1` |
| S02 正常事务 + 响应反压 | 合法描述符，`rsp_ready` 压 5 拍 | status=0，128 个输出逐项核对 |
| S03 命令错误 + 反压 | k_len=0，压 4 拍 | status=1，`nice_rsp_err=1` |
| S04 DMA 错误 + 反压 | out 越界，压 3 拍 | status=2，`nice_rsp_err=1` |
| S05 请求跨响应保持 | 在 RESPONSE 期间拉高下一次 `req_valid` | RESPONSE 期间 `req_ready=0`；握手后请求不丢失 |
| S06 总线反压全程 | 每 5 拍停 2 拍 `bus_ready` | 事务仍 status=0，输出逐项核对 |
| S07 描述符地址上边界 | rs1=EXT_MEM_BYTES-32 | 接受，status=0 |
| S08–S11 指令/描述符地址非法 | 错 opcode / 错 funct3 / 非零 funct7 / 未对齐 rs1 / rs1+32 越界 | status=3 |
| S12–S16 命令字段非法 | k_len=0、k_len=65、row_mask=0、col_mask=0、a/b 地址未对齐 | status=1 |
| S17–S20 DMA 越界 | out / A / B / bias 目标越界 | status=2 |
| S21 事务中复位 | 计算中拉低 rst_n 3 拍后释放 | 复位期间无响应；释放后可正常完成 |
| S22 连续任务 | 连续 3 条合法命令 | 均 status=0，输出正确 |

断言总数 865，总线节拍计数 450。同一 TB 在 Icarus 14 与 Verilator 5.030 下
结果一致（scenarios=24, commands=25, assertions=865, bus_beats=450）。

## 判据

- **响应保持**：`nice_rsp_valid && !nice_rsp_ready` 时，下一拍必须仍 valid
  且 `data`/`err` 不变（`RESPONSE_HOLD_VIOLATION`）。
- **总线稳定**：`bus_valid && !bus_ready` 被停拍时，下一拍必须仍 valid 且
  `addr/wdata/wstrb/write` 不变（`BUS_STABILITY_VIOLATION`）。
- **状态码**：合法=0；命令/字段非法=1；DMA 越界=2；指令或描述符地址非法=3；
  `nice_rsp_err = (status != 0)`；所有输出与整数点积+偏置+Relu 的独立期望核对。

## 故障注入负例

`run_nice_protocol.sh` 用 `iverilog -D` 打开：

1. `NICE_FAULT_RSP`：用 `force` 把 DUT 的 `nice_rsp_data` 固定错值，检查器
   必须在第一次比较时报 `RSP_VALUE_MISMATCH`（子运行 exit=1）。
2. `NICE_FAULT_BUS`：在被停拍的总线节拍上注入地址抖动，检查器必须报
   `BUS_STABILITY_VIOLATION`（子运行 exit=1）。

脚本只有在正向 exit=0 且两个负例都 exit≠0、命中对应断言文本时才 `SUITE_RESULT PASS`。

## 证据层级与未证事项

- 全部为 **行为级 / 双仿真器**结果：外部存储、总线、SRAM 与脉冲阵列为简化
  模型；不含时序、面积、功耗结论。
- `assertions`/`bus_beats` 只是 harness 观察计数，不是覆盖闭包证明。
- 未覆盖：官方 E203 NICE 时序握手、真实 ICB/System Bus、dwt/nice 中断与
  长描述符链、k_len=64 全深度大张量（会与其它数据区重叠，需重排存储布局）。
- 未发现真实缺陷；若后续把 k_len=64 全量搬入，应重排 A/B/bias/out 地址后再测。
