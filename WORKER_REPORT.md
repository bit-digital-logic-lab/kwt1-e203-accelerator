# WORKER_REPORT — NICE-like 协议边界验证

分支：`codex/kwt-nice-protocol`（本工作树）。范围：只在
`hardware/verification/nice_protocol/` 新增验证文件，不改核心 RTL、驱动、
16-row E203 集成、主目录或其他分支。未用子智能体，未 push/merge/rebase。

> 本文件此前记录的是 tile 级 edge-case 验证，内容仍保留在 Git 历史
> （`git show c12a0ea:WORKER_REPORT.md`）。此处改写为本轮任务报告。

## 改动

新增（均在 `hardware/verification/nice_protocol/`）：

- `tb_nice_protocol_boundaries.sv`：对 `kwt_nice_system_model`（含 tile 控制、
  DMA、128-bit 外部存储模型、脉冲阵列）的自检 TB。24 个场景、25 条命令、
  865 条断言，覆盖响应保持、请求/总线反压、复位、非法与越界描述符、
  DMA error/done。
- `run_nice_protocol.sh`：Icarus 入口。正向必须 exit 0 且命中 PASS 横幅；
  两个 `-D` 故障注入负例必须 exit≠0 且命中指定断言文本，否则 `SUITE_RESULT FAIL`。
- `run_nice_verilator.sh`：用 Verilator 复跑正向，交叉确认同一行为。
- `NICE_PROTOCOL_TESTS.md`：覆盖项、判据、证据层级、未证事项。

## 依据真实端口确定的覆盖项

按 `kwt_nice_system_model.sv` / `kwt_tile_system_model.sv` 的真实实现确定，
不假设官方 E203 协议：

- **合法路径**：合法描述符 → 两拍取指 → tile 命令 → 8×16 输出，逐项核对。
- **响应保持**：`rsp_valid && !rsp_ready` 时 data/err 必须保持不变。
- **请求反压**：RESPONSE 期间 `req_ready=0`；跨响应保持的请求握手后不丢失。
- **总线反压**：`bus_valid && !bus_ready` 时 addr/wdata/wstrb/write 保持稳定。
- **复位**：上电复位后状态；事务计算中复位 3 拍再释放并恢复。
- **非法/越界描述符**：错 opcode/funct3/funct7、未对齐或越界 rs1（status 3）；
  k_len=0/65、mask=0、地址未对齐（status 1）。
- **DMA error/done**：out/A/B/bias 目标越界 → status 2，`rsp_err=1`。

## 实际执行的命令与退出结果

| 命令 | 退出码 | 结果 |
| --- | ---: | --- |
| `bash hardware/verification/nice_protocol/run_nice_protocol.sh` | 0 | `SUITE_RESULT PASS`；正向 `scenarios=24 commands=25 assertions=865 bus_beats=450`；两个负例均被检出 |
| `bash hardware/verification/nice_protocol/run_nice_verilator.sh` | 0 | Verilator 5.030 下同一结果 `scenarios=24 commands=25 assertions=865 bus_beats=450` |

关键输出：

```
PASS nice_protocol_boundaries: scenarios=24 commands=25 assertions=865 bus_beats=450
positive        0  zero     PASS nice_protocol_boundaries                1 1
negative_rsp_fault 1 nonzero RSP_VALUE_MISMATCH                          1 1
negative_bus_fault 1 nonzero BUS_STABILITY_VIOLATION                     1 1
SUITE_RESULT PASS
```

故障注入负例的断言文本：

```
FATAL: ... RSP_VALUE_MISMATCH data=deadbeef err=0 expected_status=0   (exit=1, -DNICE_FAULT_RSP)
FATAL: ... BUS_STABILITY_VIOLATION valid=1 addr=00000510(want 00000500) write=0 (exit=1, -DNICE_FAULT_BUS)
```

## 真实缺陷

在已覆盖范围内**未发现真实缺陷**：所有合法场景 status=0 且输出逐项正确，
所有非法/越界场景返回预期的 1/2/3 且 `rsp_err` 正确，复位与反压均符合预期。
未牵强断言官方 E203 NICE 协议已验证。

## 证据层级与未证事项

- 全部为**行为级 / 双仿真器**结果（Icarus 14 与 Verilator 5.030）；外部存储、
  总线、SRAM、脉冲阵列为简化模型，不含时序、面积、功耗结论。
- `assertions`/`bus_beats` 仅为 harness 观察计数，不是覆盖闭包证明。
- 未覆盖：官方 E203 NICE 时序握手、真实 ICB/System Bus、中断/长描述符链、
  k_len=64 全深度搬运（会与现有数据区重叠，需重排地址后再测）。
