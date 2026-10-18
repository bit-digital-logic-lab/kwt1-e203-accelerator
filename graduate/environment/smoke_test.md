# 工具链 smoke test（真实退出码）

- 生成时间（UTC）：2026-10-18T07:47:02.597433+00:00
- 工作目录：`/tmp/opencode/kwt_env_smoke.m973s4bz`
- 汇总：pass=7 fail=0 skipped_missing_tool=0 skipped_missing_input=0

| 用例 | 工具 | 状态 | 退出码 | 证据 |
| --- | --- | :-: | --- | --- |
| Verilator 最小计数器编译并运行 | verilator | **pass** | [0, 0] | - S i m u l a t i o n   R e p o r t: Verilator 5.030 2024-10-27 / - Verilator: $finish at 115ps; walltime 0.000 s; speed |
| Verilator 对仓库内 SRAM RTL 做 lint | verilator | **pass** | [0] |       /                      ^~~~~ /                        ... For warning description see https://verilator.org/warn/U |
| Icarus Verilog 最小计数器编译并运行 | iverilog | **pass** | [0, 0] | SMOKE_Q=10 |
| Yosys 最小计数器综合并统计 | yosys | **pass** | [0] |        24 cells /        24 cells |
| Yosys(oss-cad-suite) 读入仓库内 SRAM SystemVerilog RTL | yosys | **pass** | [0] | End of script. Logfile hash: e452c8c03a, CPU: user 0.00s system 0.00s, MEM: 13.27 MB peak / Yosys 0.63+97 (git sha1 1111 |
| iEDA 可执行（STA/综合流程） | iEDA | **pass** | [0] | exit=0 / stdout/stderr='' |
| RV32 交叉编译最小 C 并核对 ELF 架构 | rv32_gcc | **pass** | [0, 0] | tiny.o:     file format elf32-littleriscv / architecture: riscv:rv32, flags 0x00000010: |

