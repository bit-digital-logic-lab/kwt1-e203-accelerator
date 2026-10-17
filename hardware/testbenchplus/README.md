# testbenchplus

只在本目录新增仿真基础设施；现有 RTL 和测试文件不改动。

运行 `bash testbenchplus/run_all_tests.sh`：先跑新向量，再统一运行现有阵列、单块、DMA、控制和 NICE 模型脚本。`--vectors-only` 只跑新向量；`--full` 还运行真实 E203 的单命令与 208 块整层测试，需上级 `资料/e203_hbirdv2_full` 及 RISC-V 工具链。可向新向量部分指定一个/多个 `case.json`。`--out /tmp/kwt-test-results` 保留结果、操作数 hex 和性能计数；`--dump --out /tmp/kwt-test-results` 另生成 VCD 内部状态波形。基础测试需要 `python3`、`iverilog` 和 `vvp`。

`vectors/<名称>/case.json` 包含 `format`（`int8`、`int16`、`int32`、`fp32`）、矩阵 `a[M][K]`、`b[K][N]` 和**独立提供**的期望矩阵 `expected[M][N]`。物理阵列固定 8×16，支持 `M≤8`、`N≤16`、`1≤K≤64`；较小矩阵用行列掩码屏蔽。按 K 顺序发射，每次等待阵列排空；脚本将结果与期望值逐项比较。整数要求完全一致，FP32 报告 NRMSE 和最大绝对误差，并按 `max_nrmse`（默认 3%）判定。期望值由向量作者提供，脚本不从 DUT 输出生成标准答案。示例数据用于打通路径，不代表 KWT-1 真实层数据或整模型准确率。

当前 perf 文件的 `status` 为仿真状态（1 运行、2 完成），`compute_cycles` 统计观察到阵列累加更新的周期，`wait_cycles` 为运行窗口内没有累加更新的周期，`total_cycles` 为复位释放后的总周期。阵列级测试没有 DMA，故 `dma_cycles=0`。这些是 harness 观察值，不是软件可读寄存器。现有 RTL 没有性能计数器或状态查询端口；要实现软件查询必须另行修改 RTL 与地址/指令接口。

此 harness 直接测试 `kwt_pulse_array` 的数据路径。当前 `kwt_tile_engine` 固定 INT16/INT32 且包含 SRAM、偏置、ReLU 和写回，原有 `rtl/pulse_array/tb/run_engine.sh` 覆盖它。`rtl/control/tb/run_e203.sh --full-layer` 覆盖 99×64×256 的公式生成整数整层；这里的 FP32 NRMSE 只针对向量所给阵列输出，不是量化误差或语音识别准确率。
