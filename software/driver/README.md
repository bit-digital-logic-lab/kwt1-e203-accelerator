# 当前 NICE-like E203 模型的 FFN1 驱动

[`kwt_nice_driver.c`](kwt_nice_driver.c) 为现有仿真模型生成 208 个 32 字节描述符，并通过 custom-0/funct3=110 逐块提交、检查返回状态。它针对本科必做的 `99×64 @ 64×256 + bias`；GELU/ReLU 均关闭。A、W 必须由主机预先按 `rt,k,r` 与 `ct,k,c` 排列，bias 为 INT32，输出按 tile 顺序写回。格式和当前真实向量生成器见[`prepare_real_e203_vectors.py`](../../hardware/rtl/control/tb/prepare_real_e203_vectors.py)。

描述符前 16 字节依次是 A、W、bias、输出字节地址；后 16 字节依次是 K/选项、行列掩码、两个保留字。`kwt_nice_prepare_ffn1` 检查 16 字节对齐、512 KiB 边界及五块区域不重叠。`kwt_nice_run_ffn1` 只在 RV32 编译时发 NICE 指令；返回第一个非零硬件状态，并可报告失败 tile 序号。当前 E203 全层仿真仍由原汇编循环 `e203_kwt_full_layer.S` 运行，此 C 驱动的**描述符和编译**已验证，尚未在 E203 仿真中执行 C 版本。

```bash
cc -std=c11 -O2 -Wall -Wextra -Werror \
  software/driver/kwt_nice_driver.c software/driver/test_kwt_nice_driver.c \
  -o /tmp/kwt-nice-driver-test && /tmp/kwt-nice-driver-test
riscv64-linux-gnu-gcc -std=c11 -O2 -Wall -Wextra -Werror -ffreestanding \
  -march=rv32imac -mabi=ilp32 -c software/driver/kwt_nice_driver.c \
  -o /tmp/kwt_nice_driver_rv32.o
```

组委会正式 SoC 的总线地址图、NICE 连接和 SRAM 宏接口目前缺失。上面的地址及指令编码与**本仓库的仿真模型**一致，移植到正式平台须重新核对，不能把宿主机编译测试当作 SoC 驱动联调通过。
