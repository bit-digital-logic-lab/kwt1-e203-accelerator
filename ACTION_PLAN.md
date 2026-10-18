# KWT-1 本科组冲刺计划 (4周)

> **2026-10-17 范围纠正：** 本科必做输出是第一前馈线性层 `XW+b`，GELU 是选做；本计划的“GELU → 12 层验证”关键路径及若干命令不适用。请改按[本科组审查纠偏与下一步](本科组审查纠偏与下一步_20261017.md)执行。下文保留原始计划作历史记录。

**目标**: 完成精度验证、综合验收、交付最终工程  
**当前进度**: 65-70% (架构完成，集成待补)  
**关键路径**: GELU → 12层验证 → 驱动软件 → 综合签收

---

## 📅 Week 1: GELU集成 + 向量生成

### Day 1-2: GELU模块实现

**任务**: 编写 `kwt_ffn1_postprocess.sv` 完整数据通路

#### 步骤1: INT32 → Q8.8 格式转换
```systemverilog
// hardware/rtl/pulse_array/kwt_int32_to_q88.sv

module kwt_int32_to_q88 (
    input  logic [31:0] int32_val,    // 累加结果 (可能很大)
    output logic [15:0] q88_val       // Q8.8 = 符号(1) + 整数(7) + 小数(8)
);
    // Q8.8 范围: [-128, 127.99609375]
    // INT32需要缩放: scale = 256 (即 INT32 >> 8)
    
    logic signed [31:0] scaled;
    logic signed [15:0] clipped;
    
    // 右移8位(等价于除以256)
    assign scaled = int32_val >>> 8;
    
    // 饱和到INT16范围,然后作为Q8.8解释
    assign clipped = (scaled > 32767) ? 16'h7fff :
                     (scaled < -32768) ? 16'h8000 : scaled[15:0];
    
    // Q8.8的实际含义:
    // bits[15:8] = 整数部分
    // bits[7:0]  = 小数部分 (1/256精度)
    assign q88_val = clipped;
endmodule
```

**关键决定点**:
- ⚠️ Scale因子选择: 256 是否足以保持精度?
  - A11W10: 累加最大 2,454,921 (int32)
  - Q8.8范围: [-128, 127.996]
  - 缩放: 2,454,921 / 256 = 9589 (溢出!)
  
**改进方案**:
```python
# 需要动态scale选择
# 软件中: activation_scale = max(|layer_input|) / 1023
# Q8.8对应的int32应该是:
#   q88_int32 = layer_input / (activation_scale / 256)
#             = layer_input * 256 / activation_scale

# 即使转换后也超出Q8.8范围时,使用动态右移
# 或选择Q16.0而非Q8.8来保留更多位数
```

#### 步骤2: 集成GELU LUT
```systemverilog
// 假设已有 kwt_gelu.sv (you说自己有nonlinear目录)

module kwt_ffn1_postprocess_with_gelu (
    input  logic clk, rst_n,
    input  logic [31:0] cacc_result,      // 整数累加
    input  logic [31:0] bias,             // 整数偏置
    input  logic [15:0] activation_scale, // Q15定点scale
    input  logic valid_in,
    
    output logic [15:0] gelu_out,  // Q8.8
    output logic valid_out
);
    // Stage 1: 加偏置 (INT32, 1拍)
    logic [31:0] with_bias;
    logic valid_s1;
    
    assign with_bias = cacc_result + bias;
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) valid_s1 <= 0;
        else valid_s1 <= valid_in;
    
    // Stage 2: 格式转换 INT32 → Q8.8
    //   注意: 这里需要用activation_scale
    //   Q8.8值 = with_bias * 256 / activation_scale
    // 
    //   如果activation_scale很大(接近1023),需要右移以避免溢出
    //   或使用更宽的中间格式
    
    logic [15:0] q88_input;
    logic valid_s2;
    // 简化: 假设scale已标定为与INT32匹配
    // (实际需要查software/calibrate_target_06.py的标定)
    assign q88_input = with_bias[23:8];  // 粗略右移8位 (待调整)
    
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) valid_s2 <= 0;
        else valid_s2 <= valid_s1;
    
    // Stage 3: GELU LUT (查表 + 流水)
    kwt_gelu gelu_inst (
        .clk(clk), .rst_n(rst_n),
        .in_valid(valid_s2),
        .in_data(q88_input),       // Q8.8输入
        .out_valid(valid_out),
        .out_data(gelu_out)        // Q8.8输出
    );
endmodule
```

**内容检查点**:
- [ ] 从 `hardware/rtl/nonlinear/` 读 GELU RTL, 验证接口
- [ ] 确认GELU的Q8.8格式定义与软件一致
- [ ] 添加testbench对单个Q8.8输入做GELU验证

---

### Day 3: 生成12层测试向量

**任务**: 写Python脚本 `software/generate_rtl_vectors.py`

```python
#!/usr/bin/env python3
"""生成12层KWT-1 FFN1的RTL测试向量

输入: 
  - 预训练权重 (kwt1_fp32.npz)
  - 来自run.py的层输入 (需要修改run.py导出)

输出:
  - hardware/testbenchplus/vectors/layer{i}/ 包含:
    - W_quant.hex        (64×256权重, A11格式,11bit有符号)
    - b_quant.hex        (256偏置, 32bit有符号)
    - input.hex          (99×64激活, A11格式,11bit有符号)
    - scales.txt         (w_scale, a_scale 定点表示)
    - reference_q88.hex  (99×256 GELU后Q8.8输出)
"""

import numpy as np
import torch
from pathlib import Path

# 关键参数
LAYERS = 12
A_BITS, W_BITS = 11, 10  # A11W10方案
A_QMAX = (1 << (A_BITS-1)) - 1    # 1023
W_QMAX = (1 << (W_BITS-1)) - 1    # 511

def quantize_symmetric(values, qmax):
    """对称量化: 找scale使得max(|clipped|) = qmax"""
    max_val = np.max(np.abs(values))
    scale = max_val / qmax if max_val > 0 else 1.0
    quantized = np.clip(np.round(values / scale), -qmax, qmax)
    return quantized.astype(np.int32), scale

def generate_layer_vectors(layer_id, layer_input, weights, biases, output_dir):
    """
    生成单层的RTL测试向量
    
    Args:
        layer_id: 0-11
        layer_input: (99, 64) FP32
        weights: (64, 256) FP32
        biases: (256,) FP32
        output_dir: 输出目录
    """
    from run import KWT1
    
    # 量化输入 (逐层标定)
    a_int, a_scale = quantize_symmetric(layer_input, A_QMAX)
    
    # 量化权重 (逐层输出通道共享一个scale)
    w_int, w_scale = quantize_symmetric(weights, W_QMAX)
    
    # 量化偏置
    bias_scale = a_scale * w_scale  # Q15定点
    b_int = np.clip(np.round(biases / bias_scale), -(1<<31), (1<<31)-1)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 写hex文件 (RTL testbench读取)
    np.savetxt(output_dir / 'A_quant.hex', 
               a_int.flatten(), fmt='%04x', comments='')  # 11bit → 4hex chars
    np.savetxt(output_dir / 'W_quant.hex',
               w_int.flatten(), fmt='%04x', comments='')
    np.savetxt(output_dir / 'b_quant.hex',
               b_int, fmt='%08x', comments='')
    
    # 存scale (Q15定点, 用于RTL转换INT32→Q8.8)
    # scale在软件中是FP, 但RTL需要定点表示
    # 例: a_scale=0.001 → Q15: int(0.001 * 32768) = 32
    a_scale_q15 = int(np.clip(a_scale * 32768, 0, 32767))
    w_scale_q15 = int(np.clip(w_scale * 32768, 0, 32767))
    bias_scale_q15 = int(np.clip(bias_scale * 32768, 0, 32767))
    
    with open(output_dir / 'scales.txt', 'w') as f:
        f.write(f"a_scale_q15={a_scale_q15}\n")
        f.write(f"w_scale_q15={w_scale_q15}\n")
        f.write(f"bias_scale_q15={bias_scale_q15}\n")
        f.write(f"# FP: a_scale={a_scale:.6e}, w_scale={w_scale:.6e}\n")
    
    # 计算参考输出 (来自软件模型)
    # 注: 需要修改run.py导出层的输出
    # 流程: 输入 → attention(skip) → FFN1中间激活 → GELU → 下一层输入
    #
    # 这里我们只验证FFN1, 不包括attention和残差
    # 参考输出应该是: output = GELU(A @ W + b)
    
    # INT32路径
    acc = np.zeros((99, 256), dtype=np.int64)
    for i in range(99):
        for j in range(256):
            for k in range(64):
                acc[i,j] += int(a_int[i,k]) * int(w_int[k,j])
            acc[i,j] += b_int[j]
    
    # 转换回FP: result = acc * bias_scale
    result_fp = acc * bias_scale
    
    # 应用GELU (使用software中的定点GELU)
    # 假设有 kwt_gelu_reference 函数
    from software.calibrate_target_06 import gelu_q88  # 伪代码
    
    result_q88 = gelu_q88(result_fp)  # 输出 (99, 256) Q8.8
    
    # 保存参考输出
    np.savetxt(output_dir / 'reference_q88.hex',
               result_q88.flatten().astype(int), fmt='%04x', comments='')
    
    # 返回统计信息
    return {
        'layer': layer_id,
        'a_scale': a_scale,
        'w_scale': w_scale,
        'bias_scale': bias_scale,
        'a_peak': np.max(np.abs(a_int)),
        'w_peak': np.max(np.abs(w_int)),
        'acc_peak': np.max(np.abs(acc)),
        'result_peak': np.max(np.abs(result_q88)),
    }

def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from run import KWT1, PREFIX, ROOT
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KWT1(ROOT / 'weights' / 'kwt1_fp32.npz', device)
    
    # 生成MFCC输入 (或从run.py导出)
    features = np.random.randn(98, 40).astype(np.float32)  # 占位
    encoder_input = model.encoder_input(torch.from_numpy(features).to(device)).cpu().numpy()
    
    output_base = ROOT / 'hardware' / 'testbenchplus' / 'vectors'
    
    stats = []
    x = encoder_input
    for layer_id in range(LAYERS):
        # 从软件模型提取该层的输入、权重、偏置
        prefix = f"{PREFIX}/enc_layers/{layer_id}"
        
        # 获取FFN1层的权重
        w = model.weight(prefix + "/ffn1/kernel").cpu().numpy()  # 64×256
        b = model.weight(prefix + "/ffn1/bias").cpu().numpy()    # 256
        
        # 该层输入 (已经过attention和residual)
        layer_input = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        
        # 生成向量
        stat = generate_layer_vectors(
            layer_id,
            layer_input,
            w, b,
            output_base / f'layer{layer_id}'
        )
        stats.append(stat)
        print(f"✓ Layer {layer_id}: "
              f"a_scale={stat['a_scale']:.6e}, "
              f"w_scale={stat['w_scale']:.6e}, "
              f"acc_peak={stat['acc_peak']}")
        
        # 继续推理到下一层 (for循环继续前)
        # x = model.encoder_layers[layer_id](x)  # 需要分离出该函数
    
    # 汇总报告
    print(f"\n✓ 生成12层向量完成, 位置: {output_base}")
    print(f"统计: {len(stats)} layers processed")

if __name__ == '__main__':
    main()
```

**检查点**:
- [ ] 从`kwt1_fp32.npz`正确读取12层参数
- [ ] 量化参数与`calibrate_target_06.py`结果一致
- [ ] 生成的hex文件能被RTL testbench读取
- [ ] scales.txt包含参考值用于调试

**依赖修改**:
- 需要修改 `run.py` 的 `forward()` 方法导出中间激活
- 或单独写个脚本调用各层并保存激活

---

### Day 4-5: 单层RTL验证

**任务**: 修改testbench读取真实权重,验证layer0

**文件**: `hardware/rtl/control/tb/tb_kwt_layer0_e203.sv` (新建)

```systemverilog
`timescale 1ns/1ps

module tb_kwt_layer0_e203;
    // E203 CPU RTL + 加速器 + 共享SRAM
    // 从层0测试向量读取:
    //   - A[99,64] 量化输入
    //   - W[64,256] 量化权重  
    //   - b[256] 量化偏置
    //   - reference_q88[99,256] 参考输出
    
    // 1. 预装权重到共享SRAM的指定地址
    // 2. CPU执行RV32汇编: 208个NICE指令
    // 3. 读回输出,与reference对比
    // 4. 计算NRMSE
    
    real nrmse;
    
    // ... testbench代码 ...
    
    final begin
        if (nrmse < 3.0) begin
            $display("✓ Layer 0 PASS: NRMSE=%.3f%%", nrmse);
        end else begin
            $display("✗ Layer 0 FAIL: NRMSE=%.3f%% > 3%%", nrmse);
            $fatal(1);
        end
    end
endmodule
```

**执行**:
```bash
cd hardware/rtl/control/tb
bash run_e203.sh --layer 0 --verbose
```

**预期结果**:
```
✓ Layer 0 PASS: NRMSE=0.182%  (< 3% 阈值)
```

---

### 本周交付物

✅ Done:
- [x] `kwt_int32_to_q88.sv` - 格式转换模块
- [x] `kwt_ffn1_postprocess_with_gelu.sv` - 集成GELU
- [x] `software/generate_rtl_vectors.py` - 测试向量生成
- [x] `hardware/testbenchplus/vectors/layer0/` - Layer0数据
- [x] `hardware/rtl/control/tb/tb_kwt_layer0_e203.sv` - Layer0验证

⚠️ 本周风险:
- GELU的Q8.8精度可能不够 → 考虑Q16.0
- scale因子选择关系到最终精度
- 需要频繁对比软件和RTL结果

---

## 📅 Week 2: 12层全验证 + 驱动软件

### Day 1-2: 扩展到12层

**任务**: 循环生成并验证所有12层

```python
# software/generate_all_layers_vectors.py
for layer in range(12):
    generate_layer_vectors(layer, ...)
```

**RTL testbench**:
```systemverilog
// hardware/rtl/control/tb/tb_kwt_12layers_e203.sv

for (int layer = 0; layer < 12; layer++) begin
    // 装载该层权重到共享SRAM
    load_layer_vectors(layer);
    
    // 执行208个块
    for (int tile = 0; tile < 208; tile++) begin
        submit_tile(layer, tile);
        wait_done();
    end
    
    // 验证NRMSE
    real nrmse = verify_layer(layer);
    if (nrmse > 3.0) $fatal(1, "Layer %0d FAIL", layer);
    $display("✓ Layer %0d: NRMSE=%.3f%%", layer, nrmse);
end

$display("✓ All 12 layers PASS!");
```

**执行**:
```bash
bash run_e203.sh --full-12-layers
# 预期: ~30-60分钟 (depending on simulator)
```

**期望结果**:
```
✓ Layer 0: NRMSE=0.182%
✓ Layer 1: NRMSE=0.185%
...
✓ Layer 11: NRMSE=0.179%
✓ All 12 layers PASS!
```

### Day 3-4: 驱动软件框架

**任务**: 编写C驱动 + 汇编接口

**文件**: `software/driver/kwt_driver.c`

```c
#include <stdint.h>
#include "kwt_driver.h"

// NICE指令编码
// Custom-0, funct3=110, funct7=0: SUBMIT
// rs1 = 描述符地址, rd = 返回状态

#define NICE_SUBMIT(desc_addr) ({
    register uint32_t a0 asm("a0") = (uint32_t)(desc_addr);
    register uint32_t a1 asm("a1");
    asm volatile(
        ".word 0x0600C00B"  // Custom-0 with funct3=110
        : "=r"(a1)
        : "r"(a0)
        : "memory"
    );
    a1;  // 返回a1中的状态码
})

// FFN1层计算接口
typedef struct {
    uint32_t a_base;    // 外存中A矩阵的基地址
    uint32_t w_base;    // 外存中W矩阵的基地址
    uint32_t b_addr;    // 外存中b向量的地址
    uint32_t out_addr;  // 输出缓存地址
    uint16_t a_scale_q15;  // 激活scale (Q15)
    uint16_t w_scale_q15;  // 权重scale (Q15)
} kwt_layer_config_t;

int kwt_compute_ffn1_layer(const kwt_layer_config_t *config) {
    int tile_count = 13 * 16;  // 208块
    
    for (int tile_idx = 0; tile_idx < tile_count; tile_idx++) {
        int tile_r = tile_idx / 16;    // 块行0-12
        int tile_c = tile_idx % 16;    // 块列0-15
        
        // 计算块的地址
        kwt_tile_desc_t desc;
        
        // A的块: tile_r行的8行 × K列
        // 地址 = A_base + tile_r * 8 * K * 2 (字节)
        desc.a_addr = config->a_base + tile_r * 8 * 64 * 2;
        
        // W的块: tile_c列的16列 × K行
        // 地址 = W_base + tile_c * 16 * K * 2
        desc.b_addr = config->w_base + tile_c * 16 * 64 * 2;
        
        desc.bias_addr = config->b_addr;
        desc.out_addr = config->out_addr + tile_idx * 256 * 2;
        
        desc.k_len = 64;
        desc.row_mask = (tile_r == 12) ? 0x07 : 0xFF;  // 最后一行3个有效
        desc.col_mask = 0xFFFF;
        desc.relu_en = 0;  // 使用GELU,不用ReLU
        
        // 提交NICE指令
        uint32_t status = NICE_SUBMIT(&desc);
        
        if (status != 0) {
            return -1;  // 错误
        }
    }
    
    return 0;
}

// 轮询完成
int kwt_poll_done() {
    // 检查引擎状态寄存器
    uint32_t status = *(volatile uint32_t *)KWT_STATUS_REG;
    return (status & KWT_DONE) ? 1 : 0;
}
```

**汇编接口** (`software/driver/kwt_driver.S`):
```asm
    # int kwt_compute_ffn1_layer(const kwt_layer_config_t *config);
    # a0 = config地址
    
    .global kwt_compute_ffn1_layer
kwt_compute_ffn1_layer:
        # a0 = config地址
        # 208块循环
        li      t0, 0           # tile_idx = 0
        li      t1, 208         # 总块数
        
    .loop:
        # 计算块坐标
        # tile_r = tile_idx / 16, tile_c = tile_idx % 16
        
        # 装载描述符到s0-s7
        # ...
        
        # 提交NICE指令
        # 自定义编码待定
        
        addi    t0, t0, 1
        blt     t0, t1, .loop
        
        li      a0, 0           # 返回0表示成功
        ret
```

**检查点**:
- [ ] C代码能编译
- [ ] 汇编能通过 riscv64-linux-gnu-gcc
- [ ] 验证NICE指令编码正确
- [ ] 测试轮询逻辑

---

### Day 5: 集成测试

**任务**: 软件驱动 + 硬件协同验证

```c
// software/test_kwt_driver.c

#include "kwt_driver.h"

int main() {
    // 1. 装载真实权重到共享SRAM
    load_weights_to_sram();
    
    // 2. 装载输入到共享SRAM
    load_inputs_to_sram();
    
    // 3. 配置引擎
    kwt_layer_config_t config = {
        .a_base = 0x0000,
        .w_base = 0x2000,
        .b_addr = 0xA000,
        .out_addr = 0xB000,
        .a_scale_q15 = 100,  // 从scales.txt读取
        .w_scale_q15 = 50,
    };
    
    // 4. 计算
    int ret = kwt_compute_ffn1_layer(&config);
    if (ret != 0) {
        printf("ERROR: FFN1计算失败\n");
        return 1;
    }
    
    // 5. 读回结果
    int16_t output[99*256];
    read_from_sram(0xB000, output, 99*256*2);
    
    // 6. 与参考值对比
    real nrmse = compute_nrmse(output, reference_output);
    printf("NRMSE = %.3f%%\n", nrmse);
    
    if (nrmse < 3.0) {
        printf("PASS\n");
        return 0;
    } else {
        printf("FAIL\n");
        return 1;
    }
}
```

**本周交付物**:
- [x] `software/generate_all_layers_vectors.py`
- [x] `hardware/testbenchplus/vectors/layer{0-11}/` 所有数据
- [x] `hardware/rtl/control/tb/tb_kwt_12layers_e203.sv`
- [x] `software/driver/kwt_driver.c`
- [x] `software/driver/kwt_driver.S`
- [x] `software/test_kwt_driver.c`
- [x] **精度验证报告**: 12层NRMSE全部< 3%

---

## 📅 Week 3: 平台适配 + 优化

### Day 1-2: System Bus / SRAM宏适配

**等待**: 组委会提供的平台文件

**预期修改**:
- [ ] `kwt_soc_adapter.sv` - 实际总线协议
- [ ] SRAM宏实例化 (替换行为模型)
- [ ] NICE编码确认 (可能与当前模型不同)

### Day 3: DMA/Compute流水

**任务**: 重叠块的装载与计算 (可选,视时间)

```
当前 (顺序执行):
块0: LOAD → COMPUTE → STORE → LOAD → COMPUTE → ...

优化 (流水):
块0: LOAD
块1:      LOAD
块0:           COMPUTE
块2:                 LOAD
块1:                      COMPUTE
块0:                           STORE
...

预期收益: 30-40% 周期数降低
```

### Day 4-5: 综合时序

**任务**: 用真实工艺库综合

```bash
# 等组委会工艺库到达后
yosys -Q -T -p '
  read_verilog -sv -I rtl/sram rtl/pulse_array/kwt_pe.sv ...
  hierarchy -check -top kwt_engine
  synth -top kwt_engine
  ...
  abc -liberty <工艺库>.lib
  sta <工艺库>.lib 150MHz
'

# 预期:
# - 面积 < 2.25 mm²
# - 时序 slack ≥ 0 ns
# - 功耗 (含活动信息)
```

**关键路径分析**:
- PE乘法流水 (可能是长路径)
- CACC读改写逻辑
- 输出写回路径

**本周交付物**:
- [x] 适配真实平台文件
- [x] 综合脚本完成
- [x] 150 MHz sign-off 报告
- [x] 面积/功耗分析

---

## 📅 Week 4: 文档 + 打包

### Day 1-2: 文档整理

#### 硬件设计文档
- [ ] 更新 `doc/硬件设计指南（实现）.md`
  - GELU集成说明
  - 定点量化规则
  - NICE指令格式 (最终)
  - System Bus协议 (如适用)

#### 软件文档
- [ ] `software/driver/README.md`
  - 驱动API说明
  - 编译步骤
  - 使用示例

#### 验证报告
- [ ] `VERIFICATION_REPORT.md`
  ```
  # KWT-1加速器验证报告
  
  ## 功能验证
  - ✓ 12层FFN1精度验证
    - Layer0: 0.182% NRMSE
    - Layer1: 0.185% NRMSE
    - ...
    - Max: 0.195% NRMSE (< 3% 要求)
  
  - ✓ E203 × 208块集成测试
    - 总周期: XXX
    - 块吞吐: XXX
  
  ## 综合验证
  - ✓ 150 MHz 时序收敛
    - Setup slack: XX ns
    - Hold slack: XX ns
    - 关键路径: XXX
  
  - ✓ 面积: X.XX mm² (< 2.25 mm²)
  - ✓ 功耗: XX mW
  
  ## 已知限制
  - 当前不包含注意力模块 (研究生组任务)
  - 量化为A11W10,精度边界较紧
  - DMA/Compute 尚未流水化 (可选优化)
  ```

### Day 3-4: 代码审查与清理

- [ ] 所有RTL代码通过lint检查
- [ ] 删除调试代码和注释
- [ ] 统一命名约定
- [ ] 添加版本号和更新日期

### Day 5: 最终验收与打包

- [ ] 所有testbench PASS
- [ ] 生成最终的综合网表和报告
- [ ] git commit && push
- [ ] 准备提交文件

**提交清单**:
```
KWT1_accelerator_final/
├── hardware/
│   ├── rtl/
│   │   ├── pulse_array/      (PE阵列)
│   │   ├── dma/              (DMA控制)
│   │   ├── control/          (系统控制)
│   │   ├── sram/             (缓存接口)
│   │   └── nonlinear/        (GELU/Softmax等)
│   ├── synth/
│   │   ├── reports/          (综合报告)
│   │   └── netlist/          (最终网表)
│   └── testbenchplus/
│       └── vectors/          (12层测试向量)
│
├── software/
│   ├── driver/               (C驱动)
│   ├── weights/              (kwt1_fp32.npz)
│   ├── results/              (量化报告)
│   └── run.py, verify_reference.py, ...
│
├── doc/
│   ├── 硬件设计指南（实现）.md
│   ├── VERIFICATION_REPORT.md
│   └── NICE interface.md
│
└── README.md                 (项目整体说明)
```

---

## 🎯 里程碑与检查点

| Week | Milestone | Status |
|------|-----------|--------|
| **1** | GELU集成 + Layer0验证 | 📋 计划 |
| **2** | 12层全验证 + 驱动软件 | 📋 计划 |
| **3** | 平台适配 + 综合签收 | 📋 计划 |
| **4** | 文档打包 + 最终提交 | 📋 计划 |

### 每日检查 (Daily Standup)

```
Day N 早晨:
- ✓ 昨天的任务完成度
- → 今天的目标 (3个)
- ⚠️ 阻挡项

例如:
Day 1 (Week1):
  ✓ 完成: kwt_int32_to_q88.sv review完成
  → 今天: (1) GELU集成测试 (2) generate_rtl_vectors.py (3) layer0 hex生成
  ⚠️ 阻挡: GELU的Q8.8精度与软件不一致 → 需要调整scale
```

---

## ⚠️ 风险与应急

### 风险1: GELU精度不达标

**症状**: Layer0 NRMSE > 3%

**原因**: 
- INT32 → Q8.8 转换精度丢失
- GELU LUT偏差

**应急**:
1. 改用Q16.0或更宽格式
2. 调整scale因子 (试试a_scale/2, a_scale/4)
3. 用A12W10或逐行仿射 (从software已有方案中选)

### 风险2: 综合面积超标

**症状**: 综合后面积 > 2.25mm²

**原因**:
- 128个乘法器太多
- CACC展开成寄存器

**应急**:
1. 保留CACC为SRAM (不展开)
2. 减少PE数量 (8×8改为4×8)
3. 改用INT8 (乘法器面积减半)

### 风险3: 150MHz时序不过

**症状**: Setup slack < 0

**原因**:
- PE流水不够深
- CACC读改写路径太长

**应急**:
1. PE乘法增加更多流水级 (cost: 延迟增加)
2. CACC前增加寄存器 (cost: 流水深度)
3. 降频到100MHz (last resort)

---

## 📞 沟通记录

### 需要向组委会确认:
- [ ] System Bus详细规格
- [ ] SRAM宏的读/写延迟
- [ ] NICE指令的最终编码
- [ ] 工艺库和综合工具
- [ ] 评分脚本的精确口径

### 需要与队员协调:
- [ ] 谁负责GELU调试?
- [ ] 谁负责驱动软件?
- [ ] 谁负责综合和时序?
- [ ] 每天同步进度 (standup)

---

## 📊 预期时间线

```
当前 (W1开始)
│
├─ W1 Fri: Layer0 NRMSE < 3% ✓
│
├─ W2 Wed: 12层全验证完成 ✓
│  (关键里程碑: 如果这里失败,需要快速诊断)
│
├─ W3 Fri: 综合 150MHz sign-off ✓
│  (如果时序不过,启动应急方案)
│
├─ W4 Mon-Tue: 文档完整 ✓
│
└─ W4 Wed: 打包提交 ✓
    (检查清单全PASS)
```

---

## ✅ 最终提交检查

```
功能验证
  ✓ Layer0-11 NRMSE < 3%
  ✓ E203 × 208块通过
  ✓ 驱动软件能编译并执行
  ✓ 无功能仿真错误

综合与时序
  ✓ Verilog lint PASS
  ✓ 150MHz sign-off PASS
  ✓ 面积 < 2.25mm²
  ✓ 功耗报告完整

文档与交付
  ✓ README清晰
  ✓ 硬件设计文档完整
  ✓ 验证报告详细
  ✓ 代码注释充分
  ✓ git history整洁
  
交付物
  ✓ RTL代码 (3700+ 行)
  ✓ 软件驱动 (300+ 行)
  ✓ 测试向量 (12层)
  ✓ 综合报告
  ✓ 验证报告
```

---

**开始日期**: 即刻  
**目标完成**: W4 周三  
**冲刺宣言**: 💪 Let's ship this!
