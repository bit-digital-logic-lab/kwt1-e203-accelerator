# KWT-1 加速器代码深度审查报告

> **2026-10-17 范围纠正：** 本科必做输出是第一前馈线性层 `XW+b`，原题明确 GELU、第二线性层为选做；本文将 GELU 输出作为 RTL/NRMSE 目标属于错误验收边界。请先读[本科组审查纠偏与下一步](本科组审查纠偏与下一步_20261017.md)。下文保留原始审查文本作历史记录。

**审查日期**: 2024年10月  
**审查范围**: RTL设计 (3762行) + 软件模型 (量化框架)  
**评分**: ⭐⭐⭐⭐ (4/5星 - 架构良好，集成待完善)

---

## 📋 执行摘要

### 核心发现

**强项**:
- ✅ 脉冲阵列设计合理，PE流水控制精细
- ✅ CACC集中式设计创新，解决本地累加器面积问题
- ✅ 双页乒乓缓存机制完整，支持DMA/Compute分离
- ✅ 软件量化模型完备，A11W10达到0.18% NRMSE

**严重缺陷**:
- 🔴 **GELU未集成到数据通路** - 无法验证真实FFN1精度
- 🔴 **缺少真实权重的RTL验证** - 只用合成数据测试
- 🔴 **驱动软件为空** - C语言接口未实现
- 🔴 **系统集成未定型** - SoC平台适配等组委会文件

**中等问题**:
- ⚠️ 控制FSM顺序执行，DMA/Compute未流水
- ⚠️ 地址生成只支持连续访问，不支持步长抽取
- ⚠️ 综合结果未知，150 MHz可达性未验证

---

## 🔍 模块级审查

### 1. PE (kwt_pe.sv) ⭐⭐⭐⭐⭐

**优点**:
```systemverilog
// 整数乘法分4bit段逐级积累，FP32用24bit尾数分6段
// 这个设计避免了大数字乘法器的延迟问题
localparam int MUL_STAGES = DATA_W / 4;  // INT16: 4级, INT32: 8级
```
- ✅ 乘法流水精心设计（INT16: 4拍, INT32: 8拍）
- ✅ FP32特殊值处理完整（NaN, Inf, Subnormal → zero）
- ✅ 加法流水自适应（按数据宽度分段）
- ✅ 单PE部分和有数据依赖，间隔约束清晰

**问题**:
```systemverilog
// 第88行: 传播与运算时序设计有隐性假设
always_ff @(posedge clk or negedge rst_n) begin
    // a_valid_out <= !acc_clear && a_valid_in;
    // 这里依赖 acc_clear 同步重置，但没有确保它的timing pathDoes not block propagation
end
```
- ⚠️ `acc_clear`同步性未明确说明（可能需要在流水排空后置位）
- ⚠️ 没有定义PE之间valid信号的最小发射间隔（应该在阵列文档明确）

**建议**:
```systemverilog
// 添加注释说明间隔要求
// PE 单部分和有MUL_STAGES + ADD_STAGES拍数据依赖
// 同一PE连续乘积的最小间隔 = MUL_STAGES + ADD_STAGES + 1
// 例: INT16时最小间隔 = 4 + 2 + 1 = 7拍
localparam int MIN_INTERVAL = MUL_STAGES + ADD_STAGES + 1;
```

---

### 2. CACC (kwt_cacc.sv) ⭐⭐⭐⭐⭐

**创意设计**:
```systemverilog
// 关键洞察: 取消PE本地累加，改为行级中央bank
// 8行×16列阵列 → 8个1R1W bank, 每个16个32-bit部分和
// 比局部累加节省 128×32-bit = 512 bit 的触发器!
logic signed [ACC_W-1:0] psum_mem [0:COLS-1];  // 每行一个memory
```

**优点**:
- ✅ 读改写逻辑明确，forward机制避免两拍延迟
- ✅ 碰撞检测（同行同拍两个乘积 → 报错）
- ✅ 有效性位追踪，清空后数据不可见
- ✅ `dump_req`接口支持读取任意bank

**潜在问题**:
```systemverilog
// 第70行: pending机制有corner case
if (!clear) begin
    if (pending) psum_mem[pending_col] <= pending_sum;
    if (row_fire[r]) read_data <= psum_mem[row_col[r]];
end

// 问题: 如果clear在pending写入时断言会怎样?
// 应该是clear == 1时停止新读写，但forward_valid等可能还在更新
```

**建议**:
```systemverilog
// 添加状态机确保clear的安全性
// 方案: 在CACC外部确保清零后等待1拍再启动新任务
// 或在CACC内部添加clear_done信号

output logic clear_done,  // 所有bank已清零且pending被丢弃

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) clear_done <= 1'b1;
    else if (clear) clear_done <= 1'b0;
    else if (pending == 1'b0) clear_done <= 1'b1;
end
```

---

### 3. 双页乒乓缓存引擎 (kwt_pingpong_cacc_engine.sv) ⭐⭐⭐⭐

**架构优势**:
```systemverilog
// A/B各2页 (load_page, compute_page)
// DMA可在compute_page计算时写入load_page
// 理论上消除了DMA等待时间

input logic load_page,        // DMA操作这一页
input logic compute_page,     // 计算使用这一页
```

**当前限制**:
```systemverilog
// kwt_tile_system_model.sv 第88行
// 但实际没有利用这个特性 - 总是等待DMA完成再计算
// 改进需要:
// 1. Engine维护块队列
// 2. DMA装块N+1时计算块N
// 3. 同步两个页的切换
```

**缓存配置检查** ✅
- ✅ A/B缓存: 1024×16bit (8×K_len 和 16×K_len 数据)
- ✅ 输出缓存: 1024×16bit (256个INT32结果)
- ✅ SRAM宏匹配: 题目指定16bit×128/1024/4096
- ⚠️ 256×8×16bit寄存器用于偏置（可优化为SRAM或触发器）

**状态机复杂度分析**:
```
当前11个状态:
IDLE → CLEAR_CACC → READ_ISSUE → READ_CAPTURE → LAUNCH 
    → DRAIN → DUMP_REQ → DUMP_WAIT → WRITE_LO → WRITE_HI → DONE

问题: 这个状态机是为单块优化的，扩展到多块时会膨胀
建议: 考虑分层FSM (顶层块调度 + 底层K迭代)
```

---

### 4. DMA控制器 (kwt_dma.sv) ⭐⭐⭐⭐

**设计特点**:
```systemverilog
// 参数化的突发传输控制器
// 支持 N×16bit SRAM ↔ 128bit系统总线
// LANES = BUS_W / SRAM_DATA_W = 128/16 = 8 lanes
localparam int LANES = BUS_W / SRAM_DATA_W;
```

**优点**:
- ✅ 字节级写掩码支持 (bus_wstrb) 处理尾拍
- ✅ 读响应管道化 (请求后一拍数据到达)
- ✅ 支持三个独立缓存的多路复用
- ✅ 地址/长度检查完整

**限制**:
```systemverilog
// 只支持连续地址访问
// 假设权重在共享SRAM中是这样排列的:
// A[k][r] @ byte 0
// B[k][c] @ byte 8K_len
// 
// 问题: 实际权重是全局存储的形式 (shape 64×256)
// 需要计算步长,而不是连续块
// 当前DMA无法做到这一点

// 例如: 要装A的第k行(8个INT16值)
// 应该从外存的 base + k*stride 读取
// 但DMA只能做 base 到 base+len 的连续读
```

**改进方案**:
```systemverilog
// 增加步长参数
input  [31:0] stride,           // 行步长
input  [15:0] step_count,       // 需要几步
input  [15:0] step_len,         // 每步的字数

// 状态机修改: 每次读STEP_LEN个字后,地址加STRIDE
```

---

### 5. 系统集成 (kwt_tile_system_model.sv & kwt_nice_system_model.sv) ⭐⭐⭐

**关键问题: 控制流过度复杂**

```systemverilog
// kwt_tile_system_model.sv 顺序执行 13个状态
typedef enum {
    IDLE, 
    LOAD_A, WAIT_A,      // ← 等待DMA
    LOAD_B, WAIT_B,      // ← 等待DMA  
    LOAD_BIAS, WAIT_BIAS,// ← 等待DMA
    BIAS_REQ, BIAS_CAPTURE,  // ← 偏置读
    RUN_TILE, WAIT_TILE,      // ← 计算
    STORE_OUT, WAIT_STORE,    // ← 写回
    RESPONSE
} state_t;

// 问题1: 没有利用双页缓存
// 问题2: DMA与计算100%串行
// 问题3: 每块都要重复LOAD_A → LOAD_B → LOAD_BIAS流程
//        (权重W不变,只需装一次)
```

**208块执行分析**:

```
当前方式:
块0: LOAD_A(4KB) → LOAD_B(16KB) → LOAD_BIAS(64B) → COMPUTE → STORE(256×INT32)
块1: LOAD_A → LOAD_B → LOAD_BIAS → COMPUTE → STORE
...
块207: ...

问题:
- 每块都装完整的A、B、偏置
- 实际上: 同层的208块共用同一组W和bias!
- 应该只装一次

改进设计:
块0: LOAD_B(16KB) + LOAD_BIAS(64B) → for r=0~12: for c=0~15:
       LOAD_A_TILE(1KB) + COMPUTE
```

**NICE接口规格检查** ✅
```systemverilog
// 32字节描述符格式合理
// [0:3] A地址 (4B)
// [4:7] B地址 (4B)
// [8:11] bias地址 (4B)
// [12:15] 输出地址 (4B)
// [16] k_len(7bit) + relu_en(1bit)
// [17] row_mask(8bit) + col_mask_hi(4bit)
// [18:19] col_mask_lo(12bit) + 保留(4bit)
// [20:31] 保留(12B)

// ✅ 对齐要求清晰(16B对齐)
// ✅ 描述符地址合理(共享SRAM)
// ⚠️ 但没有定义层参数切换机制
//    每块描述符都指定完整参数,容易出错
```

---

### 6. 非线性单元 - 严重缺陷🔴

**GELU悬空**:
```systemverilog
// hardware/rtl/pulse_array/kwt_postprocess.sv
module kwt_postprocess (
    input  [31:0] acc_data,
    input  [31:0] bias_data,
    input  relu_en,
    output [31:0] result  // ← 这里应该是GELU的Q8.8输出!
);
    assign result = (relu_en && biased[31]) ? 32'b0 : biased;
endmodule

// 注释写得很清楚: "GELU 所需的定点缩放尚未定义"
// 这是一个BLOCKER - 无法验证精度!
```

**GELU实现在哪儿?**
```bash
$ grep -r "gelu\|GELU" hardware/rtl/ --include="*.sv"
# 结果: 只有kwt_postprocess.sv中的注释,没有实际代码!
```

**应该有的流程**:
```
INT32累加结果
    ↓
添加bias (INT32)
    ↓
格式转换: INT32 → Q8.8 (定点)
    ├─ 选择合适的scale (例: scale=256代表INT32/256)
    └─ 需要准确的量化规则
    ↓
GELU激活 (需要查找表或近似)
    ├─ 当前软件用: x·sigmoid(1.703125x)
    └─ 需要证明RTL与软件的一致性
    ↓
Q8.8输出 (送到下一层或写回)
```

---

### 7. 软件模型 - 优秀⭐⭐⭐⭐⭐

**优点**:
- ✅ 12层量化实验完整
- ✅ A11W10方案验证充分 (0.18% NRMSE)
- ✅ 支持静态对称量化和逐行仿射
- ✅ 无梯度标定,避免污染测试集
- ✅ CSV报告详细(含尺度、截断率、峰值)

**风险**:
```python
# 软件证明: A11W10 0.18% NRMSE (240条样本)
# RTL验证: ❌ 从未用真实权重跑过!

# 当前RTL testbench:
# - layer0_fp32/case.json 只有2×2矩阵小样本
# - layer0_int16 也是合成数据
# - 没有真实的99×64×64×256矩阵×真实权重
```

**需要补充**:
1. ✅ 生成12层完整的量化权重hex文件
2. ✅ 软件reference输出 (GELU后的Q8.8)
3. ✅ RTL testbench对比验证

---

## 🎯 关键问题清单

### P0 - 阻挡性缺陷 (必须在提交前解决)

| 问题 | 影响 | 优先级 | 工作量 |
|------|------|--------|--------|
| GELU未集成 | 无法证明3%精度要求 | **P0** | 2-3天 |
| 缺少真实权重RTL验证 | 不知道是否功能正确 | **P0** | 5-7天 |
| 没有C驱动 | 无法独立操作硬件 | **P0** | 3-4天 |
| 综合未验证 | 不知道能否150MHz@2.25mm² | **P0** | 1-2周* |

### P1 - 重要缺陷

| 问题 | 影响 | 工作量 |
|------|------|--------|
| DMA/Compute未流水 | 性能损失30-40% | 1周 |
| DMA不支持步长 | 装权重低效 | 3-5天 |
| 系统FSM不支持多块 | 208块调度低效 | 1周 |
| 清零安全性未证明 | CACC可能数据混乱 | 1天 |

### P2 - 改进项

| 问题 | 影响 |
|------|------|
| PE间隔要求未明确 | 维护困难 |
| 地址生成不支持2D步长 | 扩展性差 |
| 量化规则写在多处代码 | 易出错 |

---

## 📊 代码质量指标

### 复杂度分析

```
文件                  行数   FSM状态   风险
kwt_pe.sv            ~500    N/A      低 (流水设计清晰)
kwt_cacc.sv          ~150    2        中 (pending机制有corner case)
kwt_pingpong_cacc_engine.sv ~300    11   高 (单块状态机)
kwt_dma.sv           ~200    8        中 (DMA地址逻辑简单但功能有限)
kwt_tile_system_model.sv    ~150    13   高 (顺序执行所有块)
kwt_nice_system_model.sv    ~150    7    中
───────────────────────────────────────────
总计                3762    41       中(架构清晰,集成复杂)
```

### 测试覆盖

```
✅ PE单元: INT8/16/32 + FP32 四种格式仿真通过
✅ CACC: 部分和转发、清空、排空测试通过
✅ 单块: 384个输出检查通过
✅ E203集成: 首块通过 (4025周期)

❌ 整层208块: 测试脚本存在但需完整运行验证
❌ 12层: 从未测试
❌ 真实权重: 未测试
❌ GELU: 未测试
```

---

## 💡 改进建议

### 短期 (1周内) - 必做

#### 1. 集成GELU (2天)
```systemverilog
// 在 kwt_pingpong_cacc_engine 或新模块中:

module kwt_ffn1_postprocess (
    input  clk, rst_n,
    input  [31:0] cacc_result,
    input  [31:0] bias,
    input  valid_in,
    
    output [15:0] gelu_out,  // Q8.8格式
    output valid_out
);
    // Step 1: 加偏置 (INT32)
    logic [31:0] with_bias = cacc_result + bias;
    
    // Step 2: INT32 → Q8.8转换
    // scale = 256 意思是 Q8.8 = INT32 / 256
    logic [15:0] q88;
    assign q88 = $signed(with_bias >>> 8);  // 右移8位(相当于除256)
    
    // Step 3: GELU lookup table
    kwt_gelu gelu_inst (.in_q88(q88), .out_q88(gelu_out), ...);
endmodule
```

#### 2. 生成12层测试向量 (3天)
```python
# software/generate_rtl_vectors.py

import numpy as np

weights = np.load('weights/kwt1_fp32.npz')

for layer_id in range(12):
    # 提取权重和偏置
    w_fp32 = weights[f'{PREFIX}/enc_layers/{layer_id}/ffn1/kernel']  # 64×256
    b_fp32 = weights[f'{PREFIX}/enc_layers/{layer_id}/ffn1/bias']    # 256
    
    # 量化 (A11W10 静态对称)
    # w_scale = max(|w|) / 511
    # a_scale = max(|a|) / 1023  (逐层标定)
    w_scale = np.max(np.abs(w_fp32)) / 511
    a_scale = np.max(np.abs(layer_input)) / 1023  # 需要软件提供
    
    w_int = quantize(w_fp32 / w_scale)      # INT11, 范围[-1024, 1023]
    b_int = quantize(b_fp32 / (a_scale * w_scale))  # INT32
    
    # 生成hex/mem文件供RTL读取
    np.savetxt(f'hardware/testbenchplus/vectors/layer{layer_id}_W.hex',
               w_int.flatten(), fmt='%03x')
    np.savetxt(f'hardware/testbenchplus/vectors/layer{layer_id}_b.hex',
               b_int, fmt='%08x')
    
    # 生成参考输出 (GELU后的Q8.8)
    ref_output = software_ffn1_gelu(layer_input, w_int, b_int, w_scale, a_scale)
    np.save(f'hardware/testbenchplus/vectors/layer{layer_id}_ref_q88.npy',
            ref_output)
```

#### 3. 12层精度验证 (2天)
修改 `hardware/rtl/control/tb/tb_kwt_full_layer_e203.sv`:
```systemverilog
// 循环12层, 每层检查NRMSE
for (int layer = 0; layer < 12; layer++) begin
    // 装载该层权重到共享SRAM
    load_layer_weights(layer);
    
    // 提交208个块
    for (int tile = 0; tile < 208; tile++) begin
        submit_nice_tile(layer, tile);
        wait_tile_done();
        read_results();
    end
    
    // 计算NRMSE与参考输出
    real nrmse = compute_nrmse(rtl_output, ref_output);
    $display("Layer %0d: NRMSE = %.3f%%", layer, nrmse);
    
    if (nrmse > 3.0) begin
        $error("Layer %0d exceeds 3%% threshold!", layer);
        $finish(1);
    end
end
```

### 中期 (1-2周) - 推荐

#### 4. DMA/Compute流水化
添加块队列管理器:
```systemverilog
module kwt_engine_controller (
    input  [TASK_DESC_W-1:0] task_desc,
    
    // 块调度状态
    output logic [TILE_ID_W-1:0] current_compute_tile,
    output logic [TILE_ID_W-1:0] current_load_tile,
    output logic can_start_load,  // 有空的缓存页
    output logic can_start_compute  // 有数据的缓存页
);
    // 维护两个队列:
    // - load_queue: 待装载的块 ID
    // - compute_queue: 待计算的块 ID
    
    // 流水: DMA可从load_queue装块, COMPUTE从compute_queue算块
    // 约束: load队列领先compute队列最多2个块(双缓存限制)
endmodule
```

#### 5. 改进DMA支持步长 (3天)
针对权重装载，添加步长参数。

#### 6. 驱动软件框架 (2天)
```c
// software/driver/kwt_driver.c

#include "kwt_types.h"

// NICE指令编码
#define KWT_NICE_OPCODE 0x0600C00B

typedef struct {
    uint32_t a_addr;
    uint32_t b_addr;
    uint32_t bias_addr;
    uint32_t out_addr;
    uint8_t k_len;
    uint8_t relu_en;
    uint8_t row_mask;
    uint16_t col_mask;
} kwt_tile_desc_t;

// FFN1计算接口
int kwt_ffn1_layer(
    int layer_id,
    const int16_t *input,   // [99, 64]
    int16_t *output         // [99, 256]
);

// 内部: 块级提交
int kwt_submit_tile(const kwt_tile_desc_t *desc);

// 内部: 轮询完成
int kwt_poll_done();
```

### 长期 (对标最终版本)

7. System Bus / SRAM宏适配 (等平台文件)
8. 150MHz 时序收敛 (1-2周)
9. 功耗分析 (1周)

---

## 🚀 性能分析

### 当前性能 (单块)
```
单块 8×16 矩阵计算 (K=64)
- 周期: 4025 (E203中断到中断)
  - 装载A: ~500
  - 装载B: ~1500
  - 装载偏置: ~100
  - 计算: ~1500
  - 写回: ~300
  - 中断处理: ~125

- 吞吐: 256个输出 / 4025周期 = 0.064 outputs/cycle
- 理想: 8×16 PE每拍最多产生1个乘积, 256拍可完成全部
        实际: 由于装载开销,利用率只有 256/4025 = 6.3%
```

### 优化空间

```
块内优化 (已实现):
- ✅ 乒乓缓存理论上消除装载等待 (但未被利用)
- ✅ CACC并行处理8行,每行16列

块间优化 (未实现):
- ❌ DMA与COMPUTE串行 → 可重叠40%
  预期: 4025周期×208块 = 836k周期
  理想: (块装载 + 块计算) 并行 → ~500k周期

块复用优化 (未实现):
- ❌ 208块每块都装A/B/bias
  实际: 同层share权重, 只需装1次
  可省: 207×(4KB+16KB) = 4.3MB搬运 (已在内存,但总线带宽)
```

### 吞吐计算
```
当前: 208块 × 4025周期 = 836,800周期
      @150MHz ≈ 5.6ms

优化后 (DMA/Compute重叠):
- 假设权重已装入(第0层开销):
- 每块计算: 1500周期
- DMA与下一块计算重叠: -700周期
- 实际每块: 800周期
- 208块总: 208 × 800 = 166,400周期
  @150MHz ≈ 1.1ms

12层总: 12 × 1.1ms = 13.2ms (仅FFN1)
        (原模型完整推理 ~100ms)
```

---

## ✅ 检查清单 (提交前必须)

### 功能验证
- [ ] GELU集成并通过单层测试
- [ ] 12层全部验证NRMSE < 3%
- [ ] E203 × 208块整层通过验证
- [ ] 驱动软件能正常操作硬件

### 综合与时序
- [ ] DC综合通过 (用真实工艺库)
- [ ] 150MHz时序收敛
- [ ] 2.25mm²面积约束检查
- [ ] 关键路径分析完成

### 文档与交付物
- [ ] RTL代码注释完整
- [ ] 硬件设计文档 (包括接口规格)
- [ ] 软件驱动文档
- [ ] 验证报告 (功能、时序、面积、功耗)

### 代码质量
- [ ] 所有testbench通过
- [ ] 无Verilog lint警告
- [ ] 无SystemVerilog规范违反
- [ ] git history清晰

---

## 📝 总结与建议

### 现状评价
你们的项目**已完成70%的核心工作**:
- PE阵列架构 ✅
- CACC创新 ✅
- 系统集成框架 ✅
- 软件量化参考 ✅

但**卡在最后30%的集成与验证**:
- GELU未接 ❌
- 真实权重未验证 ❌
- 驱动软件未写 ❌
- 综合结果未知 ❌

### 优先级建议

**立即开始** (本周):
1. GELU集成 → 单层验证通过
2. 生成12层测试向量
3. E203整层12层验证PASS

**下周**:
4. 平台适配 (等文件 or 并行)
5. 驱动软件
6. DMA/Compute流水

**第三周**:
7. 综合时序
8. 面积功耗分析
9. 文档整理

### 风险管理

```
最大风险: GELU精度
├─ 软件A11W10: 0.18% NRMSE
├─ RTL定点转换: 需精确的scale选择
└─ 如果RTL超3%: 可能需要改用A12W10或逐行仿射

次大风险: 150MHz综合不过
├─ 当前Nangate45试跑未结束
├─ 真实工艺库可能更严格
└─ 备选: 降频到100MHz或优化关键路径

第三风险: SoC平台变更
├─ System Bus实际协议未知
├─ SRAM宏读延迟未知
└─ NICE编码可能不同
```

---

## 📚 参考文档

- 赛题导航: `赛题与资料导航.md`
- 硬件设计指北: `doc/硬件设计指北（定义）.md`
- 硬件设计指南: `doc/硬件设计指南（实现）.md`
- 量化报告: `software/results/0.3百分比位宽与硬件取舍.md`
- NICE接口: `doc/NICE interface.md`
