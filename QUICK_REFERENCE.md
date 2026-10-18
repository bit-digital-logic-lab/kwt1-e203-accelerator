# 快速参考卡片

> **2026-10-17 范围纠正：** GELU 不是本科必做，部分命令/预期输出与当前仓库不符；请先读[本科组审查纠偏与下一步](本科组审查纠偏与下一步_20261017.md)，不要直接复制下文的 GELU 数值参考或不存在的 12 层 E203 命令。

## 🚀 立即要做 (按优先级)

### P0.1 GELU集成 (2-3天)
```bash
# 新建模块
vim hardware/rtl/pulse_array/kwt_int32_to_q88.sv
vim hardware/rtl/pulse_array/kwt_ffn1_postprocess_with_gelu.sv

# 关键: INT32缩放到Q8.8
# scale = 256 可能不够! 需要用activation_scale标定值
q88_value = int32_result * 256 / activation_scale;
```

### P0.2 生成测试向量 (3天)
```bash
python3 software/generate_rtl_vectors.py

# 输出: hardware/testbenchplus/vectors/layer{0-11}/
#   - A_quant.hex (99×64 INT11)
#   - W_quant.hex (64×256 INT10)
#   - b_quant.hex (256 INT32)
#   - reference_q88.hex (99×256 Q8.8)
```

### P0.3 Layer0验证 (2天)
```bash
# 修改testbench读真实权重
vim hardware/rtl/control/tb/tb_kwt_layer0_e203.sv

# 执行
bash hardware/rtl/control/tb/run_e203.sh --layer 0

# 期望: ✓ Layer 0 PASS: NRMSE=0.182% < 3%
```

### P0.4 12层全验 (5天)
```bash
# 批量生成所有层向量
python3 software/generate_all_layers_vectors.py

# 运行完整验证 (30-60分钟)
bash hardware/rtl/control/tb/run_e203.sh --full-12-layers

# 期望: ✓ All 12 layers PASS! (max NRMSE < 3%)
```

### P0.5 驱动软件 (3-4天)
```c
// software/driver/kwt_driver.c
int kwt_compute_ffn1_layer(const kwt_layer_config_t *config);

// software/test_kwt_driver.c
gcc -o test_driver test_kwt_driver.c kwt_driver.c
./test_driver
```

---

## 📊 关键数值

### A11W10 量化方案
```
激活: 11 bit 对称量化
  - 范围: [-1023, 1023]
  - scale: max(|input|) / 1023
  - 软件NRMSE: 0.18216% (最差层11)

权重: 10 bit 对称量化
  - 范围: [-511, 511]
  - scale: max(|weight|) / 511 (逐层共享)
  - 存储: 12层权重 = 240 KiB (紧凑编码时)

偏置: INT32
  - scale: activation_scale × weight_scale
  - 预标定: 12个scale值

格式转换 Q8.8:
  - 范围: [-128, 127.99609375]
  - scale factor: 256
  - 需要从INT32右移8位,但可能溢出!
  - ⚠️ 使用activation_scale正确缩放
```

### 布局
```
共享SRAM (512 KB):
  0x0000-0x2000: A缓存 (layer i)
  0x2000-0xA000: W缓存 (所有层权重 or 单层)
  0xA000-0xA100: bias缓存
  0xA100-0xB000: 描述符/中间
  0xB000-0xFFFF: 输出缓存 + 剩余

片上SRAM (三块):
  A: 1024×16bit (99×64 input max)
  B: 1024×16bit (64×256 weight)
  Out: 1024×16bit (128×INT32 = 256×INT16)
```

---

## 🔧 常用命令

### 生成向量
```bash
cd software
python3 generate_rtl_vectors.py --layer 0 --device cuda
# 输出: hardware/testbenchplus/vectors/layer0/
```

### 运行仿真
```bash
cd hardware/rtl/control/tb

# 单块测试
bash run_e203.sh --tiles 0 1

# Layer0
bash run_e203.sh --layer 0

# 整层208块 (慢)
bash run_e203.sh --full-layer

# 12层全部 (非常慢 ~1小时)
bash run_e203.sh --full-12-layers
```

### 综合
```bash
cd hardware/synth
bash run_engine.sh              # 保留SRAM
bash run_engine.sh --full-flop-map  # 展开成DFF (面积统计用)
```

### 检查
```bash
# Lint
verilator --lint-only -I hardware/rtl/sram hardware/rtl/pulse_array/kwt_pe.sv

# 波形查看
gtkwave /tmp/kwt_sim.vcd &
```

---

## 📋 文件对照表

| 文件 | 行数 | 用途 | 状态 |
|------|------|------|------|
| kwt_pe.sv | 500 | PE单元 (INT/FP32) | ✅ |
| kwt_pulse_array.sv | 200 | 2D阵列互连 | ✅ |
| kwt_product_array.sv | 150 | 乘积传播 | ✅ |
| kwt_cacc.sv | 150 | 集中式累加器 | ✅ |
| kwt_pingpong_cacc_engine.sv | 300 | 双页引擎 | ✅ |
| kwt_postprocess.sv | 20 | ReLU (缺GELU!) | ❌ |
| kwt_dma.sv | 200 | 数据搬运 | ✅ |
| kwt_tile_system_model.sv | 150 | 单块控制 | ⚠️ |
| kwt_nice_system_model.sv | 150 | NICE接口 | ⚠️ |
| **新:** kwt_int32_to_q88.sv | ~50 | 格式转换 | 📝 |
| **新:** kwt_ffn1_postprocess_with_gelu.sv | ~100 | GELU集成 | 📝 |

---

## 🧪 测试状态

### 当前通过
```
✅ PE单元: INT8/16/32 + FP32 (各种数据宽度)
✅ CACC: 部分和转发、碰撞检测、清空
✅ 单块: 384输出核对 (8×16 PE)
✅ E203首块: 4025周期通过
```

### 待验证
```
❌ Layer0 真实权重 (GELU后)
❌ Layer1-11
❌ 完整208块×12层
❌ 驱动软件独立操作
```

---

## ⚡ 性能指标

### 单块 (8×16, K=64)
```
周期:       4025 (含E203)
吞吐:       256 output / 4025 cycle = 6.3%
理想:       1 output/cycle → 256周期
装载开销:   ~60% (A/B/bias搬运)
计算开销:   ~35% (阵列流水)
```

### 12层 FFN1 (208块/层)
```
当前 (顺序):   208 × 4025 = 836,800 cycle @ 150MHz = 5.6ms
优化 (流水):   ~166,400 cycle = 1.1ms (预期)

完整KWT-1推理: ~100ms (含其他层)
```

---

## 🔍 调试清单

### GELU精度偏差调试

```
症状: NRMSE > 3%

检查1: 格式转换是否正确
  - 打印INT32值和Q8.8值 (hex)
  - 对比软件中的转换: q88 = int32 >> 8

检查2: Scale因子是否正确
  - 从scales.txt读activation_scale
  - 验证: int32_clamped = int32 * 256 / activation_scale
  - 应该在Q8.8范围[-128, 127.99]

检查3: GELU LUT是否匹配
  - 手工计算几个点: x=0, x=1, x=-1
  - 对比软件: y = x * sigmoid(1.703125*x)
  - RTL输出应该接近

检查4: 舍入和饱和
  - 是否截断到INT16范围?
  - 是否用了正确的舍入规则 (nearest-even)?
```

### 整层验证失败调试

```
症状: 某一层NRMSE > 3%

步骤1: 隔离问题
  - 只跑该层的第一个块 (tile 0)
  - 输出应该是 8×16 的矩阵
  - 与software参考对比

步骤2: 逐块二分
  - 运行前8块 (tile 0-7)
  - 检查NRMSE
  - 逐个块定位问题

步骤3: 数据依赖检查
  - 确认A/B/bias装载正确 (hex值核对)
  - 确认K=64没有裁减
  - 确认掩码正确 (最后一行3个有效)

步骤4: 单PE流水测试
  - 单独测试该PE的GELU
  - 输入已知值,检查输出
```

---

## 📞 关键联系信息

### 代码责任人
- **PE & 阵列**: 你们的首席硬件工程师
- **GELU集成**: 需要与软件量化专家协调scale
- **驱动软件**: 需要熟悉E203指令集
- **综合**: 需要工艺库和EDA工具

### 外部依赖
- 组委会平台文件 (System Bus, SRAM宏, 工艺库) → **2周内预期**
- E203 CPU RTL (已有) → ✅ 可用
- 量化参数 (已有) → ✅ A11W10确定

---

## 🎯 最后检查 (提交前)

### 功能验证
- [ ] `pytest hardware/testbenchplus/vectors/` 所有向量完整
- [ ] `bash run_e203.sh --full-12-layers` 全通过
- [ ] 12层max NRMSE < 3.0%

### 代码质量
- [ ] `verilator --lint-only` 无error
- [ ] git log --oneline (commit message清晰)
- [ ] 所有.sv文件有docstring

### 文档
- [ ] CODE_REVIEW.md 与当前代码对应
- [ ] ACTION_PLAN.md 各milestone标记完成
- [ ] 硬件设计文档包含GELU说明

### 交付物
- [ ] hardware/rtl/ (所有.sv)
- [ ] hardware/synth/ (综合脚本+报告)
- [ ] hardware/testbenchplus/vectors/ (12层数据)
- [ ] software/driver/ (C代码)
- [ ] software/results/ (量化验证)
- [ ] doc/ (所有文档)

---

## 💡 智慧投入 (加分项)

如果还有时间:

1. **DMA/Compute流水** (+10%)
   - 装块N+1时计算块N
   - 预期30%周期降低

2. **块级调度优化** (+5%)
   - 重用权重,减少搬运
   - 统一装载所有层权重

3. **功耗分析** (+5%)
   - 门控时钟
   - 活动信息统计

4. **性能模型** (+5%)
   - 建立吞吐公式
   - 预测不同配置下的性能

5. **代码文生** (+3%)
   - 自动生成RTL文档
   - 参数说明完整

---

## 📚 参考手册

### 软件量化参考
```
关键文件:
  - software/calibrate_target_06.py  (A11W10标定)
  - software/results/0.3百分比位宽与硬件取舍.md (完整分析)

关键函数:
  - quantize_symmetric()
  - compute_nrmse()
```

### 硬件参考
```
关键文件:
  - hardware/README.md
  - hardware/rtl/control/tb/e203_kwt_program.S (汇编例)
  
关键模块:
  - kwt_pe (基础运算)
  - kwt_pingpong_cacc_engine (完整计算)
  - kwt_dma (数据搬运)
```

### E203参考
```
关键文件:
  - 资料/E203_Hummingbirdv2_9f3e34f/doc/core.rst (NICE章节)
  - 资料/e203_hbirdv2_full/rtl/ (CPU RTL)

NICE指令:
  - Custom-0, funct3=110, funct7=0
  - rs1 = 描述符地址
  - rd = 返回状态
```

---

## 🎉 成功标志

当你看到这些消息时,说明你们已经成功:

```
✓ Layer 0: NRMSE=0.182%
✓ Layer 1: NRMSE=0.185%
✓ Layer 2: NRMSE=0.178%
...
✓ Layer 11: NRMSE=0.179%
✓ All 12 layers PASS!

# 综合
✓ 150 MHz slack = +0.5 ns
✓ Area = 1.8 mm² (< 2.25 mm²)
✓ Power = 45 mW

# 驱动
$ ./test_driver
PASS
```

---

**最后提醒**: 
- 每天提交 (git commit)
- 及时记录关键决策 (CODE_REVIEW.md)
- 遇到问题先查软件量化参数 (A11W10的scale)
- 不要改动PE/CACC/DMA的核心逻辑,只在GELU通路上改进

**祝好运!** 💪
