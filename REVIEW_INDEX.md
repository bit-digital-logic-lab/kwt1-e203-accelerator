# 代码审查文档索引

> **2026-10-17 范围纠正：** 本科必做输出是第一前馈线性层 `XW+b`，GELU 是选做；本导航的 P0 排序和 12 层 RTL 状态已经过时。请从[本科组审查纠偏与下一步](本科组审查纠偏与下一步_20261017.md)开始。下文保留原始导航作历史记录。

本次深度审查共生成4份文档，帮助你们快速定位问题、制定计划、执行任务。

---

## 📖 文档导航

### 1. **AUDIT_SUMMARY.md** - 总体审查报告 ⭐ 从这里开始
**适合**: 了解项目整体状况  
**内容**: 
- 65-70%完成度评估
- 8个模块逐一评价 (PE/CACC/DMA/系统/GELU/驱动)
- 风险分析与缓解方案
- 最终打分与建议

**读法**: 按优先级读
1. 执行总结 (5分钟) → 快速了解现状
2. 关键问题排序表 → 明确要做什么
3. 详细技术评价 → 理解为什么
4. 最终建议 → 获得行动指引

**关键输出**: 
```
💡 需要在W1完成的4个P0任务:
   P0.1 GELU集成 (2-3天)
   P0.2 生成12层向量 (3天)
   P0.3 Layer0验证 (2天)
   P0.4 12层全验证 (5天)

⚠️ 最大风险: GELU精度不达3%
```

---

### 2. **CODE_REVIEW.md** - 深度代码审查 ⭐ 技术人员必读
**适合**: 深入了解设计细节和问题  
**内容**:
- 每个模块的优缺点详解 (PE/CACC/DMA/系统/GELU/软件)
- 代码片段与改进方案
- 性能分析
- 测试覆盖情况
- 检查清单

**读法**: 按模块精读
1. 第i个模块的优点 → 学习
2. 第i个模块的问题 → 修正
3. 改进建议 → 动手

**关键输出**:
```
PE (4.5/5): 流水设计优秀,但同步性文档不足
CACC (4.8/5): 创新架构,pending机制需加固
DMA (3.8/5): 基础功能完善,不支持步长
系统 (3.0/5): 功能正确,过度顺序化
GELU (1.0/5): 🔴 BLOCKER - 缺失!
驱动 (0.5/5): 🔴 BLOCKER - 缺失!
```

---

### 3. **ACTION_PLAN.md** - 4周冲刺计划 ⭐ 项目经理必读
**适合**: 制定和跟踪执行计划  
**内容**:
- Week 1-4 的每日任务分解
- 详细的代码实现指导 (伪代码)
- 依赖关系与检查点
- 风险与应急方案
- Daily standup模板

**读法**: 按周次按日次执行
1. Week 1 Monday → 执行P0.1 (GELU集成)
2. Week 1 Wednesday → 执行P0.2 (生成向量)
3. ...按此推进

**关键输出**:
```
Week 1: GELU + Layer0验证 ← 最紧张的一周!
Week 2: 12层全验证 ← 决定成败的一周
Week 3: 平台适配 + 综合
Week 4: 文档打包 + 最终提交

必须在W1 Friday达到: Layer0 NRMSE < 3% ✓
必须在W2 Friday达到: 所有12层验证通过 ✓
```

---

### 4. **QUICK_REFERENCE.md** - 快速参考卡片 ⭐ 开发者日常工具
**适合**: 快速查询命令、数值、接口  
**内容**:
- P0任务立即清单
- 关键数值 (A11W10参数)
- 常用命令 (生成、仿真、综合)
- 文件对照表
- 调试清单
- 成功标志

**读法**: 需要时查询
```bash
# 例: 我需要生成layer0向量
→ 查 QUICK_REFERENCE.md "生成向量" 部分
→ python3 software/generate_rtl_vectors.py --layer 0

# 例: Layer0 NRMSE超过3%,怎么调试?
→ 查 QUICK_REFERENCE.md "调试清单" → GELU精度偏差调试
```

**关键输出**:
```
P0.1 GELU集成 → 实现 INT32→Q8.8 + GELU LUT 集成
P0.2 生成向量 → python3 generate_rtl_vectors.py
P0.3 Layer0验证 → bash run_e203.sh --layer 0
P0.4 12层验证 → bash run_e203.sh --full-12-layers

关键参数: A11W10
  激活: [-1023, 1023]
  权重: [-511, 511]  
  偏置: INT32
  输出: Q8.8 (需要正确的scale转换!)
```

---

## 🎯 使用场景

### 场景1: 项目负责人的早晨例会 (5分钟)
```
1. 打开 AUDIT_SUMMARY.md → 执行总结
2. 看 关键问题排序表 → 今天做什么
3. 打开 ACTION_PLAN.md → Week 1 Day ? 的任务
4. 散会,各自开工
```

### 场景2: 开发者开始一个任务 (10分钟)
```
1. 打开 ACTION_PLAN.md → 当前Day的任务
2. 打开 CODE_REVIEW.md → 该模块的问题分析
3. 打开 QUICK_REFERENCE.md → 关键参数和命令
4. 开始编码
```

### 场景3: 遇到问题需要调试 (查询)
```
1. 问题类型 → GELU精度不达标
2. 查 QUICK_REFERENCE.md → "调试清单" → GELU精度偏差调试
3. 按步骤逐一排查
4. 或查 CODE_REVIEW.md → GELU模块详解
```

### 场景4: 周末回顾周进度 (30分钟)
```
1. 打开 ACTION_PLAN.md → Week N 总结
2. 检查各checkpoint是否PASS
3. 打开 QUICK_REFERENCE.md → 最后检查
4. 更新ACTION_PLAN.md的☑️状态
```

---

## 📚 与其他文档的关系

```
已有文档:
  ├─ 赛题与资料导航.md          (项目背景)
  ├─ hardware/README.md         (当前硬件设计)
  ├─ software/README.md         (当前软件设计)
  ├─ doc/硬件设计指北.md        (设计规范)
  └─ software/results/*.md      (量化分析)

新增审查文档 (本次):
  ├─ AUDIT_SUMMARY.md           ← 整体评价
  ├─ CODE_REVIEW.md             ← 逐模块分析
  ├─ ACTION_PLAN.md             ← 执行计划
  ├─ QUICK_REFERENCE.md         ← 快速查询
  └─ REVIEW_INDEX.md            ← 本文(导航)

建议阅读顺序:
  1. 赛题与资料导航.md (理解背景)
  2. AUDIT_SUMMARY.md (了解现状)
  3. ACTION_PLAN.md (制定计划)
  4. CODE_REVIEW.md (详细学习)
  5. QUICK_REFERENCE.md (开发时查用)
```

---

## ✅ 检查清单 (确保你读了所有重要部分)

### 项目负责人
- [ ] 读 AUDIT_SUMMARY.md 的执行总结 (5分钟)
- [ ] 读 ACTION_PLAN.md 的里程碑表 (5分钟)
- [ ] 确认团队分工 (谁负责GELU,谁负责驱动)
- [ ] 建立每日standup制度

### 硬件工程师
- [ ] 读 CODE_REVIEW.md 的模块评价 (30分钟)
- [ ] 读 ACTION_PLAN.md 的Week 1 Day 1-2 (15分钟)
- [ ] 检查GELU的Q8.8格式定义 (查software/calibrate_target_06.py)
- [ ] 开始写 kwt_int32_to_q88.sv

### 软件工程师
- [ ] 读 AUDIT_SUMMARY.md 的P0问题 (10分钟)
- [ ] 读 ACTION_PLAN.md 的向量生成部分 (20分钟)
- [ ] 读 CODE_REVIEW.md 的软件模型部分 (10分钟)
- [ ] 开始写 generate_rtl_vectors.py

### 验证/测试工程师
- [ ] 读 CODE_REVIEW.md 的测试覆盖部分 (10分钟)
- [ ] 读 ACTION_PLAN.md 的Week 1 Day 4-5 (15分钟)
- [ ] 读 QUICK_REFERENCE.md 的调试清单 (查用)
- [ ] 准备修改E203 testbench

---

## 🚀 快速开始 (今天要做)

### 如果只有30分钟
```
1. 读 AUDIT_SUMMARY.md 的执行总结 → 知道现状
2. 读 ACTION_PLAN.md 的W1任务 → 知道今天做什么
3. 开工!
```

### 如果有2小时
```
1. 读 AUDIT_SUMMARY.md (30分钟)
2. 读 CODE_REVIEW.md 的关键问题 (60分钟)
3. 读 QUICK_REFERENCE.md (10分钟)
4. 开工!
```

### 如果有半天
```
1. 读完 AUDIT_SUMMARY.md (45分钟)
2. 读完 CODE_REVIEW.md (90分钟)
3. 扫过 ACTION_PLAN.md (30分钟)
4. 详读 QUICK_REFERENCE.md (20分钟)
5. 整理笔记,确认理解
6. 开工!
```

---

## 📞 常见问题 (FAQ)

**Q: 这4份文档要全读吗?**
A: 不用。按你的角色:
- 管理层: AUDIT_SUMMARY + ACTION_PLAN
- 开发者: CODE_REVIEW + QUICK_REFERENCE + ACTION_PLAN
- 验证者: CODE_REVIEW 的测试部分 + QUICK_REFERENCE + ACTION_PLAN

**Q: 为什么GELU/驱动这么重要?**
A: 因为
- GELU是本科组的必做题 (不能用ReLU)
- 精度要求是NRMSE < 3% (GELU直接影响)
- 没有驱动无法独立操作硬件
- 这两个缺失会导致评分直接低20-30分

**Q: 这份审查是否等同于最终检收?**
A: 不是。这是一份技术审查, 指出问题和建议。最终还需要:
- 通过综合 (等平台文件)
- 通过时序 (150MHz)
- 所有验证完整

**Q: 如果我们不同意某个建议怎么办?**
A: 可以。但建议至少在GELU和驱动这两个P0问题上 follow 指导。其他P1/P2项目可以根据时间灵活调整。

**Q: 能否压缩时间线?**
A: 理论上可以,但风险很高。当前规划已经很紧张了(12天完成4个P0)。除非你们能调动更多人力并行工作,否则建议按计划执行。

**Q: 代码审查有没有遗漏?**
A: 有。主要是:
- 没有看存储宏的接入方式 (等组委会平台文件)
- 没有详细分析综合结果 (当前无结果)
- 没有完整的功耗分析 (低优先级)
这些都在W3-W4处理。

---

## 📝 文档版本与更新

**本次审查**: 2024年10月  
**有效期**: 至项目提交截止  
**下次更新**: W1 Friday (Layer0验证结果后)

**更新计划**:
- [ ] W1 Friday: 更新GELU部分的具体数值
- [ ] W2 Friday: 更新12层验证结果
- [ ] W3 Friday: 更新综合报告
- [ ] W4 Wednesday: 最终提交前的整合

---

## 🎓 学习价值

这份审查不只是为了"修复bug", 还有学习价值:

**Architecture Design** (从CACC中学):
- 如何取舍局部 vs 集中式设计
- 如何识别和消除瓶颈 (PE本地累加 → CACC集中)
- 如何用握手协议松耦合模块

**Engineering Practice** (从系统FSM中学):
- 如何避免过度顺序化 (当前的教训)
- 如何设计可扩展的控制逻辑
- 如何在时序和面积间平衡

**Verification** (从GELU缺失中学):
- 完整的验证流程为何重要
- 如何保证"软件正确"和"硬件正确"的一致性
- 集成测试的必要性

---

## 👉 下一步

**立即**:
1. 分配这4份文档给对应的人
2. 安排明天上午的kickoff会议
3. 确认GELU的格式定义

**今天**:
1. 至少有一个人把AUDIT_SUMMARY读完
2. 至少有一个人看懂ACTION_PLAN的W1 Day 1

**明天开始**:
1. 按ACTION_PLAN.md 的日程执行
2. 每天提交代码
3. 每天更新checklist

---

**祝你们成功!** 💪

如有任何疑问,可参考:
- CODE_REVIEW.md 的改进建议
- QUICK_REFERENCE.md 的调试清单
- ACTION_PLAN.md 的详细步骤

或回到本文档查看导航。
