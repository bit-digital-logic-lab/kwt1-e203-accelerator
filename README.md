# KWT-1 芯片设计竞赛工程

本工程围绕赛题指定的 ARM Keyword Transformer KWT-1 和 E203 接口，开发前馈线性层加速器，并在独立分支探索研究生组的注意力与编码器。**赛题原文以 [`题目/`](题目/) 的截图为准**；[赛题与资料导航](赛题与资料导航.md) 列出模型、权重、数据集来源和校验结果。

## 任务边界

| 组别 | 硬件需要完成 | 计算机端离线完成 |
| --- | --- | --- |
| 本科生组必做 1–3 | 12 套独立参数下的 FFN 第一线性层：`X[99,64] @ W[64,256] + b[256] → Y[99,256]`，E203 调度、数据搬运与综合 | 语音规范化、`98×40` MFCC、token/位置编码、对应层输入 `X[layer]` 的提取 |
| 研究生组必做 4–7 | 单头注意力、含 GELU 与两处 LayerNorm 的单层 PostNorm 编码器、12 层顺序执行、同一配置的综合/周期/功耗分析 | 音频前端与统一软件分类头 |

固定使用题目指定的公开预训练模型与权重。当前没有采用剪枝或改写预训练权重的提交方案。数值标定只使用训练划分，验证划分用于评估；测试划分和组委会隐藏样本不得用于调参数。软件误差、RTL 逐位一致、E203 仿真和官方 SoC/工艺签核是不同证据层级。

## 目录与复现入口

| 路径 | 内容 |
| --- | --- |
| [`题目/`](题目/) | 赛题原图与本科范围解读 |
| [`资料/`](资料/) | 官方模型快照、权重、E203 参考资料、数据集和来源校验 |
| [`software/`](software/) | FP32 参考、`99×64` 输入生成、训练划分标定、A12W10 等量化与驱动原型；见 [软件说明](software/README.md) |
| [`hardware/`](hardware/) | PE/CACC、DMA、NICE-like 控制、RTL 仿真与综合脚本；见 [硬件说明](hardware/README.md) |
| [`hardware/testbenchplus/`](hardware/testbenchplus/) | 独立 golden 向量回归与故障注入；见 [验证说明](hardware/testbenchplus/README.md) |
| [`doc/`](doc/) | 接口和设计笔记 |
| `docs/` | 在整合分支 `codex/kwt-integration` 上的来源报告与集成状态；主分支未合并 |

从工程根目录运行已有的小范围测试（具体工具依赖见各子目录 README）：

```bash
bash hardware/rtl/pulse_array/tb/run_pingpong_cacc_verilator.sh
bash hardware/testbenchplus/run_all_tests.sh
bash hardware/rtl/control/tb/run_e203.sh
```

`run_e203.sh` 默认仅做其脚本定义的简化 E203/NICE-like 测试；真实层/整层参数参见脚本正文。勿将小范围 smoke、示例库综合或软件量化结果写作竞赛验收。

## 当前工程状态（2026-10-19 快照）

- 本科 FFN1 已有 A12W10 软件整数参考、多样本 RTL 测试、12 层独立参数和 E203-like 单样本验证；统计样本数、层数、tile 数须按各报告分别引用。主分支保留未提交的验证与结果文件，打包时按文件系统现状保存。
- 研究生组独立分支已有环境检测、FP32 12 层 golden、单头注意力 RTL 原型和验证设施。非线性/FFN2/LayerNorm 分支正在继续；单层及 12 层硬件集成和统一配置的性能签核尚未完成。分支没有合并到 `main`；可查看 `/home/yang/kwt-worktrees/orchestration/status/grad_dispatch.json` 与各分支 `WORKER_REPORT.md`。
- 组委会专用 SoC、指定工艺库、SRAM 宏、约束和隐藏 600 条评测当前缺失。因此不能声称达到官方 150 MHz、`2.25 mm²`、功耗或隐藏集精度要求。开源 E203 与示例库仅供开发验证。
- 根目录较早的六份“审查”文档把 GELU 写成本科必做，与赛题不符；以原题截图和本 README 的本科/研究生边界为准。

## 数据和打包说明

`资料/数据集/speech_commands_v0.02.tar.gz` 为约 2.3 GB 的官方数据；`software/.venv` 约 1.3 GB，属于本机运行环境，解压到别的机器后应重建。项目完整归档会保留这两者、Git 历史、当前工作树、权重及参考 E203 工程；归档不是可直接提交给组委会的精简包。归档校验结果与成员清单另随归档提供。外部独立 worktree 的**已提交**成果可从本项目 Git 分支恢复；其未提交临时改动不属于主目录归档。
