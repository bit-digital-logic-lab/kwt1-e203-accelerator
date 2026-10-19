# 研究生组 DeepSeek 任务成果索引

本索引对应整理分支 `codex/kwt-grad-accepted-20261019`。内容来自独立 DS worktree，经主验收筛选后复制；**只合并可复核的成果到本整理分支，不改变原 `grad-encoder` 分支，也不表示研究生第 4–7 项整体通过。** 详细证据见 [`DS_ACCEPTANCE_20261019.md`](DS_ACCEPTANCE_20261019.md)。

## 主验收通过的独立组件

| 内容 | 目录 | 证据边界 |
| --- | --- | --- |
| 题目输入预处理与序列化校对 | `ds_tasks/r2_preprocess/` | 13 条真实 WAV；电脑端 98 帧到 99×64、`.npy` 往返；不含 RTL |
| 12 层 FFN1 整数包 | `ds_tasks/layer_params/` | 训练标定、240 条验证、A12W10；仅 FFN1 软件整数路径 |
| 注意力与 PostNorm 冻结尺度一致性候选 | `ds_tasks/r2_scale_contract/` | 240 条训练共享尺度；240 条验证最差 1.6742%；注意力子模块 3 样本逐位 |
| 12 层软件数值链 | `ds_tasks/r2_full_numeric/` | 240 条 NPZ 验证池化最差 2.7528%；官方 600 条评测未运行 |
| NRMSE 错误门禁 | `ds_tasks/encoder_gate/` | 正例、阈值、位错误和精度错误共 7 项；`verify_encoder.py` 已集成门禁 |
| 跨设备可复现性 | `ds_tasks/meta_determinism/` | 生成物 SHA 从干净树复现；元数据浮点规范化，整数 RTL 参数不变 |
| 张量/字节打包 RTL | `ds_tasks/r2_pack_format/` | Verilator lane 测试与泛型 Yosys；不代表目标工艺签核 |
| 数据集划分/哈希审计 | `ds_tasks/dataset/` | 12 个张量内容哈希匹配；工具检测到验证标定与旧测试缓存违规 |

其他目录保留对照材料或未收口实验，见详细验收报告中的判定；不能仅因目录存在就视作通过。

## 复现

在原工程环境，先准备题目指定的官方模型权重和 Speech Commands V2 数据，再运行对应目录的 `repro.sh` / `run.sh`。脚本当前部分通过主工程只读数据目录读取 gitignored 数据；该整理分支不包含数据集、训练缓存或模型检查点。

此分支的独立组件还没有按共享训练尺度重生组装编码器常数头。尚缺正式 E203 SoC/NICE、目标 PDK/宏/SDC 的 150 MHz 综合签核，以及官方 600 条评测。因此本分支不声明完整加速器通过竞赛验收。
