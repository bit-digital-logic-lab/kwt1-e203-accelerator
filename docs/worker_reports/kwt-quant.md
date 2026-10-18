# WORKER_REPORT（分支 codex/kwt-quant）

## 任务

在题目许可范围内做新的 FFN1 量化对照，只量化 RTL 执行的第一线性层
`X[99,64] @ W[64,256] + b → Y[99,256]`，冻结官方权重，训练划分标定、验证划分只评估，
给出 12 层最差 NRMSE、INT32 溢出边界、权重/激活存储、MAC 位宽与是否需要不同硬件。
未改 RTL，未重跑 12 层 E203 全仿真，未用测试划分，未用梯度标定。

## 改动

- 新增 `software/compare_ffn1_container.py`：单 16 位容器内的位宽对照与 INT32 保守上界地图。
  复用既有 `base_scales`/`encode_params`/`black_box_error`（`calibrate_blackbox_ffn1.py`）和
  `collect_activations`（`quantize_ffn1.py`），指标与既有报告一致；训练最大值的静态尺度，
  验证划分只评估；脚本显式拒绝 training/validation 以外的划分与不交叉的路径约束。
- 新增 `software/results/FFN1单容器精度余量与INT32边界.md`：完整方法、结论、INT32 边界地图、
  需要/不需要不同硬件的论证、证据层级与未证事项。
- 新增结果数据（本分支必要产物）：`ffn1_container_headroom_12labels_40.csv`（逐层）、
  `.json`（元数据 + 逐层 + 上界地图 + 样本/权重 SHA-256）、
  `ffn1_container_headroom_12labels_40_bounds.csv`（INT32 上界地图）。
- 在 `software/results/README.md` 增加一节索引指向上述报告。
- 新增本文件 `WORKER_REPORT.md`。
- 未改任何 RTL、驱动、综合或主目录文件；运行缓存写在 gitignore 的 `software/.build/`，不进入提交。

## 实际执行的命令与退出结果

```bash
# 1) 只读复制主目录既有 480/480 样本缓存到本分支（gitignore）
cp /home/yang/Desktop/ic设计竞赛/software/.build/quant_samples_12labels_40.npz \
   /home/yang/kwt-worktrees/quant/software/.build/
sha256sum software/.build/quant_samples_12labels_40.npz
# -> 43d639fb64ea8d907db967b071010e41e68827368532880ba8a99b92769bbae8（与主目录一致）

# 2) 位宽对照（退出码 0）
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  /home/yang/Desktop/ic设计竞赛/software/.venv/bin/python \
  software/compare_ffn1_container.py --device cuda
```

命令 2 输出（退出码 0）：

```
A11W10 worst=0.181954% layer=11 bound=    33873712 headroom=1.58%  peak=     2453412
A12W10 worst=0.147047% layer=11 bound=    67780536 headroom=3.16%  peak=     4910460
A12W12 worst=0.070279% layer=11 bound=   271520075 headroom=12.64% peak=    19667707
A16W10 worst=0.133404% layer=11 bound=  1084985265 headroom=50.52% peak=    78602736
A14W12 worst=0.036708% layer=11 bound=  1086478227 headroom=50.59% peak=    78700639
PASS wrote .../ffn1_container_headroom_12labels_40.csv and .json
```

480 条训练 / 480 条验证，12 类各 40 条，含 `_unknown_` 与 `_silence_`。
控制点 A11W10 `0.181954%`、A12W10 `0.147047%` 与既有报告逐位吻合，证明对标口径一致。

## 结论（证据层级 L1：本轮软件整数路径实测）

- 三个新方案（每个都在一个 16 位字内）12 层最差层 NRMSE：
  **A12W12 = 0.070279%**、**A16W10 = 0.133404%**、**A14W12 = 0.036708%**，最差层都是第 11 层。
- **权重位宽是瓶颈**：权重 10→12 位使最差层相对降 52.2%；激活 12→16 位只降 9.3%。
- **同样 INT32 余量下权重位更划算**：A16W10 与 A14W12 保守上界几乎相同（约 `1.085×10⁹`，
  占 INT32 的 50.5%），但 A14W12 误差是 A16W10 的 1/3.6。
- **INT32 溢出边界**：保守上界 `64×qmaxA×qmaxW + max|qbias|` 是证明（与样本无关）。
  `qmaxA×qmaxW > ~3.35×10⁷` 即溢出；A12W14/A14W12/A11W15/A16W10 安全，
  A12W15/A14W13/A16W11 已溢出。A14W12 上界 `1.086×10⁹`（50.59%）。
- **存储与乘数**：现有核是 16 位容器 + 16×16 物理乘数、每层搬运 147,840 B、staged 154,240 拍/层。
  在未打包情况下 A12W12/A16W10/A14W12 的容器权重、搬运与拍数都与 A12W10 相同；
  只有实现 W12 紧密打包后 A12W12/A14W12 的 12 层权重才从 240 KiB 增到 288 KiB。
  **结论：现有 16 行/16×16 核上 A14W12 不需要"不同硬件"**，只是把乘数有效位从 12×10 提到 14×12、
  INT32 上界占用从 3.16% 提到 50.59%。
- **梯度标定不采用**：题目规定不重训练/微调/蒸馏，梯度类量化会把尺度并入优化循环，合规边界不清；
  本轮完全不用梯度，连既有离散坐标搜索也未启用，只用训练集最大值的静态尺度。

## 未证事项

- A12W12/A16W10/A14W12 没有 RTL 逐位对照（既有 RTL 逐位证据只覆盖 A8W8/A12W8/A12W9/A11W10/A12W10
  的 12 条验证子集），本轮未重跑 E203 全仿真。
- 未重新综合 12×10 与 14×12 两版核；"接入更多有效位不改变 16×16 核面积/时序/功耗"是基于既有
  RTL 结构的推断，不是本轮实测。
- 未实现 W12 紧密打包/解包；288 KiB 是理论容量。
- INT32 边界地图里 `A12W14`、`A11W15`、`A13W13` 等权重偏重分配点只算了保守上界，未跑误差，
  "权重位更划算"的误差支撑只有 A12W12/A16W10/A14W12 三点。
- 未测官方隐藏 600 条评测、150 MHz 真实 STA、目标 SRAM 宏端口/面积与实测功耗。
