# WORKER_REPORT — 硬件/量化性能存储取舍（本科 FFN1 第一线性层）

分支：`codex/kwt-architecture`（worktree：`/home/yang/kwt-worktrees/architecture`）

## 目标与范围
- 限定本科必做第一线性层 `Y[99,256]=X[99,64]@W[64,256]+b[256]`；GELU/第二线性层/注意力/12 层编码器不在内。
- 用已有实测（8×16 E203 208-tile、16 行 staged、量化候选）建可运行可校验的流量/容量/周期脚本，比较三种配置。
- 严格区分四层证据：L0 算术理想、L1 RTL 核+测试台、L2 E203 NICE-like SoC、L3 目标库综合。

## 改动（仅新增，未改任何 RTL）
- 新增 `hardware/optimization/ffn1_perf_storage_tradeoff.py`：自检 + 三配置流量/容量/周期/参数装载对照，输出 JSON/Markdown。
- 新增 `hardware/optimization/ffn1_perf_storage_tradeoff.md`：结论、证据层级、未证事项。

## 实际执行的命令与退出结果
```bash
python3 hardware/optimization/ffn1_perf_storage_tradeoff.py \
  --json /tmp/kwt-ffn1-tradeoff.json --markdown /tmp/kwt-ffn1-tradeoff.md
# 退出码 0；自检 PASS：60 次整数 GEMM 精确比较 + 4 项流量恒等式 + INT32 上界

python3 hardware/optimization/run_hook.py ffn1_tradeoff --timeout 60 -- \
  python3 hardware/optimization/ffn1_perf_storage_tradeoff.py \
  --json /tmp/kwt-ffn1-tradeoff.json --markdown /tmp/kwt-ffn1-tradeoff.md
# PASS ffn1_tradeoff rc=0 elapsed=0.183s
# 状态文件 /tmp/kwt-optimization-hooks/ffn1_tradeoff.json：exit_code=0, timed_out=false
```

## 主要结论（均带证据层级）
- 流量：8×16 保留 B 列块把输入从 326,144 降到 **123,392** 字（16-bit 字，L0 计数）。
- 周期：16 行 staged **154,240 拍/层**（L1），比 8 行 **251,328**（L1）少 38.6%；8×16 E203 208-tile 整层 **838,028 拍**（L2，另一边界）。
- 精度：A8W8 480 条最差 **1.121379%**、A12W10 **0.147048%**（软件整数模型），均过 3% 门槛；A12W10 240 条 RTL **0.147394%**。
- 存储：单层操作数 144.375 KiB；12 层 W 容器 384 KiB；12 层 W+A+bias+输出 644.875 KiB **> 512 KiB**，故必须逐层换参。
- 参数装载：每层 **73,920 字**（单字/拍 73,920 拍；理想 128-bit/拍 9,240 拍），须计入任务计时。
- INT32 累加：A12W10 保守上界 67,826,716 < 2^31−1，实测最大绝对值 4,733,248。
- 16-bit 宏：B 列块恰一个 1024 字宏；psum/输出为逻辑 32-bit，需拼宽或寄存器读改写，容量够≠带宽成立。

## 未证事项
1. 无平台吞吐：L1 与 L2 边界不同，不可相减当加速比或当组委会平台耗时。
2. 无 150 MHz / STA / 功耗；L3 面积仅为 Nangate45 同库相对值，不能换算 2.25 mm²。
3. 位宽无实测收益：现有 16-bit 路径下 A8W8/A12W9/A12W10 搬运与拍数相同。
4. 复用方案（123,392 / 23,232 字）未实现，依赖二维 stride、缓存标签与有效尾行写回。
5. 480 条软件与 240/12 条 RTL 是不同统计；官方 600 条隐藏集未知。
6. 208-tile E203 模型只覆盖 8×16；16×16 无对应 SoC 模型周期。
