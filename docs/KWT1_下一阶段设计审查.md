# KWT-1 下一阶段设计审查（本科必做 1–3 项）

> 范围：仅本科必做第 1–3 项——64 点积与数值处理、前馈线性层 `X[99,64]@W[64,256]+b→Y[99,256]`、E203 SoC 集成与综合。GELU、第二线性层、注意力不在本科必做内（题目 `640.jpg 8` 原文：“GELU 和第二线性层列为本科选做内容”）。
>
> 本文件是**下一轮工程拆解**，不是综述。所有判断带文件/行/命令证据；沿用 design-integration 设计决策的证据层级（L0 算术、L1 RTL 核+测试台、L2 E203+NICE-like 仿真、L3 目标库综合、模型推算、正式平台未知）。
>
> 只读核对对象：题目图片 `640.jpg`（封面）、`8/9/11/12/13`、[`题目/README.md`](../题目/README.md)、[`design-integration/docs/KWT1_整体设计决策_20261018.md`](../../design-integration/docs/KWT1_整体设计决策_20261018.md)、9 份 `WORKER_REPORT.md`、主目录 `/home/yang/Desktop/ic设计竞赛` 的 RTL/driver/测试脚本与真实产物。未改任何 RTL/量化/驱动源码，未开子智能体，未 push/merge/rebase。

---

## 0. 一句话结论

**当前可交付的“功能/数值”门槛已经跨过（L1 240 样本 + L2 单样本 12 层逐位），但“必做 3 的 E203 驱动”尚未真正发生**：现用的 E203 通过路径是**测试台层次引用预置描述符 + 16 条手写汇编循环**，C 驱动从未进入 E203 指令流（[`software/driver/README.md`](../../../Desktop/ic设计竞赛/software/driver/README.md) 第 5 行原文：“尚未在 E203 仿真中执行 C 版本”）。下一阶段的最短关键路径 = **接口语义冻结 → C 驱动真执行 → 多样本/复位/连续任务**；综合与官方交付门槛大部分被**外部缺失资源**（PDK/SRAM 宏/SDC/600 条评测集）阻塞，可控的只是“把已能做的事实做扎实、诚实标注”。

---

## 1. 最短关键路径（按功能 / 数值 / 驱动 / 总线 / 综合 / 交付门槛排序）

排序不是按文档章节，而是按**依赖关系 + 是否被外部资源阻塞**：

| 序 | 环节 | 现状（证据层级） | 为什么排在这里 | 完成判据 | 外部阻塞 |
| ---: | --- | --- | --- | --- | --- |
| 0 | **总线/接口语义冻结** | 自定义 “NICE-like” 控制 + 128-bit 仿真总线；官方 NICE 的 `nice_icb` 被置零，共享 RAM 在 co-unit 内部（L2，[`kwt_nice_system_model.sv`](../../../Desktop/ic设计竞赛/hardware/rtl/control/kwt_nice_system_model.sv)、[`e203_kwt_nice_core.sv`](../../../Desktop/ic设计竞赛/hardware/rtl/control/tb/e203_kwt_nice_core.sv) 第 38 行） | 题目要求 System Bus **或** NICE 二选一；接口语义决定驱动怎么写、数据怎么搬、综合把什么当 IO。它是驱动的前置 | 明确选一种并写清地址/握手/完成语义，列出与官方的缺口 | 官方规范缺失 → 只能“冻结本仓库语义”，不能声称合规 |
| 1 | **驱动（必做 3）** | 仅宿主测试 + RV32 交叉编译（L3）；E203 上是汇编循环（见 §3.1） | 必做 3 明确要求“完成驱动、数据交换、启动与完成状态管理，并通过 SoC 仿真验证”；这是唯一“看着像过了、其实没过”的门槛 | C 程序在 E203 指令流中发 NICE、读回状态/输出、12 层全层 PASS | 无（可自证） |
| 2 | **功能（必做 1–2）** | L1 240 样本逐位 + L2 单样本 12 层逐位（真实验产物） | 单样本通过 ≠ 覆盖多样性；需要多样本、复位、连续任务、边界在 SoC 模型上覆盖 | 多样本 L1 矩阵 + L2 缩减调度通过（见 §3.2） | 无（可自证） |
| 3 | **数值（3% 门槛）** | 本地 A12W10 最差层 0.1470%（RTL 逐位），门槛 3%，余量约 20× | 本地门槛已过；官方 600 条口径无法本地复现，不能作为可控关键路径 | 冻结 A12W10 口径；官方口径标记未测 | 官方 600 条不发布 → 不能算官方 NRMSE |
| 4 | **综合/时序/面积** | 仅 Nangate45 示例库相对面积；iSTA abort（exit 134），无时序报告（L3 缺） | 被 PDK/SDC/SRAM 宏阻塞；可控部分是“可综合网表 + 失败即报” | 网表产出 + iSTA 成败明确；不得出 150 MHz 结论 | PDK/SDC/宏缺失 → 不能签核 150 MHz / 2.25 mm² |
| 5 | **交付门槛/可复现** | 证据包在分支上；600 条、SAIF/VCD 功耗、面积签核缺 | 依赖 0–4 + 外部资源；可控部分是工程可复现与证据诚实 | 同一冻结配置下 12 任务端到端记录 + 证据核对脚本通过 | 600 条/宏/功耗条件缺失 → 不计效率/创新分 |

**依赖图**：`0 接口冻结` → `1 驱动` → `2 功能多样本` →（数值冻结贯穿）→ `4 综合（可做部分）` → `5 交付`。**数值不阻塞驱动**，综合/交付被外部资源阻塞。

---

## 2. 值得质疑 / 矛盾的具体地方（≥3，带文件/行/证据）

> 每条都给出可复现证据；同时指出哪些是“文档缺陷”、哪些是“口径不一致”，避免把正常的边界声明误判为错误。

### Q1【文档缺陷】同一份设计报告里 A12W10 保守上界自相矛盾

- 证据：`design-integration/docs/KWT1_整体设计决策_20261018.md`
  - 第 24 行：“A12W10 保守累加上界 **67,780,536**（INT32 的 3.16%）”
  - 第 109 行（§2.4 表）：“A12W10 … **67,826,716** … 240 样本逐位 RTL + 1 样本 E203 全层”
- 交叉核对：
  - E203 向量 `hardware/architecture/results/real_e203_validation0_vectors/manifest.json`：**逐层 12 个 `conservative_abs_accumulator_bound`，最大 = 67,780,536**（`docs/check_next_stage_facts.py` 输出）。
  - 两个值在仓库都能找到：`ffn1_a12w10_static_sym.json` = 67,780,536；`多方案无梯度尺度标定重跑.md` / `ffn1_multi_scale_12labels_20.json` = 67,826,716。差异来自**不同标定批次**（12labels_40 静态 vs 12labels_20/多方案），不是算术错误。
- 影响：数值协议要“冻结”，上界口径必须唯一；否则写报告/答辩时会被追着问。**这是文档缺陷，不是数值错误**（两者都 < 2³¹−1）。

### Q2【口径不一致】当前 E203 12 层状态在两份报告里相反

- `packaging/WORKER_REPORT.md` 第 31 行：“主目录 `real_e203_validation0` 12 层**全 FAIL（0/12）**，疑中文路径 `$readmemh`；本报告记为 FAIL”。
- 实际产物（主目录 `hardware/architecture/results/`）：
  - 旧运行 `real_e203_validation0_finalizer.status` = `SUMMARIZE_EXIT 1`，12 个 `layer*.exit` 全 `1`（这是**被替换的**中文路径运行）。
  - 修正运行 `real_e203_validation0_ascii_finalizer.status` = `SUMMARIZE_EXIT 0`，12 个 `layer*.exit` 全 `0`；每层 `outputs=25344 padding=1280 task_cycles=838028`。
  - `docs/check_design_evidence.py` 退出码 0、`SUMMARY: PASS=72 WARN=0 FAIL=0`。
- 影响：packaging 报告把**过期运行**当成当前状态，没有并置 ASCII 修正运行。design-integration §3 已要求更正。核对时必须以 `*_ascii_*` 为当前状态。**这是口径不一致**（packaging 对旧目录的描述本身为真，但结论会误导）。

### Q3【整合方案缺陷】§5 的整合顺序既有“假冲突”，又漏了真冲突和 synth 分支

- **假冲突**：`KWT1_整体设计决策` 第 224/229/230 行把 `run_e203.sh` 标为“与主目录未提交改动同文件的冲突点”。证据：主目录工作副本 `hardware/rtl/control/tb/run_e203.sh` 与 `git show d94fb6f:hardware/rtl/control/tb/run_e203.sh` **逐字节相同**（sha256 `5fda3e15…`，`check_next_stage_facts.py` 报 FINDING）。即：以 `d94fb6f` 为基线时，主目录那份未提交改动**已被基线包含**，driver 分支只是在其上叠加 ASCII 修复，**不存在文本冲突**。
- **真冲突（被漏掉）**：**9 个分支都在仓库根新增 `WORKER_REPORT.md`，且 blob 互不相同**（`git rev-parse <branch>:WORKER_REPORT.md` 各不同；基线无此文件）。顺序 merge 时第 2 个分支起必产 **add/add 冲突**。§5 表格却对 verification/architecture/packaging 标注“纯新增，无冲突”。
- **漏分支**：第 215 行写“**六个分支**”领先基线，但表格列了 7 行；且**完全没有出现 `codex/kwt-synth`**（它带来了唯一被修正的综合脚本 `hardware/synth/run_ffn1_8x16_sta.sh` 与 `FFN1_8x16_synthesis_status.md`）。若按 §5 整合，会丢掉综合分支。
- 影响：按 §5 直接操作会（a）浪费精力处理不存在的冲突，（b）在 `WORKER_REPORT.md` 上卡住，（c）漏合 synth。见 §5 修正流程。

### Q4【证据归属不精确】§2.4 把 bound 归因给“240 样本 RTL + 1 样本 E203”

- 第 109 行把 A12W10 的 `67,826,716` 与该行的“240 样本逐位 RTL + 1 样本 E203 全层”绑定；但 67,826,716 只出现在**软件标定**产物（480 条训练），E203 manifest 的逐层最大是 67,780,536。该行的行内证据标签与数字来源不匹配。
- 影响：报告的“证据—数字”映射不够严格，评审时容易被抓。**属表述缺陷**。

> 已排除的“疑似矛盾”（避免假阳性）：154,240（16 行，L1 测试台）与 838,028（8×16，L2 E203）**不可相减当加速比**——设计报告 §2.3 已正确标注为“不可比声明”，这是正确做法而非矛盾；`A14W12 上界离 INT32 仅剩 ~3%（实为占用 50.59%）` 属括号内自我更正，语义可辨，不列为缺陷。

---

## 3. 专项评估

### 3.1 如何让 C 驱动真的在 E203 指令流中执行，比“测试台预置描述符”缺什么

**现状（读源码核对）**：

- 测试台在 `initial` 里用层次引用**直接把 208 个描述符写进 shared RAM**：`tb_kwt_real_layer_e203.sv` 第 128–146 行；A/W/bias 也由 `$readmemh` 直接铺进 RAM（第 114–127 行）。
- 运行的程序是 **16 条手写汇编**：`e203_kwt_full_layer.S` 第 10–18 行 = `li a0,0x1000+32*start` → `.insn r 0x0b,6,0,a1,a0,zero` → `bnez` → `addi` → 循环。逻辑与 C 驱动 `kwt_nice_run_ffn1`（`kwt_nice_driver.c` 第 48–67 行）等价，但**不是 C 驱动**。
- `run_e203.sh` 第 24 行把 `program_source` 固定成 `e203_kwt_full_layer.S`；第 6 行 build 目录用 `mktemp -d`（driver 分支已改 ASCII 版本）。

**缺的环节（都可不动核心数值 RTL）**：

1. **链接/入口**：没有 `crt0`/`main`/链接脚本把 `kwt_nice_driver.c` 编进同一镜像。ITCM 只加载 **64 字节**：`tb_kwt_real_layer_e203.sv` 第 16 行 `program_bytes[0:63]`、第 105–110 行只填 word 0..7，**装不下编译后的 C 程序**（ITCM 实际 64 KB，`config.v` `E203_CFG_ITCM_ADDR_WIDTH 16`）。→ 需要把“按镜像大小装载 ITCM”通用化。
2. **CPU 数据通路**：`e203_cpu_instance.svh` 第 41–44 行把 `mem_icb_cmd_ready(1'b0)`、`mem_icb_rsp_valid(1'b0)` 等**全部置零**，`ext2itcm/ext2dtcm` 也置零。共享 RAM 在 co-unit 内部的 `kwt_ext_memory_model`（`e203_kwt_nice_core.sv` 第 57–62 行），**CPU 侧没有任何合法通路能 store 描述符**。→ `kwt_nice_prepare_ffn1` 要真跑，必须为 CPU 给一条到该 RAM 的地址译码通路（例如经 `ext2dtcm`/`nice_icb` 暴露）。
3. **输出/状态回读**：现在只有 TB 用层次引用读 shared RAM 比对（`check_tile` 第 48–49 行）；C 侧只发指令、读 2-bit 状态（`kwt_nice_driver.c` 第 53–62 行）。题目要求“数据交换、启动与完成状态管理”，驱动应能读回 Y 或至少端到端校验。→ 属数据通路同一件事。
4. **指令编码**：C 的 `.insn r 0x0b,6,0,%0,%1,zero` 与 `kwt_nice_system_model.sv` 第 27、97–99 行的 opcode/funct3/funct7 校验**一致**（这一项已具备，不是缺口）。
5. **官方接口**：即使 C 跑起来，仍是本仓库自定义总线；官方 NICE/System Bus 语义**未证**。

**结论**：真正缺的是 **(a) 链接与 ITCM 按镜像大小装载、(b) CPU→共享 RAM 数据通路、(c) 输出回读**。三者都在“新增 TB/入口/译码”范围内，不必改 `kwt_pingpong_cacc_engine`/`kwt_product_array`/`kwt_cacc` 等核心数值 RTL。

### 3.2 如何增加多样本 12 层验证，而避免 12×208×N 的爆炸仿真

**事实**：L2 每层 838,028 拍、单样本 12 层 = 10,056,336 拍；每 tile ~4029 拍，主导项是 16-bit 单口 SRAM 的逐字填充（`kwt_dma.sv` 第 146–154 行 LOAD_FILL 每 128-bit beat 填 8 拍）。实测单样本 12 层墙钟约 **1–2 分钟**（results 文件 mtime：`layer12.log` 14:31:06 → `summary_run.log` 14:31:33）。按此 **600 样本仅 L2 就要 ~10–20 小时**，且每次迭代都要乘。

**分层策略（推荐）**：

1. **数值统计只放 L1**：Verilator 计算核批量跑（现有 `run_quantized_ffn1_batch_rtl.py --per-label`，向量化 case + 一次构建），240 样本已过。**L2 只做集成/控制面抽检**，不重复 N×12×208。
2. **L2 用“层轮转”覆盖**：样本 `i` 只跑 `layer (i mod 12)`，另加 **1 个完整 12 层样本**（已有）。总层运行数 ≈ `N + 12`，而非 `12N`：既覆盖 12 层换参，又覆盖样本多样性。
3. **把 12 层的“控制面等价性”讲清**：layer02–12 与 layer01 描述符链完全同构（`run_e203.sh` 对每层只换 `layer??_*` 向量），差异只在数据；因此“12 层数据正确性”可由 L1 保证，“12 层换参链”由 L2 的 1 个完整样本覆盖即可，无需 12N。
4. **降低 per-tile 拍数（可选、需显式标注）**：把行为 SRAM 换 128-bit 端口模型，可把 ~4029 拍/tile 压到个位数百拍；但这是**改模型边界**，必须标注“不是宏证据”，不能用于综合/面积结论。
5. **并行**：多个 `vvp` 实例按样本并行；向量已离线冻结（`prepare_real_e203_vectors.py`）。

**采样目标应是“控制/换参/边界”，不是“统计精度”。**

### 3.3 16 行核集成应在关键路径前还是后

**后。** 理由：

- 必做 1–2 的功能正确在 **8×16** 上已 L1+L2 可交付；16 行只改善吞吐，而吞吐是评分项、**非准入**（题图 12/13：只有完成全部必做、过必测功能/数值/RTL 对照并满足时序面积上限，才进入效率与创新计分）。
- 16 行目前只在 **Verilator 计算核边界**（`kwt_row_broadcast_tile.sv` + `kwt_tile_output_stage.sv`，`run_row_broadcast.py`）；E203 路径编译的是 `kwt_pingpong_cacc_engine`（默认 ROWS=8，`run_e203.sh` 第 80 行），**两者不是同一硬件边界**。
- 建议：**8×16 作为功能/驱动冻结基线**；16 行作为并行 WP，只有在“同一冻结向量在 E203 上 16 行核 12 层逐位一致”通过后，才作为效率项申报。**不得用 154,240 与 838,028 相减当加速比。**

### 3.4 示例库 iSTA abort 是否应阻塞功能开发

**不应阻塞。** iSTA 在 `link_design` 解析层次化 `assign` 时 `Sta.cc:834 the left port is not exist` → SIGABRT（exit 134），两份网表 `slack/report_timing` 匹配 **0 行**（`synth/WORKER_REPORT.md`）。它只阻塞 **G4（150 MHz/面积签核）**，与功能/驱动/数值正交。

- synth 分支已把脚本改成“**失败即透传非 0 退出**、不无条件解析不存在的时序报告”（`hardware/synth/run_ffn1_8x16_sta.sh` 修正），这是正确做法：防止把 abort 当通过。
- 已额外生成 `*_ics_noexpr.v`（`assign` 数 = 0）但**尚未 STA**——这是一个独立工程风险，应与功能开发并行处理（WP4），**不 gate 功能**。
- 交付状态里必须显式写“无时序结论”。

### 3.5 A14W12 是否值得切换

**不值得作为当前切换目标。**

| 方案 | 最差层 NRMSE | 证据 | 保守上界 / INT32 | 切换代价 |
| --- | ---: | --- | ---: | --- |
| **A12W10（当前）** | 0.1470% | 软件 480 + **RTL 逐位 240** + E203 单样本 | 3.16% | — |
| A14W12 | 0.0367% | **仅软件** | **50.59%** | 需 14×12 有效位乘数、无 RTL 逐位、无综合、需显式溢出/饱和策略 |

- 门槛是 3%，A12W10 余量约 **20×**；切到 A14W12 只把 0.147% 降到 0.037%，对“过门槛”无功能收益。
- A14W12 保守上界占 INT32 **50.59%**，且**只有软件证据**（`quant/WORKER_REPORT.md`、设计报告 §2.4）。切换会同时增加数值风险与硬件验证成本。
- **建议**：冻结 A12W10；A14W12/A12W12/A16W10 仅作“数值—硬件取舍”**对照实验（创新项）**，且必须在同一冻结配置 + RTL 逐位 + 吞吐不变时才可上报。**不得**在无 RTL 逐位的情况下声称已切换。

---

## 4. 工作包拆解（3–5 个，互不冲突、可并行）

> 前提：先做 §5 的整合分支 `codex/kwt-integration`（= `d94fb6f` + 9 个分支）。以下每个 WP **从整合分支拉出**，**只新增自己的文件**，不改其它 WP 或已合入分支的独占文件。凡“冻结不可改”列出的文件，任何 WP 都不得修改。
>
> 时间为**墙钟上限**（基于实测：单层 208 tile 数分钟、12 层单样本 ~1–2 分钟、示例库 coarse 综合 1.98 s），不是“完成天数”。

### WP-A｜C 驱动在 E203 指令流中真实执行（关键路径，必须最先）

- **独占新增**：`hardware/rtl/control/tb/kwt_driver_main.c`、`kwt_driver_crt0.S`、`kwt_driver_link.ld`、`kwt_cpu_shared_path.sv`、`tb_kwt_driver_e203.sv`、`run_e203_driver.sh`
- **冻结不可改**：`tb_kwt_real_layer_e203.sv`、`run_e203.sh`、`kwt_nice_system_model.sv`、`kwt_tile_system_model.sv`、`kwt_pingpong_cacc_engine.sv`、`kwt_product_array.sv`、`kwt_cacc.sv`
- **输入**：整合分支；`software/driver/kwt_nice_driver.c/h`；冻结向量 `hardware/architecture/results/real_e203_validation0_vectors`（只读）
- **产物**：能在 ITCM 里链接运行的 C 程序；CPU 经合法通路访问共享 RAM（store 描述符、读回状态/输出）；12 层全层逐位 PASS
- **验收命令**：
  ```bash
  PATH=/home/yang/yosys-sta/oss-cad-suite/bin:$PATH \
    bash hardware/rtl/control/tb/run_e203_driver.sh \
      --real-layers /home/yang/Desktop/ic设计竞赛/hardware/architecture/results/real_e203_validation0_vectors
  # 期望：退出码 0；12 层 PASS；日志含 C 程序入口/驱动符号执行证据
  riscv64-linux-gnu-objdump -d /tmp/kwt-driver.elf | grep -n "insn"   # 确认 .insn 落在驱动循环
  ```
- **最长运行**：≤ 30 min（编译 + 12 层单样本；单层 208 tile 数分钟）
- **失败停止**：若 CPU 无法在不改核心数值 RTL 的前提下取得共享 RAM 数据通路 → **停止并出报告“驱动受限于数据通路”**；**禁止**用测试台预置描述符冒充“C 驱动通过”。
- **证明边界**：只证明“本仓库 NICE-like 仿真模型 + 8×16 引擎”上 C 驱动执行；**不得**声称官方 NICE/System Bus、综合网表、150 MHz。

### WP-B｜多样本 / 12 层验证矩阵与仿真预算控制

- **独占新增**：`hardware/architecture/run_multisample_ffn1_rtl.py`、`docs/KWT1_验证矩阵.md`
- **冻结不可改**：`run_quantized_ffn1_batch_rtl.py`、`run_row_broadcast.py`、`tb_kwt_row_broadcast_staged_task.sv`、WP-A 的 runner
- **输入**：`software/.build/quant_samples_12labels_*.npz`（只读）；现有 L1 批量脚本；WP-A 的 E203 runner（只读，可用后再接）
- **产物**：N 样本 × 12 层 L1 逐位矩阵 + L2 “层轮转”缩减调度结果与预算表
- **验收命令**：
  ```bash
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
    python3 hardware/architecture/run_multisample_ffn1_rtl.py \
      --scheme A12W10 --per-label 3 --rows 16 --staged --out /tmp/kwt-multisample
  # 期望：退出码 0；逐样本逐层逐位 PASS；汇总 NRMSE < 3%
  ```
- **最长运行**：≤ 20 min
- **失败停止**：出现 **1 处** mismatch 即停止并定位到层/元素；**禁止**“取多数样本”或跳过失败层。
- **证明边界**：多样本 L1 **不等于**官方 600 条；L2 缩减调度**不等于**端到端吞吐/能耗。

### WP-C｜接口冻结与总线 / SRAM 带宽证据

- **独占新增**：`docs/KWT1_接口冻结.md`、`hardware/optimization/kwt_bus_bandwidth_model.py`
- **冻结不可改**：`kwt_dma.sv`、`kwt_dma_2d.sv`、`kwt_nice_system_model.sv`、`kwt_ext_memory_model.sv`
- **输入**：题图（System Bus 或 NICE 二选一）；当前 co-unit/总线/SRAM 行为模型
- **产物**：选定接入方式的理由 + 缺口清单；SRAM 宏（16 bits×128/1024/4096 单/双口）端口/每拍带宽模型，能与 838,028 拍/层做分解比对
- **验收命令**：
  ```bash
  python3 hardware/optimization/kwt_bus_bandwidth_model.py --check \
    --bus-bytes 16 --sram-word-bits 16 --layer-cycles 838028
  # 期望：退出码 0；打印装载/计算/写回分解与恒等式校验
  ```
- **最长运行**：≤ 10 min
- **失败停止**：官方规范/宏数据缺失 → 明确标记“接口语义未证”，**禁止**声称合规。
- **证明边界**：无官方规范/宏数据时只给**相对趋势**，不换算 2.25 mm²、不给 150 MHz。

### WP-D｜综合 / STA 诚实基线与 iSTA abort 隔离

- **独占新增**：`hardware/synth/run_ffn1_full_synth.sh`、`hardware/synth/KWT1_综合诚实清单.md`
- **冻结不可改**：`hardware/synth/run_ffn1_8x16_sta.sh`、`FFN1_8x16_synthesis_status.md`（synth 分支已交付，勿覆盖）
- **输入**：synth 分支网表/脚本；Yosys `0.63+97`、iEDA/iSTA `071226fa`
- **产物**：可综合网表 + “iSTA 是否通过”的布尔状态；失败即打印失败码并不产出“时序通过”字样
- **验收命令**：
  ```bash
  bash hardware/synth/run_ffn1_full_synth.sh --coarse-only      # 期望退出码 0
  bash hardware/synth/run_ffn1_full_synth.sh --sta              # iSTA abort 时必须非 0 退出
  ```
- **最长运行**：coarse ≤ 5 min；完整映射/STA 可能 > 15 min（默认验收只用 `--coarse-only`，长任务按 15 分钟技能登记）
- **失败停止**：iSTA 异常即停止，**不产出任何时序结论**。
- **证明边界**：示例库面积**不得**换算 2.25 mm²；无 150 MHz 结论；本对象只是计算核，不含 E203/总线/控制器。

### WP-E｜交付证据包与文档更正（含整合复核）

- **独占新增**：`docs/KWT1_下一阶段证据核对.md`、`docs/check_next_stage_facts.py`（本审查已附）
- **冻结不可改**：`submission/*`（packaging 分支独占）、`docs/KWT1_整体设计决策_20261018.md`（design-integration 分支独占；只在其上加更正说明或新文件）
- **输入**：主目录 results、各 WP 产物、各分支 `WORKER_REPORT.md`
- **产物**：更正 Q1（A12W10 上界口径唯一化）、Q2（E203 0/12 标记为过期、并置 ASCII 通过）、Q3（整合顺序含 synth、处理 `WORKER_REPORT.md`）
- **验收命令**：
  ```bash
  python3 docs/check_next_stage_facts.py \
    --repo /home/yang/Desktop/ic设计竞赛 \
    --design-doc /home/yang/kwt-worktrees/design-integration/docs/KWT1_整体设计决策_20261018.md
  # 期望：退出码 0；FINDING 与本文 §2 一致
  ```
- **最长运行**：≤ 5 min
- **失败停止**：只核对文件与哈希，**不重跑仿真**；发现结构性缺失（如 `.exit` 非预期）则报 FAIL。
- **证明边界**：核对器只验证“文件/哈希/退出码/分支元数据”，不产生新的性能或精度结论。

**冲突自查**：WP-A/B/C/D/E 的文件集两两不相交；共享只读输入（results、`.build`、现有脚本）全部只读调用；核心数值 RTL 与量化算法不被任何 WP 触碰。

---

## 5. Git 整合顺序（不覆盖主目录）

**现状（实测）**：

- 主目录 `/home/yang/Desktop/ic设计竞赛`：`main` HEAD = `1494597`，工作树有 8 个已改文件 + 多个未跟踪文件。
- `d94fb6f`（`codex/kwt-dispatch-base` / `codex/kwt-next-analysis`）是 **main 工作树的快照**，已包含主目录工作副本的全部内容（`run_e203.sh --real-layers`、`software/driver/*`、`tb_kwt_real_layer_e203.sv`、`prepare_real_e203_vectors.py` …）。
- **9 个已完成分支**（architecture / design-integration / driver / hook-fix / packaging / preprocess / quant / synth / verification）**全部以 `d94fb6f` 为 merge-base**（`check_next_stage_facts.py` 9/9 PASS）。
- **唯一真实冲突面 = `WORKER_REPORT.md`**：9 个分支都在仓库根新增、blob 互不相同（基线无此文件）→ 顺序 merge 必产 add/add 冲突。

**推荐流程（在主仓库执行，但用独立 worktree，不切 main、不动主目录）**：

```bash
cd /home/yang/Desktop/ic设计竞赛
git worktree add /home/yang/kwt-worktrees/integration -b codex/kwt-integration d94fb6f
cd /home/yang/kwt-worktrees/integration

for b in codex/kwt-hook-fix codex/kwt-verification codex/kwt-preprocess \
         codex/kwt-quant codex/kwt-architecture codex/kwt-driver \
         codex/kwt-synth codex/kwt-packaging codex/kwt-design-integration; do
  if git merge --no-commit --no-ff "$b"; then
    :
  fi
  # 冲突只应出现在根 WORKER_REPORT.md：按分支归档，消除 add/add。
  if git ls-files -u -- WORKER_REPORT.md | grep -q .; then
    mkdir -p docs/reports
    target="docs/reports/${b#codex/kwt-}_WORKER_REPORT.md"
    git show "$b:WORKER_REPORT.md" > "$target"
    git add "$target"
    git rm -f --quiet WORKER_REPORT.md 2>/dev/null || true
  fi
  git add -A
  git commit --no-edit -m "Integrate $b"
done
```

> 上面先归档再 `git add -A`，把 9 份报告保留在 `docs/reports/`，同时消除根文件的 add/add 冲突。**不要**用 `-X theirs` 覆盖（会丢其它报告）。

**整合后逐段快速验收（每段退出码 0 才进入下一段）**：

| 序 | 分支 | 验收命令（在整合 worktree 根） | 期望 |
| ---: | --- | --- | --- |
| 1 | hook-fix | `python3 -m unittest discover -s .opencode/skills/quarter-hour-check/tests` | `OK` |
| 2 | verification | `PATH=/home/yang/yosys-sta/oss-cad-suite/bin:$PATH bash hardware/rtl/control/tb/run_tile_edges.sh` | 退出 0，正向 PASS + 负向检出 |
| 3 | preprocess | `/home/yang/miniconda3/bin/python -m unittest software/test_ffn1_input_boundary.py` | `OK` |
| 4 | quant | `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 software/compare_ffn1_container.py --device cpu` | 退出 0，A11W10 0.181954%（控制点） |
| 5 | architecture | `python3 hardware/optimization/ffn1_perf_storage_tradeoff.py --json /tmp/x.json --markdown /tmp/x.md` | 退出 0，自检 PASS |
| 6 | driver | `PATH=/home/yang/yosys-sta/oss-cad-suite/bin:$PATH bash hardware/rtl/control/tb/run_e203.sh --real-layers <ASCII_CASE_ROOT> 0 208` | 退出 0，PASS |
| 7 | synth | `bash hardware/synth/run_ffn1_8x16_sta.sh --coarse-only` | 退出 0，0 problems |
| 8 | packaging | `python3 submission/evidence_check.py --run-fast` | 退出 0（报告需已按 Q2 更正） |
| 9 | design-integration | `python3 docs/check_design_evidence.py --results-dir /home/yang/Desktop/ic设计竞赛/hardware/architecture/results` | 退出 0，`PASS=72 FAIL=0` |

**注意**：

- 主目录的 8 个未提交文件内容与 `d94fb6f` 一致；**不要** `git checkout`/`git stash` 覆盖主目录，也**不要** `git add -A` 到 `main`。整合全程在 `codex/kwt-integration` worktree 内进行。
- 若某分支后来被重写或改了 base，重算 `git merge-base <b> d94fb6f`；非 `d94fb6f` 者先 rebase 到整合分支再合。
- **在整合分支 9 段验收全过之前，不要 push 或 merge 进 `main`**；`main` 的既有未提交改动保留原样，是否 fast-forward 由人工在校验后决定。

---

## 附. 未证事项（禁止夸大）

1. 正式组委会 SoC / System Bus / NICE 规范、PDK、SRAM 宏、SDC **全部缺失**；无 150 MHz、≤2.25 mm²、功耗/能耗的正式签核。
2. **C 驱动未在 E203 仿真中执行**（仅宿主测试 + RV32 交叉编译）。
3. **16 行核未接入 E203**；154,240（L1 测试台）与 838,028（L2 E203）边界不同，**不得相减**。
4. 官方 600 条评测集不发布 → 官方 NRMSE / 分类准确率**未测**；240 RTL、480 软件、1 条 E203、官方 600 是**四个不同统计**。
5. iSTA abort（exit 134）→ **无时序结论**；示例库面积不可换算 2.25 mm²。
6. A14W12/A12W12/A16W10 **只有软件证据**。
7. 本审查的 `docs/check_next_stage_facts.py` 只核对**文件/哈希/退出码/分支元数据**，未重跑任何仿真。
