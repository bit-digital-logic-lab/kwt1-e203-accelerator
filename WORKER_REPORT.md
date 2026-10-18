# WORKER_REPORT — codex/kwt-grad-attention（研究生必做第 4 项：单头注意力独立 RTL 原型）

- 分支 / worktree：`codex/kwt-grad-attention` @ `/home/yang/kwt-worktrees/grad-attention`
- 范围：题目[`640.jpg 9`](题目/640.jpg%209) 第 4 项**单头注意力**独立可综合 RTL 原型：
  `Q/K/V 64→64` 投影、`99×99` 的 `QKᵀ/√64`、逐行 Softmax、概率加权 V、输出投影。
- 新增（仅此目录）：`graduate/rtl/attention/{README.md, rtl/, model/, tb/, vectors/}`。
  **未改动现有核心 RTL / 主目录 / 他人分支**；未 `merge`/`rebase`/`cherry-pick`/`push`；未开子智能体。
- 真实数据：官方权重 `software/weights/kwt1_fp32.npz`（`enc_layers/0/att/*`）+
  真实输入 `software/validation/real_input_chain_20261017/sample_00_encoder.npy`（99×64，
  第 0 层注意力输入）。张量顺序核对自官方 `transformer_utils.py`。

## 运行命令与退出码（本次实跑）

| 命令（在 `graduate/rtl/attention/` 下） | 退出码 | 结果 |
| --- | ---: | --- |
| `python3 model/attn_int_model.py` | **0** | 生成 `rtl/attn_params.svh`、`vectors/*` |
| `verilator --cc --exe --build ... tb_attn.cpp` | **0** | 仅警告（WIDTH*），无错误 |
| `./obj_dir/tb_attn vectors` | **0** | `41481/41481` 逐位一致；NRMSE 0.15886% |
| `yosys 0.63 read_verilog -sv -Irtl rtl/attn_single_head.sv; hierarchy -check; proc; check` | **0** | 前端可综合性检查通过 |

一键复现：`bash graduate/rtl/attention/tb/run.sh`。

## 关键结果

- **逐位对照独立整数模型**：`q,k,v,c,o` 各 6336/6336，`p`（Softmax，全 99×99）9801/9801，
  合计 **41481/41481，0 失配**。乘法/Softmax 全在 RTL，TB 只做装载、读回与比较。
- **FP32 误差（单列）**：NRMSE 0.15886%（< 3% 门槛）；首行 0.15766%、末行 0.15077%；
  max abs err 0.00124016。
- **周期**：实测 `2,957,728` 拍（串行 P=1，每拍 1 次乘加）；150 MHz 推算 ≈19.7 ms（模型推算，非平台实测）。
- **存储**：≈128.7 KB（int16 容器，含验证用 `P`；去掉 `P` 可省 19.6 KB）。

## 未证事项（不得宣称整项通过）

1. 未接入 E203 / NICE / System Bus，无 SoC 集成、驱动、系统级复位/连续任务覆盖。
2. 未做目标 PDK 综合与 150 MHz STA；组合除法器/64b 乘法器/组合读端口未做时序优化，
   面积与功耗未测。
3. 大型数组与组合读端口未映射到题目规定的 SRAM 宏。
4. 仅 1 个真实样本（第 0 层 attention）；未做 12 层、多样本、量化标定泛化。
5. Softmax/量化误差仅本样本统计；换层/换样本需重新标定并重跑。

## 边界声明

- 只在本 worktree 内操作；`git status` 仅新增 `graduate/` 与改本文件。
- 未修改 `hardware/`、`software/`、`docs/` 等既有核心内容。
- 一切结论基于本次真实运行的退出码与日志，未凭进程存在推断通过。
