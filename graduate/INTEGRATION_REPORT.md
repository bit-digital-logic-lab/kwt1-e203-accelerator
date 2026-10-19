# 研究生组第 4–7 项集成状态（2026-10-19）

本记录只列已核实的证据与当前阻塞。25 个 OpenCode Go `deepseek-v4.1-flash` 独立任务由 [`fleet/dispatch.py`](fleet/dispatch.py) 运行，状态在 `/home/yang/kwt-worktrees/ds-fleet-20261019/summary.json`，每项有独立分支、日志与终态文件。`submitted` 仅表示 worker 退出码为 0、有新提交、工作树干净且报告改动；不等于独立验收通过。最后一项审查等前 24 项终止后才执行。

## 已核实

- `grad-attention` 的独立 RTL 对照已复跑，41,481/41,481 值一致，单样本输出 NRMSE 0.15886%，运行 2,957,728 拍。该配置目前从验证输入求量化尺度和部分 shift，不能证明冻结配置的跨样本精度。
- `grad-golden` 已与官方 TensorFlow 源码作 3 条验证样本、12 层对照，最大绝对差 9.298e-06；这是 FP32 参考层级。
- `grad-encoder` 已在从提交导出的干净树中独立执行 `bash graduate/encoder/tb/run.sh`（退出 0，22.684 s）和 `run_smoke.sh`（退出 0，172.044 s）。单层 99 行逐级一致，12 层 smoke 只验证调度；第 1–11 层使用第 0 层参数和尺度。`verify_worker.py grad-encoder` 因生成 `meta.json` 的 SHA 不符，整体 `verified_ok=false`。证据在 `/home/yang/kwt-worktrees/orchestration/status/grad-encoder.verify.json`。

## 阻塞项

1. `graduate/encoder/model/attn_int_model.py` 从 `software/validation/.../sample_00_encoder.npy` 读取输入，并对该样本的中间整数张量 `fit_shift`；`kwt_postnorm_ref.py` 的 `calibrate()` 也从相同验证样本的 FP32 中间结果求 `s_a/s_g/s_r1/s_r2/s_out`。应在训练划分冻结数值配置，再于独立验证集评估。
2. `graduate/encoder/model/verify_encoder.py` 第 137–148 行只打印 3% NRMSE，没有在超限时退出失败。现有 `VERIFY_PASS` 只证明逐位模型一致，不能作为精度门禁。
3. `grad-test-infra` 的阶段 runner 仍检查旧的 `hardware/rtl/attention` 和 `hardware/rtl/encoder` 路径，无法正确识别当前 `graduate/rtl/...` 模块。
4. 未提供正式 E203 SoC、目标 PDK/指定 SRAM 宏/SDC，也未有训练集冻结的 12 层 RTL 数值链、官方 600 条隐藏样本和同一配置的 150 MHz/面积/功耗证据。因此现在不能宣称研究生组全部完成或通过官方验收。

## 推送与合并规则

DeepSeek 工作区互不合并。完成后逐一看提交、测试退出码与独立复跑，再将可用成果推送组织仓库的独立分支；`main` 保持不动。新成果和最终验收矩阵待任务终态后更新。
