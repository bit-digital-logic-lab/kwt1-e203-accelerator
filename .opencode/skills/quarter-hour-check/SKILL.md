---
name: quarter-hour-check
description: Register and inspect long-running project agent jobs with the shared 15-minute status hook. Use when starting or resuming work expected to exceed 15 minutes, especially tests, simulations, and archive jobs.
---

# 15 分钟检查

项目根目录的 OpenCode 钩子自动登记会话。运行独立进程、后台服务或其他智能体的长任务时，用 `scripts/check.py` 登记；运行中按实际进度更新，结束后标记成功或失败，并填入结果文件路径。不要凭进程存在推断测试通过。

```bash
python3 .opencode/skills/quarter-hour-check/scripts/check.py update \
  --id TASK_ID --state running --result /absolute/path/to/result --note '当前阶段'
python3 .opencode/skills/quarter-hour-check/scripts/check.py update \
  --id TASK_ID --state complete --result /absolute/path/to/result --note '已核验退出码'
python3 .opencode/skills/quarter-hour-check/scripts/check.py scan
```

有 systemd 用户服务时加 `--service UNIT.service`；有独立进程时加 `--pid PID`。每 15 分钟的系统定时器只写 `.ai-checks/latest.json`，不会调用模型、自动重试、发送通知或唤醒任何对话；`needs_attention` 表示失联、失败或结果缺失，检查后由任务负责人处理。

## 状态语义

| state | 含义 | 15 分钟扫描如何判定 |
| --- | --- | --- |
| `running` / `busy` / `retry` | 任务应当仍在运行 | 校验已登记的 service 和/或 pid：service 未运行 → `service_stopped`；pid 不存在 → `process_stopped`；两者都没登记且超过 30 分钟没更新 → `stale_over_30m` |
| `idle` | 交互式会话空闲等待输入 | 不期望有后台进程，永不报失联 |
| `complete` | 正常完成（终态） | 若填了 `--result` 而文件不存在或为空 → `result_missing` |
| `failed` | 失败（终态） | `task_failed` |

要点：

- **只登记 service 的任务（没有 pid）在 service 停止时同样必须报警。** 不能写成“service 和 pid 都失联才报警”：只有 service 的任务 `pid` 为 `None`，按那个规则会漏报 `service_stopped`。
- 已经登记的 service 会在 `is-active` 与预期不符时立即触发；`Type=oneshot` 服务正常跑完后同样是 inactive，因此任务结束必须由 finalizer 主动登记 `complete`，否则会被误判为 `service_stopped`。
- `idle` 只是“没有活着的工作”，不是失败，不要据此报失联。

## finalizer 的异常兜底

长任务用 EXIT trap（`scripts/finalize_hook.sh`）保证无论正常还是异常退出都登记终态，避免实际完成后仍被扫描报成 `service_stopped`：

```bash
source "$repo/.opencode/skills/quarter-hour-check/scripts/finalize_hook.sh"
check_finalize_init "$check_id" "$repo/.opencode/skills/quarter-hour-check/scripts/check.py"
# ... 任务正文 ...
check_finalize_state complete "$report" "12 层完成，报告与汇总已核验"   # 仅成功时调用
```

若正常流程未调用 `check_finalize_state`，trap 会登记 `failed` 并写明异常结束；收到 SIGTERM/SIGINT 也会先转成退出码再登记。登记失败不会改变 finalizer 的退出码。

## 通知与醒目文件

- 每次 `scan` 都会在 `.ai-checks/` 写出 `latest.json`；有需要关注项时额外写出醒目文件 `ATTENTION.md`，全部恢复后自动删除。
- 传 `--notify` 时，仅在某个任务的关注原因**发生变化**时调用一次 `notify-send`（汇总为一条），持续存在的同一问题不会每 15 分钟重复弹窗。
- `notify-send` 不存在或调用失败都不影响 `scan` 的退出码，也不会中断扫描。
- 这些机制只在本地留痕；它们**不会**唤醒 Codex 对话、不会重试、不会修改成果。

## 测试

```bash
python3 -m unittest discover -s .opencode/skills/quarter-hour-check/tests -v
```

测试覆盖正常完成、服务失联、PID 失联、会话 idle、结果缺失、finalizer 异常收尾，以及重复扫描去重；所有状态都写入临时目录，不触碰项目真实的 `.ai-checks`。

这个技能只负责可见状态；任务的数值校对、文件完整性和验收仍须由对应工作流程验证。
