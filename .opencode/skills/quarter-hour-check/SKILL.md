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

有 systemd 用户服务时加 `--service UNIT.service`；有独立进程时加 `--pid PID`。每 15 分钟的系统定时器只写 `.ai-checks/latest.json`，不会调用模型、自动重试或发送通知。`needs_attention` 表示失联、失败或结果缺失；检查后由任务负责人处理。没有实际后台进程的交互式会话，在空闲时不算失联。

这个技能只负责可见状态；任务的数值校对、文件完整性和验收仍须由对应工作流程验证。
