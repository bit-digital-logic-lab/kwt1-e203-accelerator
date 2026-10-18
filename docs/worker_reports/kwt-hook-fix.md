# WORKER_REPORT: 15 分钟状态钩子可靠性修复

分支：`codex/kwt-hook-fix`
范围：仅 `kwt-worktrees/hook-fix` worktree。未调用子智能体，未修改主目录、systemd 现有配置或其他 worktree。

## 问题与判断

- 12 层仿真实际完成后，旧 finalizer 只写 `*.status`，从不登记 `complete`。任务此前以 `--service` 登记为 `running`，`Type=oneshot` 服务跑完即 inactive，扫描据此报 `service_stopped` 误报。
- Claude 建议“`service is False and pid is False` 才报警”**未采纳**：只登记 service 的任务 `pid` 为 `None`，按该规则会漏报 `service_stopped`。改为保持“service 失联即报警”，并新增测试 `test_service_only_without_pid_still_flags_service_stopped` 锁定该语义。
- 仅文档化状态、加 EXIT trap、异常兜底与去重，不声称能唤醒 Codex 对话。

## 改动文件

| 文件 | 改动 |
| --- | --- |
| `.opencode/skills/quarter-hour-check/scripts/check.py` | 明确状态语义；抽出 `evaluate`；`service_active` 对 systemctl 缺失/异常返回 `None`（未知，不误报）；新增 `--state-dir` 状态目录覆盖；有异常时写醒目文件 `ATTENTION.md`；`--notify` 仅在关注原因变化时汇总弹一条；`notify-send` 缺失/失败不影响退出码 |
| `.opencode/skills/quarter-hour-check/scripts/finalize_hook.sh` | 新增可复用 EXIT trap 助手，保证异常退出也登记终态，保留退出码 |
| `.opencode/skills/quarter-hour-check/tests/test_check.py` | 新增 20 个用例，全部写入临时目录 |
| `.opencode/skills/quarter-hour-check/SKILL.md` | 文档化状态语义、finalizer trap 用法、通知去重与“不唤醒对话”边界 |
| `hardware/rtl/control/tb/finalize_real_e203_validation0_ascii.sh` | 接入 EXIT trap：完成登记 `complete`，超时/summarize 失败/异常退出登记 `failed` |

## 实际运行的命令与退出码

```text
$ python3 -m unittest discover -s .opencode/skills/quarter-hour-check/tests -v
Ran 20 tests in 0.257s
OK                                                            # 退出码 0

$ bash -n hardware/rtl/control/tb/finalize_real_e203_validation0_ascii.sh   # 0
$ bash -n .opencode/skills/quarter-hour-check/scripts/finalize_hook.sh       # 0
$ python3 -m py_compile .opencode/skills/quarter-hour-check/scripts/check.py \
      .opencode/skills/quarter-hour-check/tests/test_check.py                # 0

# 手工端到端（状态目录 /tmp/opencode/qhc-demo2，不触碰项目 .ai-checks）
$ python3 check.py scan --state-dir DEMO --notify     # 首次变化 new=1 → 弹 1 次，退出码 0
$ python3 check.py scan --state-dir DEMO --notify     # 重复 → 不再弹窗，退出码 0
$ python3 check.py scan --state-dir DEMO --notify     # 重复 → 不再弹窗，退出码 0
  notify.log 行数 = 1
$ bash -c 'source finalize_hook.sh; check_finalize_init fx check.py; exit 7'  # 退出码 7，登记 failed
$ bash -c 'source finalize_hook.sh; check_finalize_init fx check.py; \
           check_finalize_state complete RESULT "ok"; exit 0'                 # 0，登记 complete
$ env PATH=<仅含 python3> python3 check.py scan --state-dir DEMO --notify     # 0，无 notify-send 仍写 ATTENTION.md
```

## 测试覆盖（20 用例）

正常完成、服务失联（含 service-only / pid 为 None）、服务存活、PID 失联、PID 存活、会话 idle、结果缺失、failed 状态、超 30 分钟 stale、通知“仅变化触发”去重、恢复后再次失败会再次通知、普通 scan 不消费去重状态、无 notify-send 退出码仍为 0、finalizer 异常退出登记 failed 且保留退出码、SIGTERM 登记 failed 并保留 143、登记失败不改退出码、终态不被 scan 覆盖、finalizer 语法与接线静态校验。所有状态写入 `tempfile` 临时目录。

## 限制

- 未运行真实 12 层 Verilator/E203 仿真（耗时且超出本次范围），仅验证 finalizer 的登记接线、语法与陷阱逻辑；未证明真实仿真通过。
- 未修改现有 systemd `.timer`/`.service` 配置；`quarter-hour-check.service` 仍指向主目录绝对路径，且未加 `--notify`，因此默认只写 `latest.json`/`ATTENTION.md`，不弹窗（需要弹窗须自行在单元里加 `--notify`，本次未改系统配置）。
- 未修改非 ASCII finalizer `finalize_real_e203_validation0.sh`（不在指定范围），其仍缺 EXIT trap，存在同类风险。
- `systemctl` 不可用时 `service_active` 返回未知（`None`），既不误报也不会确认存活。
- 本环境无 `shellcheck`（退出码 127），改用 `bash -n` 校验。
- 本机制只在本地留痕，**不会**唤醒 Codex 对话、不重试、不修改成果。
- 测试与手工演示均使用临时状态目录，主目录 `Desktop/ic设计竞赛/.ai-checks` 未被读写。
