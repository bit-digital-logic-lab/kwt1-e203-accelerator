# FSM 偏置读取流水优化

## 改动

`rtl/control/kwt_tile_system_model.sv` 的偏置读取仍使用输出 SRAM 的同步单端口读。控制器先发出第 0 字的读请求，下一拍开始每拍锁存前一字并发出下一字请求；最后一字锁存时不再发请求。读取 32 个 16 位字从原先的 64 个状态拍缩为 33 个状态拍，每 tile 少 31 拍。地址、结果和 NICE/DMA 接口未变。

`rtl/control/tb/tb_kwt_tile_system_model.sv` 新增断言：首个有效 tile 必须按 256–287 地址顺序在连续 32 拍内发出偏置读请求，且已有 128 个输出仍逐项比对。

## 当前验证

以下运行均为 2026-10-17 当前工作树结果，完整日志及退出码保存在 `/tmp/kwt-optimization-hooks/fsm_*.log` 和同名 JSON。

| 检查 | 结果 |
| --- | --- |
| `bash hardware/rtl/control/tb/run.sh` | PASS，128 输出、命令错误、DMA 错误、连续偏置读断言 |
| `bash hardware/rtl/control/tb/run_nice.sh` | PASS，描述符读取、背压响应、128 输出及错误路径 |
| `bash hardware/rtl/control/tb/run_e203.sh` | PASS，真实 E203 发出 1 次 NICE 请求和接受 1 次响应，128 输出 |
| `bash hardware/rtl/control/tb/run_e203.sh --tiles 0 4` | PASS，4 tile、512 输出 |
| Verilator `--lint-only`，top=`kwt_tile_system_model` | PASS |

E203 单 tile 的总任务周期受并行修改的 DMA 和阵列实现影响，不能把工作树前后总周期差全部归于本 FSM。31 拍来自本状态机的精确转移数，且连续读请求已由断言覆盖。

## 证明边界

当前使用抽象的 128 位共享总线和仿真 SRAM；没有赛题 System Bus 时序、真实 SRAM 宏、目标 PDK/STA。这里的拍数只针对模型中的偏置读取阶段，不能推断赛题平台频率或整层加速比。描述符获取仍为串行两拍事务；是否允许多个未完成读请求取决于未来平台协议，暂未改动。

## 独立复核命令

```bash
python3 hardware/optimization/run_hook.py --timeout 120 fsm_luna_recheck -- bash hardware/rtl/control/tb/run.sh
```

只需检查 `/tmp/kwt-optimization-hooks/fsm_luna_recheck.json` 的 `exit_code` 和对应日志；无需轮询代理聊天。
