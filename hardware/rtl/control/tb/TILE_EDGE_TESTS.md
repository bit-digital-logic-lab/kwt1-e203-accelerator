# Tile-level edge-case tests

`run_tile_edges.sh` 在 `kwt_tile_system_model`（DMA + 8×16 引擎 + 行为级
128 bit 总线/外部存储）上运行一小组真实 FFN1 数据切块，专门补本科必做验证
缺口，不重跑 208 块/12 层整层回归。

## 运行

```bash
PATH=/home/yang/yosys-sta/oss-cad-suite/bin:$PATH \
  bash hardware/rtl/control/tb/run_tile_edges.sh
```

- 需要能展开冻结 RTL 未打包数组的 Icarus（已在 Icarus 14/oss-cad-suite 实测；
  Ubuntu 自带的 Icarus 11 会报 `unresolved wire array`）。可用
  `IVERILOG=/path/iverilog VVP=/path/vvp` 覆盖。
- 向量默认取主 worktree 的只读目录
  `hardware/architecture/results/real_e203_validation0_vectors/layer01_*`；
  可用 `--vectors DIR` 或 `KWT_VECTOR_ROOT` 指定。脚本把所需算子在临时目录
  写成 ASCII 文件再交给 Icarus，避免中文路径问题。

## 覆盖与判据

| 项目 | 做法 | 判据 |
| --- | --- | --- |
| 独立 golden | 由冻结 `a16/w16/bias32` 用整数点积重算；与 `expected32` 逐项交叉核对并校验 SHA256 | 1 328 个被检查的有效输出全部一致 |
| 末 tile 三行有效/五行填充 | row tile 12（行 96..98）全部 16 个列 tile，`row_mask=0x07` | 768 个有效输出等于 golden；1 280 个填充位置等于 0 |
| 连续任务 | 首个 case 之后无复位连续下发命令 | 19 次连续命令全部返回 status 0 且输出正确 |
| 复位注入 | 在 tile 计算中拉低 `rst_n`，复位后重新下发 | 复位期间无响应；复位后接口恢复并再次核对 golden |
| 负向注入 | 复制 golden 并把最后一个被检查的有效元素 +1 后再跑同一 TB | 必须出现 `MISMATCH` 且退出码非 0；正常 golden 必须 PASS |

## 证据层级

- 这是 **行为级/Icarus tile 级** 证据：外部存储、总线、SRAM 和时序均为简化
  模型，不是组委会 SoC，也不是综合后网表。
- 打印的 `task_cycles/total_cycles` 是 harness 观察值，用于确认任务确实运行，
  **不能**当作 150 MHz 性能或吞吐证据；本套件不做综合、时序、面积或功耗结论。
- 真实 E203 驱动、整层 208 块、12 层换参与 Verilator 真实样本验证由
  `rtl/control/tb/run_e203.sh` 与既有其它脚本覆盖，此套件不重复。
