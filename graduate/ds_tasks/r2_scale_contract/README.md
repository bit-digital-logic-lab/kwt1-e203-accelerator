# r2_scale_contract — attn_scale × postnorm_scale 冻结契约交叉核查

第二轮任务。目标：核查第一轮两项成果（`ds-grad-attn_scale`、`ds-grad-postnorm_scale`）的
**训练样本、输入尺度 `sx`/`scale_x`、量化边界与 RTL 常数**能否合成**一套**冻结配置。
两项第一轮 worktree 全程**只读**。

## 结论（一句话）

**不能原样合成**：两项对**同一个 encoder 输入张量 X** 冻结了**两个不同的量化尺度**
（`attn.scale_x = 1.961475e-3` vs `postnorm.s_x = 6.737881e-3`，相差 3.435×），
而组装 RTL 让注意力量化后的 X（`x_buf`）直接进入 PostNorm 残差、再用 `postnorm.s_x`
推出的 `MA1` 加权，两者必须相等才自洽；朴素合并的残差 `r1` NRMSE 达 **247%**（门槛 3%）。
**可以合成**：在**共享训练划分**上重新冻结**一个**输入尺度（两项的训练样本同属官方
split，attn 的 3 条是 postnorm 240 条的子集），用该尺度在 240 条训练样本上重拟合注意力
常数、保留 postnorm 常数、重算 `M_A`，则整链在**验证划分**（240+3 条）最差 NRMSE
**1.67% < 3%**，且注意力 RTL 逐位一致（`41481/41481`）。候选冻结包见 `candidate/`。

## 一、训练样本与尺度冲突

| 项 | 第一轮 attn_scale | 第一轮 postnorm_scale |
| --- | --- | --- |
| 标定文件 | `attn_calib_train.json`（sha `9a535f04…`） | `frozen_postnorm_calib.json`（sha `3a1cf51b…`） |
| 标定划分 | training | training |
| 训练样本数 | **3**（`real_input_chain` `sample_01/03/04`） | **240**（`quant_samples_12labels_20.npz` training） |
| 验证样本数 | 3（`sample_00/02/05`） | 240（npz validation）+ 3 real |
| 输入尺度 | `scale_x = 1.961475e-3`（max\|X\|=4.015） | `s_x = 6.737881e-3`（max\|X\|=13.792） |
| 权重 sha256 | `8f359abf…` | `8f359abf…`（同一 checkpoint） |

- 两批样本属**同一官方 split**：attn 训练 3 条 wav（`go/0ba018fc…`、`no/7e7ca854…`、
  `yes/676f8138…`）全部属于 npz training；attn 验证 3 条属于 npz validation；
  npz train/val 互斥。跨源 X 一致（npz 投影 vs `real_input_chain` npy，max\|Δ\|=5.8e-6）。
- 因此**同一张量 X 被冻结了两个尺度**，且 attn 只有 3 条样本，本身就是**子集尺度**。

## 二、为什么必须 `scale_x == s_x`（RTL 结构原因）

`graduate/rtl/encoder/kwt_encoder_layer.sv`：

```text
L_ROW_X : x_buf[d]  <= attn_rd_data   // 读注意力内部已按 scale_x 量化的 X
L_ROW_O : o_buf[d]  <= attn_rd_data   // 读注意力整数输出（scale_o）
a       = (o*M_A + 2^15) >>> 16       // M_A = round(scale_o/s_a*2^16)
r1      = (x_buf*MA1 + a*MB1) >>> 16  // MA1 由 postnorm s_x 推出
```

`x_buf` 的唯一来源是注意力用 `scale_x` 量化后的 X；而 `MA1 = round(s_x/s_r1*2^16)`
用的是 postnorm 的 `s_x`。二者不相等时残差 x 项被整体缩放 `s_x/scale_x = 3.435`。

朴素的“把两个第一轮结果拼在一起”在三条真实验证样本上的残差 NRMSE：

| 验证样本 | 失配 r1 NRMSE |
| --- | ---: |
| `sample_00` | 247.19% |
| `sample_02` | 243.70% |
| `sample_05` | 245.76% |

→ **远超过 3% 门槛，INCOMPATIBLE**。

## 三、量化边界不一致

- attn：`ACT_BITS=12` 但 `quantize_activation` 只夹到 **int16 容器 ±32767**，
  `|Xi|>2047` 合法（其 v1 报告已记录 sample_05 有 2 个 `|Xi|=2095>2047`）。
- postnorm：`CLAMP_A=2047` 在 `quantize_stimulus/residual/LN/GELU` **硬夹**。
- RTL：`o→a` requant 饱和在 int16 ±32768，**不是** 2047。
- 若把 attn 3 样本尺度套到 240 条训练集：**57/240** 条样本出现 `|Xi|>2047`（共 58011 个元素），
  与 postnorm 的 ±2047 边界不兼容。改用共享训练尺度后 **0/240** 越界。

## 四、RTL 常数与冻结模型不一致

| 常数 | grad-encoder 基线头 | 第一轮冻结模型 | 一致 |
| --- | --- | --- | :---: |
| attn `KC` | 6680 | 6986 | ✗ |
| attn `SH_KC` | 38 | 37 | ✗ |
| attn `RQ/RK/RV/RC/RO` | 6/7/7/14/11 | 6/7/7/14/11 | ✓ |
| postnorm `MA1,MB1,MGN,MO1,MC2,MO2,EPS_Q1,EPS_Q2` | 旧逐样本 | 训练冻结 | 8 项仅 `MO1` 1 项一致 |
| `tb_encoder.sv M_A` | 5981 | —— | 旧值 |

headline：注意力头只差 `KC/SH_KC` 两项；postnorm 头几乎全错；`M_A` 仍是旧值。
以上对比由 L1 驱动的 `[5]` 段直接打印（`logs/cross_check_console.log`、
`results.json → rtl_constants`）：postnorm 8 项中仅 `MO1=410` 一致，其余 7 项与训练冻结模型
逐项不符（如 `MA1` 61302 vs 65960、`MGN` 267644 vs 129003）。
另注 `gen_vectors.py` 读 `cal["scales"]["sx"]`，而该键实为 **FFN1 包的激活尺度**
（`layer01_activation_scale≈2.437e-3`），不是 encoder 输入尺度——集成时的命名陷阱；
`[5]` 段同样打印该键（`0.00243741778`）与 encoder 输入尺度（`0.00673788148`）的差别。

## 五、可行的一套冻结配置（候选，仅 L1 + attn L2）

在**共享训练划分**（240 条）上：

- 输入尺度唯一化：`s_x = max_train|X| / 2047 = 13.7924434/2047 = 6.737881e-3`
  （== postnorm 的 `s_x`；attn 的 3 条训练样本 max 4.015 是其子集）。
- 注意力常数在 240 条训练样本上**无梯度重拟合**（只做 max 统计与整数取整）：
  `RQ=5 RK=7 RV=6 RC=15 RO=11 KC=5152 SH_KC=34 scale_o=1.229728e-4`。
- postnorm 冻结尺度不变（`s_a,s_g,s_r1,s_r2,s_out`）。
- `M_A = round(scale_o/s_a*2^16) = 5247`（旧 RTL 为 5981）。

验证划分（**仅评估**）结果：

| 指标 | 值 |
| --- | ---: |
| 硬件忠实链（含 o→a requant）最差 NRMSE | **1.6742%** |
| 软件参考链最差 NRMSE | 1.6898% |
| 平均分级 r1/g/r2/y | 0.3648% / 1.1723% / 0.8694% / 0.7828% |
| 验证样本 | 240（npz）+ 3（real） |
| max\|X_int\| | 2047（= 边界） |
| 注意力 RTL 逐位 | 3× `41481/41481`，NRMSE 0.1923%/0.2497%/0.2171% |

候选产物：`candidate/attn_params.svh`、`candidate/coherent_frozen_calib.json`。
**它们只是提案**：尚未写入 `third_party/`、postnorm 常数头与整链 TB 尚未按此重生。

## 六、证据层级

| 层级 | 状态 | 依据 |
| --- | --- | --- |
| L1 软件参考 | **已证** | `run_cross_check.py` → `results.json`、`logs/cross_check*.log`（7/7 gate PASS；`[5]` 段含 attn/postnorm 常数逐项对比） |
| L2 RTL 逐位（注意力） | **已证** | `rtl_check_coherent.sh` → 3 样本 41481/41481，`logs/rtl_coh_*.log` |
| L2 RTL 逐位（整链/12 层） | **未证** | postnorm 常数头 + 整链 TB 未按候选重生 |
| E203 集成 | **未证** | 无官方 SoC/NICE/总线 |
| 综合/时序/面积 | **未证** | 未做 PDK 综合、150 MHz STA |
| 官方 600 条验收 | **未证** | 未跑官方评测脚本 |

## 七、文件

| 文件 | 作用 |
| --- | --- |
| `run_cross_check.py` | L1 交叉核查与候选冻结包生成（默认 `--max-val 0` 用全部 240 验证样本） |
| `gen_coherent_vectors.py` | 用候选常数生成注意力 RTL 激励/参数（只写 build 目录） |
| `rtl_check_coherent.sh` | L2：Verilator 注意力逐位（复用第一轮 RTL+TB，只读） |
| `repro.sh` | 一键复现（L1 + L2，记录退出码） |
| `candidate/` | 候选冻结包（`attn_params.svh`、`coherent_frozen_calib.json`），仅为提案 |
| `logs/` | 全部控制台日志与退出码 |
| `results.json` | 机器可读汇总 |

## 八、复现

```bash
bash graduate/ds_tasks/r2_scale_contract/repro.sh
```

构建目录固定在 `/tmp/opencode/r2_scale_contract/`；不写任何其他 worktree。
数据来自主工程 gitignored `software/.build/quant_samples_12labels_20.npz`（只读），
未提交大数据/密钥/缓存。
