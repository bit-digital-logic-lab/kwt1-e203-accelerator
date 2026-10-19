# r2_full_numeric — provenance

本任务只消费第一轮冻结资产，不重新标定。所有只读引用如下（本任务目录内没有复制任何数据数组）。

## 只读冻结资产

| 资产 | 来源 | SHA-256 |
| --- | --- | --- |
| `frozen_12layer_calib.json` | `ds-grad-encoder_12numeric @ ea7c07b`, `graduate/ds_tasks/encoder_12numeric/` | `5f039ec745e2b4dda1e6de5abb6fdf97316c3e98a0265c31212b53eee669d444` |
| `frozen_12layer_int.npz` | 同上 | `5cf836e0e5108d9a20d7d817ef24423d9cd9f196dd353872f0de5e82da317e5b` |
| `attn_calib_train.json` | `ds-grad-attn_scale`, `graduate/ds_tasks/attn_scale/` | `9a535f04b99f0cebce941e88fc0ef8d6fcc9e30f29656d32433dff8d07fa267f` |
| `frozen_postnorm_calib.json` | `ds-grad-postnorm_scale`, `graduate/ds_tasks/postnorm_scale/` | `3a1cf51b2f00716b0dc6a182d80e6d4cdf5886fa196affad268a47167bbf53fb` |
| `quant_samples_12labels_20.npz` | 主工程 `/home/yang/Desktop/ic设计竞赛/software/.build/`（gitignored，只读） | `ee04e849bc407149618e4ecd5ff99575687586d4e57c0ade6a0b3f7b66c8c149` |
| `kwt1_fp32.npz` | 本 worktree `software/weights/` | `8f359abf7adbe287b8f89fca98af46194f98f90f2146784731bff551a155825a` |

## 只读代码复制

`model/chain.py`、`model/ref12.py` 逐字节复制自第一轮
`ds-grad-encoder_12numeric @ ea7c07b` 的 `graduate/ds_tasks/encoder_12numeric/model/`；
本任务只放同深度目录，`WORKTREE = HERE.parents[3]` 仍解析到本 worktree 根。源哈希：

| 文件 | 源 SHA-256 | 本目录 SHA-256 |
| --- | --- | --- |
| `chain.py` | `850080323588a92be143a87c75c6c4c477ac1ad8792696656c31f8c863fa62c5` | 同 |
| `ref12.py` | `383a95055da924f20d8ddd3a52fffbaa092b0a92b170179040389270c52b8111` | 同 |

`chain.py` 依赖的 FP32 golden `graduate/encoder/reference/kwt1_fp32_golden.py` 来自本 worktree
（`codex/kwt-grad-encoder @ 7b6abee` 基线）。

## 本任务新增

`model/run_full_numeric.py`、`model/run_conflict_witness.py`、`model/check_rtl_config.py`、
`repro.sh`、`results.json`、`conflict.json`、`rtl_config.json`、`logs/*.console.log`、`logs/exit_codes.txt`。

验证划分只用于评估，未参与任何尺度/移位/LUT/阈值选择；本任务未做训练划分标定。
