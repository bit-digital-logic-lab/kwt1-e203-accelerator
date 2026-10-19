# Independent acceptance check

The frozen A16W12 pack was rerun after making the search/frozen comparison a required gate. Command (from the repository root):

```bash
OPENBLAS_NUM_THREADS=4 python3 graduate/ds_tasks/p027_precision/model/run_frozen.py \
  --pack graduate/ds_tasks/p027_precision/frozen \
  --out /home/yang/kwt-worktrees/orchestration/p027_precision_verified.json
```

Exit 0. `state=PASS`, `comparison_identical=true`, 240 validation examples, 12 layers, worst L11 pooled NRMSE 0.151701%, search/frozen per-layer maximum delta 0. The check now fails if the search result is absent, the selected name is missing, layer counts differ, config hashes differ, or metrics differ.

The search used an initial prefix of validation examples for some diagnostics. With the frozen pack held fixed, `run_search.evaluate` was also run on validation indices 24..239 only (216 examples), using `golden_val_cache` for an FP32 reference; `review_216holdout.json` records the result. Exit 0, worst L11 0.154313%. This is a partial sensitivity check, not an untouched official test set.

These checks validate software integer numerics only. RTL bit exactness for A16W12, target SRAM mapping, 150 MHz timing, and official contest accuracy remain unproven.
