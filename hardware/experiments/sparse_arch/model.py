#!/usr/bin/env python3
"""Deterministic FFN1 sparse schedule model; no RTL/SoC timing claim."""
import argparse
import json
import random
from math import ceil

M, K, N, ROWS, COLS = 99, 64, 256, 16, 16
ROW_TILES, COL_TILES = ceil(M / ROWS), N // COLS
BASE_COMPUTE = ROW_TILES * K * N
MEASURED_TASK = 154_240
NONCOMPUTE = MEASURED_TASK - BASE_COMPUTE


def simulate(zero_rate, seed):
    rng = random.Random(seed)
    a = [[rng.randint(-5, 5) for _ in range(K)] for _ in range(M)]
    w = [[0 if rng.random() < zero_rate else rng.choice((-3, -2, -1, 1, 2, 3)) for _ in range(N)] for _ in range(K)]
    bias = [rng.randint(-7, 7) for _ in range(N)]
    dense = [[bias[n] + sum(a[m][k] * w[k][n] for k in range(K)) for n in range(N)] for m in range(M)]
    sparse = [bias.copy() for _ in range(M)]
    nonzero = 0
    for k in range(K):
        for n in range(N):
            if w[k][n] == 0:
                continue
            nonzero += 1
            for m in range(M):
                sparse[m][n] += a[m][k] * w[k][n]
    assert dense == sparse
    zero = K * N - nonzero
    # Ideal compressed stream: one cycle per nonzero W entry per row tile.
    ideal_compute = ROW_TILES * nonzero
    # A dense bitmap scanned one bit/cycle offers no cycle gain with one W/cycle.
    # If one bitmap word can skip a full 16-column block, only all-zero blocks save.
    block_skips = sum(all(w[k][c] == 0 for c in range(c0, c0 + COLS))
                      for k in range(K) for c0 in range(0, N, COLS))
    block_compute = BASE_COMPUTE - ROW_TILES * COLS * block_skips
    # Bitmap + nonzero W10 stream; 16-bit offsets for 1025 block boundaries.
    compressed_bytes = ceil(K * N / 8) + ceil(nonzero * 10 / 8) + (K * COL_TILES + 1) * 2
    return {
        "requested_zero_rate": zero_rate, "actual_zero_rate": zero / (K * N),
        "nonzero_weights": nonzero, "zero_weights": zero,
        "independent_dense_sparse_equal": True,
        "ideal_compressed_compute_cycles": ideal_compute,
        "ideal_task_cycles_floor_if_other_phases_unchanged": NONCOMPUTE + ideal_compute,
        "whole_16_block_skip_count": block_skips,
        "whole_16_block_compute_cycles": block_compute,
        "whole_16_block_task_cycles_floor_if_other_phases_unchanged": NONCOMPUTE + block_compute,
        "bitmap_w10_with_block_offsets_bytes": compressed_bytes,
        "dense_w10_bytes": ceil(K * N * 10 / 8),
        "dense_w16_bytes": K * N * 2,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=20261019)
    p.add_argument('--rates', type=float, nargs='+', default=[0, .05, .1, .25, .5, .75])
    p.add_argument('--out')
    args = p.parse_args()
    out = {"shape": [M, K, N], "rows": ROWS, "cols": COLS,
           "baseline_compute_cycles": BASE_COMPUTE,
           "measured_staged_task_cycles_reference": MEASURED_TASK,
           "other_phase_residual_cycles": NONCOMPUTE,
           "cases": [simulate(rate, args.seed + i) for i, rate in enumerate(args.rates)]}
    payload = json.dumps(out, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        with open(args.out, 'w') as f:
            f.write(payload)
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
