#!/usr/bin/env python3
"""Small FFN1 tile correctness and 16-bit traffic experiment; no timing model."""

import argparse
import json
import random
from math import ceil
from pathlib import Path


STRATEGIES = ("reload", "row_keep_a", "col_keep_b", "full_a_keep_b")


def tile_indices(m: int, n: int, rows: int, cols: int, strategy: str):
    mr, nc = ceil(m / rows), ceil(n / cols)
    if strategy == "row_keep_a":
        return [(i, j) for i in range(mr) for j in range(nc)]
    return [(i, j) for j in range(nc) for i in range(mr)]


def count(m: int, k: int, n: int, rows: int, cols: int,
          strategy: str, clipped: bool = False, k_block: int = 0,
          spill_psum: bool = False) -> dict:
    """Count external transfers; A/B/bias caches are logical, not RTL claims."""
    if min(m, k, n, rows, cols) <= 0 or strategy not in STRATEGIES:
        raise ValueError("positive dimensions and a known strategy required")
    if k_block < 0:
        raise ValueError("k_block must be nonnegative")
    chunks = ceil(k / k_block) if k_block else 1
    traffic = {x: 0 for x in ("a_read", "b_read", "bias_read", "psum_read",
                             "psum_write", "output_write")}
    row_tiles, col_tiles = ceil(m / rows), ceil(n / cols)
    a_loaded = b_loaded = bias_loaded = None
    for i, j in tile_indices(m, n, rows, cols, strategy):
        vr, vc = min(rows, m - i * rows), min(cols, n - j * cols)
        a_tag = "full" if strategy == "full_a_keep_b" else i
        b_tag = j
        need_a = strategy == "reload" or a_tag != a_loaded
        need_b = strategy in ("reload", "row_keep_a") or b_tag != b_loaded
        need_bias = strategy in ("reload", "row_keep_a") or b_tag != bias_loaded
        if need_a:
            traffic["a_read"] += (m if a_tag == "full" else
                                  (vr if clipped else rows)) * k
            a_loaded = a_tag
        if need_b:
            traffic["b_read"] += (vc if clipped else cols) * k
            b_loaded = b_tag
        if need_bias:
            traffic["bias_read"] += (vc if clipped else cols) * 2
            bias_loaded = b_tag
        outputs = vr * vc if clipped else rows * cols
        traffic["output_write"] += 2 * outputs
        if spill_psum:
            traffic["psum_read"] += 2 * outputs * (chunks - 1)
            traffic["psum_write"] += 2 * outputs * (chunks - 1)
    cache_a = m * k if strategy == "full_a_keep_b" else rows * k
    cache_b = cols * k
    psum = 2 * rows * cols
    output = 2 * rows * cols
    return {
        "shape": [m, k, n], "tile": [rows, cols], "strategy": strategy,
        "clipped_edges": clipped, "k_block": k_block or k,
        "spill_psum": spill_psum, "tile_count": row_tiles * col_tiles,
        "traffic_16bit_words": traffic,
        "input_read_words": sum(traffic[x] for x in ("a_read", "b_read", "bias_read")),
        "total_external_words": sum(traffic.values()),
        "onchip_logical_16bit_words": {
            "a": cache_a, "b": cache_b, "bias": 2 * cols,
            "psum": psum, "output": output,
        },
        "capacity_1024word_macros_ceil_by_bank": {
            "a": ceil(cache_a / 1024), "b": ceil(cache_b / 1024),
            "bias": ceil(2 * cols / 1024),
            "psum": ceil(psum / 1024), "output": ceil(output / 1024),
        },
        "k_step_port_demand": {
            "vector_launch_a_reads": rows,
            "vector_launch_b_reads": cols,
            "scalar_broadcast_a_prefetch_reads": 1,
            "scalar_broadcast_b_reads": 1,
            "scalar_broadcast_psum_read_write_lanes": rows,
        },
        "useful_mac": m * k * n,
        "padded_mac": row_tiles * col_tiles * rows * cols * k,
        "padded_output_elements": row_tiles * col_tiles * rows * cols - m * n,
    }


def golden(a, b, bias):
    return [[sum(a[i][z] * b[z][j] for z in range(len(b))) + bias[j]
             for j in range(len(b[0]))] for i in range(len(a))]


def tiled(a, b, bias, rows, cols, k_block, strategy):
    m, k, n = len(a), len(b), len(b[0])
    out = [[0] * n for _ in range(m)]
    # Materialize source tiles from ordinary row-major matrices. Cache tags
    # mirror the traffic model; no expected result is derived from this path.
    a_cache = b_cache = a_tag = b_tag = None
    for i, j in tile_indices(m, n, rows, cols, strategy):
        tag_a = "full" if strategy == "full_a_keep_b" else i
        if a_tag != tag_a or strategy == "reload":
            first, last = (0, m) if tag_a == "full" else (i * rows, min(m, (i + 1) * rows))
            a_cache = [a[r][:] for r in range(first, last)]
            a_tag = tag_a
        if b_tag != j or strategy in ("reload", "row_keep_a"):
            b_cache = [line[j * cols:min(n, (j + 1) * cols)] for line in b]
            b_tag = j
        vr, vc = min(rows, m - i * rows), min(cols, n - j * cols)
        sums = [[0] * vc for _ in range(vr)]
        step = k_block or k
        for z0 in range(0, k, step):
            for z in range(z0, min(k, z0 + step)):
                for r in range(vr):
                    av = a_cache[i * rows + r if tag_a == "full" else r][z]
                    for c in range(vc):
                        sums[r][c] += av * b_cache[z][c]
        for r in range(vr):
            for c in range(vc):
                out[i * rows + r][j * cols + c] = sums[r][c] + bias[j * cols + c]
    return out


def self_check():
    rng = random.Random(20261017)
    cases = [(5, 7, 9), (9, 5, 7), (1, 3, 1), (8, 4, 16)]
    checks = 0
    for m, k, n in cases:
        a = [[rng.randint(-7, 7) for _ in range(k)] for _ in range(m)]
        b = [[rng.randint(-5, 5) for _ in range(n)] for _ in range(k)]
        bias = [rng.randint(-9, 9) for _ in range(n)]
        expected = golden(a, b, bias)
        for rows, cols in ((2, 3), (4, 4), (8, 16)):
            for strategy in STRATEGIES:
                for k_block in (0, 2):
                    actual = tiled(a, b, bias, rows, cols, k_block, strategy)
                    if actual != expected:
                        raise AssertionError((m, k, n, rows, cols, strategy, k_block))
                    counts = count(m, k, n, rows, cols, strategy, clipped=True,
                                   k_block=k_block, spill_psum=k_block != 0)
                    if counts["traffic_16bit_words"]["output_write"] != 2 * m * n:
                        raise AssertionError("clipped output byte count")
                    checks += 1
    return {"small_matrix_cases": len(cases), "exact_comparisons": checks,
            "independent_reference": "direct triple-loop dot product plus bias",
            "result": "PASS"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=99)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--tiles", nargs="*", default=["8x16", "16x16", "8x32"])
    p.add_argument("--k-block", type=int, default=0)
    p.add_argument("--spill-psum", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    try:
        tiles = [tuple(map(int, t.lower().split("x"))) for t in args.tiles]
        if any(len(t) != 2 for t in tiles):
            raise ValueError("each tile must be ROWSxCOLS")
        checks = self_check()
        results = [count(args.m, args.k, args.n, r, c, s, clipped=clipped,
                         k_block=args.k_block, spill_psum=args.spill_psum)
                   for r, c in tiles for s in STRATEGIES for clipped in (False, True)]
    except (ValueError, AssertionError) as exc:
        p.error(str(exc))
    report = {"unit": "16-bit words", "reference_check": checks,
              "description": "analytical external transfer counts, no bus cycles or RTL performance",
              "results": results}
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
        print(f"PASS exact={checks['exact_comparisons']} cases={len(results)} report={args.out}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
