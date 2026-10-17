#!/usr/bin/env python3
"""Count 16-bit operand transfers for FFN1 tiling; no bus timing assumptions."""

import argparse
from math import ceil


def count(m: int, k: int, n: int, rows: int, cols: int) -> dict[str, int]:
    row_tiles = ceil(m / rows)
    col_tiles = ceil(n / cols)
    tiles = row_tiles * col_tiles
    # Each descriptor currently fills a full padded A and B tile.
    a_tile = rows * k
    b_tile = cols * k
    bias_tile = cols * 2  # INT32 bias in 16-bit words
    out_tile = rows * cols * 2  # INT32 output in 16-bit words
    return {
        "tiles": tiles,
        "a_tile": a_tile,
        "b_tile": b_tile,
        "baseline_a": tiles * a_tile,
        "baseline_b": tiles * b_tile,
        "row_outer_a": row_tiles * a_tile,
        "row_outer_b": tiles * b_tile,
        "col_outer_a": tiles * a_tile,
        "col_outer_b": col_tiles * b_tile,
        "baseline_bias": tiles * bias_tile,
        "col_outer_bias": col_tiles * bias_tile,
        "output": tiles * out_tile,
        "useful_mac": m * k * n,
        "padded_mac": tiles * rows * cols * k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=99)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=16)
    args = parser.parse_args()
    if min(args.m, args.k, args.n, args.rows, args.cols) <= 0:
        parser.error("dimensions must be positive")
    result = count(args.m, args.k, args.n, args.rows, args.cols)
    print(f"shape={args.m}x{args.k} x {args.k}x{args.n} "
          f"tile={args.rows}x{args.cols} tiles={result['tiles']}")
    print("strategy        A_words B_words bias_words output_words input_total")
    for strategy in ("baseline", "row_outer", "col_outer"):
        bias = result["col_outer_bias"] if strategy == "col_outer" else result["baseline_bias"]
        a = result[f"{strategy}_a"]
        b = result[f"{strategy}_b"]
        print(f"{strategy:14} {a:7} {b:7} {bias:10} "
              f"{result['output']:12} {a+b+bias:11}")
    print(f"MAC useful={result['useful_mac']} padded={result['padded_mac']}")


if __name__ == "__main__":
    main()
