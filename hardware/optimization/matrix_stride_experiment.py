#!/usr/bin/env python3
"""Exercise row-major 2D tile copies and exact tile GEMM, without bus timing."""

from dataclasses import dataclass
from math import ceil


M, K, N = 99, 64, 256
ROWS, COLS = 8, 16
BUS_WORDS = 8  # 128-bit abstract beat with 16-bit words; count only


def s16(word: int) -> int:
    return word - 65536 if word & 0x8000 else word


@dataclass(frozen=True)
class Copy2D:
    rows: int
    row_words: int
    src_stride_words: int
    dst_stride_words: int

    @property
    def beats(self) -> int:
        return self.rows * ceil(self.row_words / BUS_WORDS)

    @property
    def words(self) -> int:
        return self.rows * self.row_words


def copy2d(src: list[int], dst: list[int], src_base: int, dst_base: int,
           shape: Copy2D) -> None:
    for row in range(shape.rows):
        first_src = src_base + row * shape.src_stride_words
        first_dst = dst_base + row * shape.dst_stride_words
        for word in range(shape.row_words):
            dst[first_dst + word] = src[first_src + word]


def a_value(row: int, k: int) -> int:
    return ((row * 13 + k * 7) % 65) - 32


def b_value(k: int, col: int) -> int:
    return ((k * 11 - col * 3) % 57) - 28


def run_tile(row_start: int, col_start: int) -> dict:
    valid_rows = min(ROWS, M - row_start)
    valid_cols = min(COLS, N - col_start)
    source_a = [a_value(r, k) & 0xFFFF for r in range(M) for k in range(K)]
    source_b = [b_value(k, c) & 0xFFFF for k in range(K) for c in range(N)]
    local_a = [0] * (ROWS * K)
    local_b = [0] * (K * COLS)
    a_copy = Copy2D(valid_rows, K, K, K)
    b_copy = Copy2D(K, valid_cols, N, COLS)
    copy2d(source_a, local_a, row_start * K, 0, a_copy)
    copy2d(source_b, local_b, col_start, 0, b_copy)

    # The row-broadcast kernel keeps an A row in registers and streams
    # B[k,c]. This also verifies the physical two-dimensional address map.
    local_c = [0] * (ROWS * COLS * 2)
    reference_c = [0] * (M * N * 2)
    for r in range(valid_rows):
        for c in range(valid_cols):
            got = sum(s16(local_a[r*K + k]) * s16(local_b[k*COLS + c])
                      for k in range(K))
            expected = sum(a_value(row_start+r, k) * b_value(k, col_start+c)
                           for k in range(K))
            assert got == expected, (row_start, col_start, r, c)
            word = got & 0xFFFFFFFF
            local_c[(r*COLS+c)*2] = word & 0xFFFF
            local_c[(r*COLS+c)*2+1] = word >> 16
            reference_c[((row_start+r)*N+col_start+c)*2] = word & 0xFFFF
            reference_c[((row_start+r)*N+col_start+c)*2+1] = word >> 16

    output_c = [0xA5A5] * (M * N * 2)
    c_copy = Copy2D(valid_rows, valid_cols * 2, COLS * 2, N * 2)
    copy2d(local_c, output_c, 0, (row_start*N + col_start)*2, c_copy)
    for r in range(valid_rows):
        start = ((row_start+r)*N+col_start)*2
        assert output_c[start:start+valid_cols*2] == reference_c[start:start+valid_cols*2]
    if row_start + valid_rows < M:
        untouched = ((row_start+valid_rows)*N+col_start)*2
        assert output_c[untouched] == 0xA5A5
    return {
        "tile": [row_start, col_start], "valid_outputs": valid_rows*valid_cols,
        "a": {"rows": a_copy.rows, "words": a_copy.words, "beats": a_copy.beats},
        "b": {"rows": b_copy.rows, "words": b_copy.words, "beats": b_copy.beats},
        "c": {"rows": c_copy.rows, "words": c_copy.words, "beats": c_copy.beats},
    }


if __name__ == "__main__":
    for row_start, col_start in ((0, 0), (96, 240)):
        result = run_tile(row_start, col_start)
        print(f"PASS tile={result['tile']} outputs={result['valid_outputs']} "
              f"A={result['a']} B={result['b']} C={result['c']}")
