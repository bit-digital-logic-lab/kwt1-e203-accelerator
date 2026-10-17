#!/usr/bin/env python3
"""Compute/dataflow bounds for one contest FFN1 task; no clock closure claim."""

from math import ceil

M, K, N, COLS = 99, 64, 256, 16
CLK_HZ = 150_000_000
MACS = M * K * N
A_BYTES = M * K * 2
W_BYTES = K * N * 2
BIAS_BYTES = N * 4
OUT_BYTES = M * N * 4
TASK_BYTES = A_BYTES + W_BYTES + BIAS_BYTES + OUT_BYTES
PACKED_A11W10_BYTES = (M * K * 11 + K * N * 10) // 8 + BIAS_BYTES + OUT_BYTES


def time_ms(cycles):
    return 1000 * cycles / CLK_HZ


print(f"FFN1: {MACS:,} useful MAC/layer; {TASK_BYTES:,} B minimum "
      "for unpacked INT16 A/W + INT32 bias/output")
print(f"128 fully fed MAC ideal: {ceil(MACS / 128):,} cycles, "
      f"{time_ms(ceil(MACS / 128)):.4f} ms/layer; requires 128 products/cycle")
print(f"I/O-only floor: {ceil(TASK_BYTES / 2):,} cycles at one shared 16-bit "
      f"word/cycle, or {ceil(TASK_BYTES / 16):,} cycles at ideal 128-bit bus; "
      "these are independent lower bounds, not additive schedules")
print(f"Perfectly packed A11W10 moves {PACKED_A11W10_BYTES:,} B/layer; "
      f"I/O-only floor {ceil(PACKED_A11W10_BYTES / 2):,} cycles at one "
      "16-bit word/cycle, before unpacking, alignment and arbitration")
print("rows  compute cycles  ms/layer  ms/12 layers  A reads/cycle  CACC banks  "
      "serial CACC dump cycles")
for rows in (8, 16, 32):
    row_tiles = ceil(M / rows)
    col_tiles = ceil(N / COLS)
    compute = row_tiles * col_tiles * K * COLS
    dump = row_tiles * col_tiles * rows * COLS
    a_read_ports = ceil(rows / COLS)
    print(f"{rows:4}  {compute:14,}  {time_ms(compute):8.4f}  "
          f"{12*time_ms(compute):12.4f}  {a_read_ports:13}  "
          f"{rows:10}  {dump:23,}")
print("Assumes one B word/cycle, ROWS parallel multipliers and CACC writes, "
      "full A residency, W reuse, no stalls. Each CACC bank needs one 32-bit "
      "read and write per cycle. One 16-bit SRAM port or one CACC bank shared "
      "among rows cannot sustain the printed compute rate.")
