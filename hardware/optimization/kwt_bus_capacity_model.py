#!/usr/bin/env python3
"""Corrected FFN1 data-movement / capacity / cycle model with RTL evidence.

Scope is the contest's *undergraduate* task: the first FFN linear layer
    Y[99,256] = X[99,64] @ W[64,256] + b[256]
for each of the 12 encoder layers. GELU, the second linear layer, attention and
the classification head are out of scope (see 题目/README.md).

Why this script exists
----------------------
Several reports quote a "necessary" one-layer payload of 147,840 B. That number
adds an **unpadded** A count (99 rows -> 12,672 B) to a **tile-padded** Y write
count, so it is internally inconsistent. The hardware reads full 8-row A tiles
(13,312 B) even for the 3-row tail, so the physically consistent "unique data"
accounting is:

    valid  caliber = padded A + W + bias + valid  Y = 148,480 B
    padded caliber = padded A + W + bias + padded Y = 153,600 B
    (147,840 B = valid A + W + bias + valid Y  <-- mixes calibers)

The script also re-derives the per-tile reload traffic from the *RTL word
counts* (kwt_tile_system_model.sv) rather than from the published formula, and
it checks whether 512 KiB shared SRAM can hold all 12 layers of weights plus the
current operands and descriptors.

Evidence tags (never fused across levels)
  L0  arithmetic/analytic   byte counts, I/O floors, no schedule
  L1  RTL structure         word counts and port limits read from RTL source
  L2  testbench-timed model Verilator / E203 task models (simulated bus)
  L3  same-library synth    relative standard-cell area, no macro, no STA

Nothing here is contest-platform throughput. The RTL byte counts are payloads
issued on the *simulation* 128-bit bus; the contest System Bus width, latency and
arbitration are not provided.

Run:
  python3 hardware/optimization/kwt_bus_capacity_model.py
  python3 hardware/optimization/kwt_bus_capacity_model.py \
      --json /tmp/kwt-bus-capacity.json --markdown /tmp/kwt-bus-capacity.md
"""

import argparse
import json
from math import ceil
from pathlib import Path

# --- frozen contest problem shape (题目/README.md) --------------------------
M, K, N = 99, 64, 256
CLK_HZ = 150_000_000

# --- numeric container widths (currently INT16 A/W, INT32 bias/Y) ----------
A_BYTES = 2          # INT16 container per A element (narrow formats still use 16-bit lanes)
W_BYTES = 2          # INT16 container per W element
BIAS_BYTES = 4       # INT32
Y_BYTES = 4          # INT32

# --- platform constants (题目/README.md 第 2 节) ---------------------------
SHARED_SRAM_BYTES = 512 * 1024               # 524,288 B
SRAM_MACRO_WORDS = {"16bitx128": 128, "16bitx1024": 1024, "16bitx4096": 4096}

# --- RTL constants, copied from source (L1) --------------------------------
# kwt_dma.sv / kwt_ext_memory_model.sv: external byte address must be 16-B aligned.
DMA_BUS_W = 128                              # hardware/rtl/sram/kwt_memory_defs.svh
DMA_BUS_BYTES = DMA_BUS_W // 8               # 16
SRAM_WORD_BYTES = 2                          # 16-bit internal SRAM word
DESC_BYTES_PER_TILE = 32                     # kwt_nice_system_model.sv: two 128-bit beats

# kwt_tile_system_model.sv (8x16 descriptors): DMA word counts per tile.
#   A    = {6'b0, k_len_q, 3'b0} = k_len*8  words
#   B    = {5'b0, k_len_q, 4'b0} = k_len*16 words
#   bias = 16'd32 words
#   out  = 16'd256 words
# kwt_nice_system_model.sv fetches DESC_BYTES_PER_TILE per NICE submit.

# --- measured L1/L2 anchors, frozen (read-only references) -----------------
MEASURED = {
    "L1_staged_8x16_cycles_per_layer": {
        "cycles": 251_328, "source": "hardware/architecture/量化格式_硬件成本与周期对照_20261017.md"},
    "L1_staged_16x16_cycles_per_layer": {
        "cycles": 154_240, "source": "hardware/architecture/staged_task_row16_summary.json"},
    "L2_e203_8x16_full_layer_cycles": {
        "cycles": 838_028, "source": "hardware/optimization/integration_result.md"},
}

# --- INT32 accumulator evidence (software bound + RTL observation) ---------
ACCUM_OBSERVED_A12W10 = 4_733_248       # 240 real samples, bit-exact RTL
ACCUM_BOUND_A12W10 = 67_826_716         # software conservative bound
INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)

TILE_GEOMS = {"8x16": (8, 16), "16x16": (16, 16)}


# --------------------------------------------------------------------------
# L0/L1 shape and byte accounting
# --------------------------------------------------------------------------
def shapes(rows, cols):
    row_tiles = ceil(M / rows)
    col_tiles = ceil(N / cols)
    tiles = row_tiles * col_tiles
    return {
        "rows": rows, "cols": cols,
        "row_tiles": row_tiles, "col_tiles": col_tiles, "tiles": tiles,
        "padded_rows": row_tiles * rows, "padded_cols": col_tiles * cols,
        "padded_mac": tiles * rows * cols * K,
        "useful_mac": M * K * N,
    }


def byte_accounting(rows, cols):
    s = shapes(rows, cols)
    valid_a = M * K * A_BYTES
    padded_a = s["padded_rows"] * K * A_BYTES
    w_bytes = K * N * W_BYTES
    bias_bytes = N * BIAS_BYTES
    valid_y = M * N * Y_BYTES
    padded_y = s["tiles"] * rows * cols * Y_BYTES
    mixed = valid_a + w_bytes + bias_bytes + valid_y                 # 147,840 for 8x16
    unique_valid = padded_a + w_bytes + bias_bytes + valid_y         # 148,480 for 8x16
    unique_padded = padded_a + w_bytes + bias_bytes + padded_y       # 153,600 for 8x16
    return {
        "shape": s,
        "valid_a_bytes": valid_a,
        "padded_a_bytes": padded_a,
        "a_pad_bytes": padded_a - valid_a,
        "w_bytes": w_bytes,
        "bias_bytes": bias_bytes,
        "valid_y_bytes": valid_y,
        "padded_y_bytes": padded_y,
        "y_pad_bytes": padded_y - valid_y,
        "mixed_valid_a_valid_y_bytes": mixed,
        "unique_valid_caliber_bytes": unique_valid,
        "unique_padded_caliber_bytes": unique_padded,
    }


# --------------------------------------------------------------------------
# L1: per-tile reload traffic, recomputed from RTL word counts
# --------------------------------------------------------------------------
def reload_traffic(rows, cols):
    """Per-tile and per-layer bytes for the current reload descriptor.

    The RTL fills whole tiles: A = rows*K words, B = cols*K words, bias = cols*2
    words (INT32), out = rows*cols*2 words (INT32). Every word count below is a
    multiple of the 8 words/beat, so no partial final beat is wasted.
    """
    s = shapes(rows, cols)
    a_words = rows * K
    b_words = cols * K
    bias_words = cols * 2
    out_words = rows * cols * 2
    per_tile_words = a_words + b_words + bias_words + out_words
    per_tile_bytes = per_tile_words * SRAM_WORD_BYTES
    beats_per_tile = ceil(per_tile_words / (DMA_BUS_W // 16))  # 8 words / 128-bit beat
    return {
        "shape": s,
        "per_tile_words": {"a": a_words, "b": b_words, "bias": bias_words, "out": out_words},
        "per_tile_total_words": per_tile_words,
        "per_tile_bytes": per_tile_bytes,
        "per_tile_bus_beats": beats_per_tile,
        "layer_total_words": per_tile_words * s["tiles"],
        "layer_total_bytes": per_tile_bytes * s["tiles"],
        "layer_bus_beats": beats_per_tile * s["tiles"],
        "descriptor_bytes_per_layer": s["tiles"] * DESC_BYTES_PER_TILE,
        "payload_all_words_multiple_of_beat": all(
            w % 8 == 0 for w in (a_words, b_words, bias_words, out_words)),
    }


# --------------------------------------------------------------------------
# Shared 512 KiB SRAM budget (capacity only; ports are separate)
# --------------------------------------------------------------------------
def shared_sram_budget(rows=8, cols=16):
    acct = byte_accounting(rows, cols)
    s = acct["shape"]
    w_one = K * N * W_BYTES
    w12_16bit = 12 * w_one
    w12_packed10 = 12 * ceil(K * N * 10 / 8)
    bias_one = N * BIAS_BYTES
    desc = s["tiles"] * DESC_BYTES_PER_TILE

    def scenario(name, parts, note=""):
        total = sum(parts.values())
        return {
            "name": name, "parts": parts, "total_bytes": total,
            "fits": total <= SHARED_SRAM_BYTES,
            "slack_bytes": SHARED_SRAM_BYTES - total,
            "note": note,
        }

    scenarios = [
        scenario("12xW16 + paddedA + bias + validY + desc", {
            "w12_16bit": w12_16bit,
            "padded_a": acct["padded_a_bytes"],
            "bias": bias_one,
            "valid_y": acct["valid_y_bytes"],
            "desc": desc,
        }, "physically consistent current-layout hold of all 12 layers"),
        scenario("12xW16 + paddedA + bias + paddedY + desc", {
            "w12_16bit": w12_16bit,
            "padded_a": acct["padded_a_bytes"],
            "bias": bias_one,
            "padded_y": acct["padded_y_bytes"],
            "desc": desc,
        }, "if the 5 tail rows are also written back"),
        scenario("12xW16 + 12xbias + paddedA + validY + desc", {
            "w12_16bit": w12_16bit,
            "bias12": 12 * bias_one,
            "padded_a": acct["padded_a_bytes"],
            "valid_y": acct["valid_y_bytes"],
            "desc": desc,
        }, "also keeping every layer bias: exceeds 512 KiB"),
        scenario("12xW10packed + paddedA + bias + validY + desc", {
            "w12_packed10": w12_packed10,
            "padded_a": acct["padded_a_bytes"],
            "bias": bias_one,
            "valid_y": acct["valid_y_bytes"],
            "desc": desc,
        }, "needs a packing/unpacking path that does not exist yet"),
    ]
    return {
        "shared_sram_bytes": SHARED_SRAM_BYTES,
        "shared_sram_KiB": SHARED_SRAM_BYTES / 1024,
        "w_one_layer_16bit_bytes": w_one,
        "w12_16bit_bytes": w12_16bit,
        "w12_packed10_bytes": w12_packed10,
        "alignment_rule_bytes": DMA_BUS_BYTES,
        "scenarios": scenarios,
    }


# --------------------------------------------------------------------------
# L1: 16-bit SRAM macro port / read-write-conflict model
# --------------------------------------------------------------------------
def macro_port_model(rows, cols):
    """Structural port limits of kwt_pingpong_cacc_engine + kwt_operand_pingpong.

    kwt_operand_pingpong has one host write page and one compute read page per
    operand memory. kwt_pingpong_cacc_engine pulls one A word and one B word per
    cycle (compute_en) and walks lane_index 0..MAX_LANES-1, so one K costs
    MAX_LANES+2 cycles (READ_ISSUE + READ_CAPTURE + LAUNCH). Integration wires
    compute_page = load_page = 0, so host fill and compute cannot overlap.
    """
    max_lanes = max(rows, cols)
    per_k = max_lanes + 2
    drain_last = rows + cols + 4
    tile_elems = rows * cols
    tile_cycles = 1 + K * per_k + (drain_last + 1) + tile_elems * 4
    s = shapes(rows, cols)
    return {
        "geom": f"{rows}x{cols}",
        "max_lanes": max_lanes,
        "cycles_per_k": per_k,
        "a_words_per_k": rows,
        "b_words_per_k": cols,
        "words_per_k": rows + cols,
        "words_per_cycle_peak": 2,          # one A read + one B read
        "tile_cycles_structure": tile_cycles,
        "layer_compute_cycles_structure": tile_cycles * s["tiles"],
        "products_per_cycle": round(rows * cols / per_k, 4),
        "ideal_products_per_cycle": rows * cols,
        "host_fill_overlaps_compute": False,  # compute_page==load_page==0 in tile_system_model
    }


# --------------------------------------------------------------------------
# Same-caliber conservative cycle / traffic formulas
# --------------------------------------------------------------------------
def cycle_formulas(geom):
    """Same-caliber formulas for the row-broadcast baseline and 16-row candidate.

    compute = row_tiles*col_tiles*K*cols   (one B word/cycle broadcasts to `rows`)
    dump    = row_tiles*col_tiles*rows*cols (serial CACC/out-stage export, if unstaged)
    load    = payload_words / words_per_beat
    """
    rows, cols = TILE_GEOMS[geom]
    s = shapes(rows, cols)
    t = reload_traffic(rows, cols)
    words = t["layer_total_words"]
    compute = s["tiles"] * K * cols
    dump = s["tiles"] * rows * cols
    return {
        "geom": geom,
        "payload_words": words,
        "payload_bytes": t["layer_total_bytes"],
        "compute_cycles": compute,
        "dump_cycles": dump,
        "load_cycles_1word": words,
        "load_cycles_128bit": ceil(words / (DMA_BUS_W // 16)),
        "serial_total_1word": words + compute + dump,
        "serial_total_128bit": ceil(words / (DMA_BUS_W // 16)) + compute + dump,
        "measured_L1_staged": MEASURED.get(f"L1_staged_{geom}_cycles_per_layer", {}).get("cycles"),
    }


# --------------------------------------------------------------------------
# Structural evidence: assert the RTL source really contains the expressions
# this model depends on. If the RTL changes, the model fails loudly.
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]


def verify_rtl_sources(root=None):
    root = Path(root) if root else REPO_ROOT
    required = {
        "hardware/rtl/sram/kwt_memory_defs.svh": ["`define KWT_DMA_BUS_W 128"],
        "hardware/rtl/control/kwt_tile_system_model.sv": [
            "{6'b0, k_len_q, 3'b0}",   # A words = k_len*8
            "{5'b0, k_len_q, 4'b0}",   # W words = k_len*16
            "16'd32",                  # bias words
            "16'd256",                 # out words
            ".compute_page(1'b0), .load_page(1'b0)",  # overlap disabled
        ],
        "hardware/rtl/sram/kwt_operand_pingpong.sv": [
            "if (a_host_en && a_host_we && load_ready)",
            "if (b_host_en && b_host_we && load_ready)",
            "if (compute_en) begin",
        ],
        "hardware/rtl/pulse_array/kwt_pingpong_cacc_engine.sv": [
            "if (lane_index == LANE_W'(MAX_LANES-1)) state <= READ_CAPTURE;",
        ],
    }
    checks = []
    for rel, needles in required.items():
        path = root / rel
        if not path.exists():
            checks.append(f"SKIP {rel} (not found under {root})")
            continue
        text = path.read_text(errors="replace")
        for needle in needles:
            assert needle in text, f"{rel} lost expected fragment: {needle!r}"
        checks.append(f"RTL {rel}: {len(needles)} fragments present")
    return checks


# --------------------------------------------------------------------------
# Self-check: assertions on shapes, capacity, repeated tile traffic, ranges
# --------------------------------------------------------------------------
def self_check():
    checks = list(verify_rtl_sources())

    # --- shapes ---
    s8, s16 = shapes(8, 16), shapes(16, 16)
    assert (s8["row_tiles"], s8["col_tiles"], s8["tiles"]) == (13, 16, 208), s8
    assert (s16["row_tiles"], s16["col_tiles"], s16["tiles"]) == (7, 16, 112), s16
    assert s8["useful_mac"] == 1_622_016
    assert s8["padded_mac"] == 1_703_936 and s16["padded_mac"] == 1_835_008
    checks.append("shapes: 208 tiles (8x16) / 112 tiles (16x16); padded MAC 1,703,936 / 1,835,008")

    # --- corrected byte accounting ---
    a8 = byte_accounting(8, 16)
    assert a8["valid_a_bytes"] == 12_672
    assert a8["padded_a_bytes"] == 13_312
    assert a8["w_bytes"] == 32_768
    assert a8["bias_bytes"] == 1_024
    assert a8["valid_y_bytes"] == 101_376
    assert a8["padded_y_bytes"] == 106_496
    # the disputed number and the two consistent calibers
    assert a8["mixed_valid_a_valid_y_bytes"] == 147_840
    assert a8["unique_valid_caliber_bytes"] == 148_480
    assert a8["unique_padded_caliber_bytes"] == 153_600
    # mixed == unique_valid - A pad
    assert a8["unique_valid_caliber_bytes"] - a8["a_pad_bytes"] == a8["mixed_valid_a_valid_y_bytes"]
    checks.append("bytes: valid/padded A 12,672/13,312; W 32,768; bias 1,024; "
                  "valid/padded Y 101,376/106,496; unique 148,480/153,600 (mixed 147,840)")

    # --- repeated tile traffic from RTL word counts ---
    t8, t16 = reload_traffic(8, 16), reload_traffic(16, 16)
    assert t8["per_tile_bytes"] == 3_648, t8
    assert t8["per_tile_words"] == {"a": 512, "b": 1024, "bias": 32, "out": 256}, t8
    assert t8["layer_total_bytes"] == 758_784, t8
    assert t8["layer_bus_beats"] == 47_424
    assert all(t8[x] for x in ("payload_all_words_multiple_of_beat",))
    assert t16["per_tile_bytes"] == 5_184 and t16["layer_total_bytes"] == 580_608, t16
    checks.append("traffic: 3,648 B/tile x208 = 758,784 B (8x16); "
                  "5,184 B/tile x112 = 580,608 B (16x16)")

    # --- capacity: 512 KiB fits the corrected current-layout hold ---
    b = shared_sram_budget(8, 16)
    fit = b["scenarios"][0]
    assert fit["fits"] and fit["slack_bytes"] == 8_704, fit
    assert b["w12_16bit_bytes"] == 393_216
    assert b["w12_packed10_bytes"] == 245_760
    over = b["scenarios"][2]
    assert not over["fits"] and over["total_bytes"] == 526_848, over
    checks.append("capacity: 12xW16+A+bias+validY+desc = 515,584 B <= 524,288 B "
                  "(slack 8,704); +12 biases overflows by 2,560 B")

    # --- macro port limits ---
    p8, p16 = macro_port_model(8, 16), macro_port_model(16, 16)
    assert p8["cycles_per_k"] == 18 and p8["words_per_cycle_peak"] == 2
    assert abs(p8["products_per_cycle"] - 128 / 18) < 1e-3
    assert abs(p16["products_per_cycle"] - 256 / 18) < 1e-3
    checks.append("ports: 2 words/cycle (1 A + 1 B), 18 cyc/K; "
                  "128/18 and 256/18 products/cycle")

    # --- cycle formulas ---
    c8, c16 = cycle_formulas("8x16"), cycle_formulas("16x16")
    assert c8["compute_cycles"] == 212_992 and c16["compute_cycles"] == 114_688
    assert c8["dump_cycles"] == 26_624 and c16["dump_cycles"] == 28_672
    assert c8["serial_total_128bit"] == 47_424 + 212_992 + 26_624
    checks.append("cycles: compute 212,992/114,688; dump 26,624/28,672; "
                  "serial@128-bit 287,040/179,648")

    # --- integer ranges ---
    assert INT32_MIN <= ACCUM_BOUND_A12W10 <= INT32_MAX
    assert ACCUM_OBSERVED_A12W10 <= ACCUM_BOUND_A12W10
    assert M * N == 25_344 and 12 * M * N == 304_128
    checks.append("ranges: A12W10 bound 67,826,716 < 2^31-1; observed max 4,733,248; "
                  "25,344 valid outputs/layer")

    return {"result": "PASS", "checks": checks}


# --------------------------------------------------------------------------
def build_report():
    return {
        "task": "FFN1 first linear layer only: Y[99,256]=X[99,64]@W[64,256]+b[256]",
        "clock_hz": CLK_HZ,
        "evidence_levels": {
            "L0_arithmetic": "byte counts and I/O floors, no schedule",
            "L1_rtl_structure": "word counts / port limits read from RTL source",
            "L2_testbench": "Verilator / E203 task models on a simulated bus",
            "L3_synthesis": "same-library relative cell area, no macro, no STA",
        },
        "self_check": self_check(),
        "byte_accounting": {g: byte_accounting(*TILE_GEOMS[g]) for g in TILE_GEOMS},
        "reload_traffic": {g: reload_traffic(*TILE_GEOMS[g]) for g in TILE_GEOMS},
        "shared_sram": shared_sram_budget(8, 16),
        "macro_ports": {g: macro_port_model(*TILE_GEOMS[g]) for g in TILE_GEOMS},
        "cycle_formulas": {g: cycle_formulas(g) for g in TILE_GEOMS},
        "measured_L1_L2": MEASURED,
        "int32": {"bound_A12W10": ACCUM_BOUND_A12W10,
                  "observed_A12W10": ACCUM_OBSERVED_A12W10,
                  "int32_max": INT32_MAX, "fits": ACCUM_BOUND_A12W10 <= INT32_MAX},
        "rtl_evidence": {
            "per_tile_words": "hardware/rtl/control/kwt_tile_system_model.sv:70-72",
            "dma_beat_alignment": "hardware/rtl/dma/kwt_dma.sv:118-125",
            "desc_two_beats": "hardware/rtl/control/kwt_nice_system_model.sv:106-120",
            "operand_single_read_port": "hardware/rtl/sram/kwt_operand_pingpong.sv:27-36",
            "pingpong_no_overlap_wiring": "hardware/rtl/control/kwt_tile_system_model.sv:90",
            "row_broadcast_core": "hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv",
            "shared_ram_dump_words": "hardware/rtl/control/tb/tb_kwt_real_layer_e203.sv:48,117-146",
        },
        "missing_official": [
            "16-bit SRAM macro access latency / minimum cycle",
            "single- vs true-dual-port semantics and byte enables",
            "contest System Bus width, latency, arbitration",
            "macro area and power for the 2.25 mm^2 / energy budget",
            "target SDC / STA at 150 MHz",
        ],
    }


def _fmt(n):
    return f"{n:,}"


def render_markdown(rep):
    sc = rep["self_check"]
    a8 = rep["byte_accounting"]["8x16"]
    a16 = rep["byte_accounting"]["16x16"]
    t8 = rep["reload_traffic"]["8x16"]
    t16 = rep["reload_traffic"]["16x16"]
    p8 = rep["macro_ports"]["8x16"]
    c8 = rep["cycle_formulas"]["8x16"]
    c16 = rep["cycle_formulas"]["16x16"]
    lines = ["# KWT FFN1 bus / capacity / cycle model (generated)", ""]
    lines.append(f"Self-check: {sc['result']} ({len(sc['checks'])} assertions)")
    lines.append("")
    lines.append("## Corrected unique-data calibers (8x16)")
    lines.append("")
    lines.append("| caliber | padded A | W | bias | Y | total |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    lines.append(f"| mixed (existing report) | {_fmt(a8['valid_a_bytes'])} | {_fmt(a8['w_bytes'])} | "
                 f"{_fmt(a8['bias_bytes'])} | {_fmt(a8['valid_y_bytes'])} | "
                 f"**{_fmt(a8['mixed_valid_a_valid_y_bytes'])}** |")
    lines.append(f"| unique valid | {_fmt(a8['padded_a_bytes'])} | {_fmt(a8['w_bytes'])} | "
                 f"{_fmt(a8['bias_bytes'])} | {_fmt(a8['valid_y_bytes'])} | "
                 f"**{_fmt(a8['unique_valid_caliber_bytes'])}** |")
    lines.append(f"| unique padded | {_fmt(a8['padded_a_bytes'])} | {_fmt(a8['w_bytes'])} | "
                 f"{_fmt(a8['bias_bytes'])} | {_fmt(a8['padded_y_bytes'])} | "
                 f"**{_fmt(a8['unique_padded_caliber_bytes'])}** |")
    lines.append("")
    lines.append("## Reload traffic from RTL word counts")
    lines.append("")
    lines.append("| geom | A/tile | B/tile | bias/tile | out/tile | total/tile | layer bytes |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for t, g in ((t8, "8x16"), (t16, "16x16")):
        w = t["per_tile_words"]
        lines.append(f"| {g} | {_fmt(w['a']*2)} | {_fmt(w['b']*2)} | {_fmt(w['bias']*2)} | "
                     f"{_fmt(w['out']*2)} | {_fmt(t['per_tile_bytes'])} | "
                     f"**{_fmt(t['layer_total_bytes'])}** |")
    lines.append("")
    lines.append("## Same-caliber conservative formulas")
    lines.append("")
    lines.append("| geom | payload words | compute | dump | load@128b | serial@128b | L1 staged |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c, g in ((c8, "8x16"), (c16, "16x16")):
        m = c["measured_L1_staged"] or "-"
        lines.append(f"| {g} | {_fmt(c['payload_words'])} | {_fmt(c['compute_cycles'])} | "
                     f"{_fmt(c['dump_cycles'])} | {_fmt(c['load_cycles_128bit'])} | "
                     f"{_fmt(c['serial_total_128bit'])} | {m} |")
    lines.append("")
    lines.append(f"Macro-port limit: {p8['cycles_per_k']} cyc/K, "
                 f"{p8['words_per_cycle_peak']} words/cycle, "
                 f"{p8['products_per_cycle']} products/cycle (8x16).")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path)
    p.add_argument("--markdown", type=Path)
    args = p.parse_args()
    try:
        rep = build_report()
    except AssertionError as exc:  # pragma: no cover
        print(f"SELF-CHECK FAILED: {exc}")
        raise SystemExit(1)
    payload = json.dumps(rep, ensure_ascii=False, indent=2) + "\n"
    md = render_markdown(rep)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md)
    print(md, end="")
    print(f"PASS self_check={rep['self_check']['result']} "
          f"assertions={len(rep['self_check']['checks'])}")
    if args.json:
        print(f"PASS wrote {args.json}")


if __name__ == "__main__":
    main()
