#!/usr/bin/env python3
"""Whole-task performance/storage trade-off for the undergraduate FFN1 layer.

Scope is fixed to the contest's undergraduate task: one layer of
    Y[99,256] = X[99,64] @ W[64,256] + b[256]
executed by the accelerator. GELU, the second linear layer, attention and the
12-layer encoder are out of scope (see 题目/README.md).

The script never fuses numbers that come from different evidence levels. Each
quantity is tagged with one of:

  L0  arithmetic ideal        -- pure parallelism / I/O floors, no schedule
  L1  RTL core + testbench    -- measured Verilator task model of one layer
  L2  E203 NICE-like SoC model-- measured 208-tile full-layer SoC model run
  L3  target-library synthesis-- same-library relative standard-cell area only

All cycle figures marked L1/L2 come from *testbench-timed* models (A/W/bias
load, prefetch, shared bus and writeback are simulated by the testbench). They
are not measured contest-platform throughput and are not clock-closed here.

Run:
  python3 hardware/optimization/ffn1_perf_storage_tradeoff.py
  python3 hardware/optimization/ffn1_perf_storage_tradeoff.py --json /tmp/kwt-ffn1-tradeoff.json
"""

import argparse
import json
from math import ceil
from pathlib import Path

# --- frozen contest problem shape (题目/README.md) -------------------------
M, K, N = 99, 64, 256
CLK_HZ = 150_000_000
USEFUL_MAC = M * K * N  # 1,622,016 per layer

# --- platform constants (题目/README.md 第 2 节) ---------------------------
SHARED_SRAM_BYTES = 512 * 1024
SRAM_MACRO_WORDS = {"16bitx128": 128, "16bitx1024": 1024, "16bitx4096": 4096}

# --- quantization candidates (software/results, 480-sample validation) -----
# worst_nrmse_pct = max over 12 layers of pooled per-layer NRMSE on 480 samples.
QUANT_SCHEMES = {
    "A8W8":   {"act_bits": 8,  "w_bits": 8,  "packed_w_kib_12": 192.0, "worst_nrmse_pct": 1.121379},
    "A12W9":  {"act_bits": 12, "w_bits": 9,  "packed_w_kib_12": 216.0, "worst_nrmse_pct": 0.261276},
    "A11W10": {"act_bits": 11, "w_bits": 10, "packed_w_kib_12": 240.0, "worst_nrmse_pct": 0.181954},
    "A12W10": {"act_bits": 12, "w_bits": 10, "packed_w_kib_12": 240.0, "worst_nrmse_pct": 0.147048},
}
VALIDATION_BOUNDARY = "480 validation samples, software integer model; RTL subset differs"

# --- measured evidence, stored with its boundary --------------------------
# L1: one-layer Verilator staged task model (testbench-timed load/writeback).
STAGED_MODEL_CYCLES_PER_LAYER = {
    8:  {"cycles": 251_328, "source": "hardware/architecture/量化格式_硬件成本与周期对照_20261017.md"},
    16: {"cycles": 154_240, "source": "hardware/architecture/staged_task_row16_summary.json (name=full)"},
}
# L2: full-layer E203 NICE-like system-model regression (abstract 128-bit bus).
E203_FULL_LAYER = {
    "rows": 8, "cols": 16, "tiles": 208, "cycles": 838_028,
    "source": "hardware/optimization/integration_result.md",
}
# L3: Nangate45 typical same-library relative cell area (library units, NOT mm^2).
SYNTH_AREA_UNITS = {
    8:  {"core": 60_358.060,  "source": "hardware/architecture/README.md"},
    16: {"core": 120_434.692, "source": "hardware/architecture/量化格式_硬件成本与周期对照_20261017.md"},
}
OUTPUT_STAGE_AREA_UNITS = 66_224.956  # extra module, not inside core area

# Observed INT32 accumulator magnitudes (bit-exact RTL, 240 validation samples).
ACCUM_OBSERVED = {"A11W10": 2_365_463, "A12W10": 4_733_248}
ACCUM_CONSERVATIVE_A12W10 = 67_826_716  # software bound, still < 2^31-1
INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)

# --- explicit configurations to compare -----------------------------------
CONFIGS = [
    {"name": "8x16/A8W8/reload",
     "rows": 8, "cols": 16, "scheme": "A8W8", "strategy": "reload",
     "note": "area-oriented candidate; loosest local accuracy"},
    {"name": "8x16/A12W10/col_keep_b",
     "rows": 8, "cols": 16, "scheme": "A12W10", "strategy": "col_keep_b",
     "note": "lowest traffic; B column block fits one 1024-word macro"},
    {"name": "16x16/A12W10/row_keep_a",
     "rows": 16, "cols": 16, "scheme": "A12W10", "strategy": "row_keep_a",
     "note": "lowest measured L1 cycles; most accumulator banks"},
]

STRATEGIES = ("reload", "row_keep_a", "col_keep_b", "full_a_keep_b")


# --------------------------------------------------------------------------
# Traffic counting (16-bit words moved through the shared SRAM path).
# Mirrors the accounting in tiling_experiment.py but is independent code.
# --------------------------------------------------------------------------
def tile_order(m, n, rows, cols, strategy):
    mr, nc = ceil(m / rows), ceil(n / cols)
    if strategy == "row_keep_a":
        return [(i, j) for i in range(mr) for j in range(nc)]
    return [(i, j) for j in range(nc) for i in range(mr)]


def count_traffic(m, k, n, rows, cols, strategy, spill_psum=False, k_block=0):
    if min(m, k, n, rows, cols) <= 0 or strategy not in STRATEGIES:
        raise ValueError("positive dimensions and a known strategy required")
    chunks = ceil(k / k_block) if k_block else 1
    t = {x: 0 for x in ("a_read", "b_read", "bias_read",
                        "psum_read", "psum_write", "output_write")}
    a_loaded = b_loaded = bias_loaded = None
    for i, j in tile_order(m, n, rows, cols, strategy):
        a_tag = "full" if strategy == "full_a_keep_b" else i
        need_a = strategy == "reload" or a_tag != a_loaded
        need_b = strategy in ("reload", "row_keep_a") or j != b_loaded
        need_bias = strategy in ("reload", "row_keep_a") or j != bias_loaded
        # full padded tiles: existing descriptors move whole rows/columns
        if need_a:
            t["a_read"] += (m if a_tag == "full" else rows) * k
            a_loaded = a_tag
        if need_b:
            t["b_read"] += cols * k
            b_loaded = j
        if need_bias:
            t["bias_read"] += 2 * cols  # INT32 bias = two 16-bit words
            bias_loaded = j
        t["output_write"] += 2 * rows * cols
        if spill_psum:
            t["psum_read"] += 2 * rows * cols * (chunks - 1)
            t["psum_write"] += 2 * rows * cols * (chunks - 1)
    return t


def onchip_buffers(m, k, n, rows, cols, strategy):
    a_words = m * k if strategy == "full_a_keep_b" else rows * k
    return {
        "a_cache_words": a_words,
        "b_cache_words": cols * k,
        "bias_cache_words": 2 * cols,      # INT32
        "psum_words": 2 * rows * cols,     # INT32
        "output_stage_words": 2 * rows * cols,  # INT32 staging
    }


def macro_plan(words, macro_name="16bitx4096"):
    size = SRAM_MACRO_WORDS[macro_name]
    return {"macro": macro_name, "count": ceil(words / size), "words": words}


def macro_options(words):
    """Macro counts for one buffer under each allowed 16-bit macro size.

    Reports the count if the buffer is implemented with a single standard
    macro type, plus the smallest total macro count (using the largest macro)
    and the tightest mix. A partial tail still costs one whole macro.
    """
    per_size = {name: ceil(words / size) for name, size in SRAM_MACRO_WORDS.items()}
    fewest = ceil(words / SRAM_MACRO_WORDS["16bitx4096"])
    remaining, mix = words, {}
    for name in sorted(SRAM_MACRO_WORDS, key=SRAM_MACRO_WORDS.get, reverse=True):
        size = SRAM_MACRO_WORDS[name]
        count, remaining = divmod(remaining, size)
        if count:
            mix[name] = count
    if remaining:
        mix["16bitx128"] = mix.get("16bitx128", 0) + 1
    return {"words": words, "count_by_macro": per_size,
            "fewest_macros": fewest, "tight_mix": mix,
            "tight_mix_count": sum(mix.values())}


# --------------------------------------------------------------------------
# Cycle / time models per evidence level.
# --------------------------------------------------------------------------
def arithmetic_ideal(rows):
    row_tiles, col_tiles = ceil(M / rows), ceil(N / 16)
    # row-broadcast: one B word/cycle, `rows` multipliers, one column per cycle
    compute = row_tiles * col_tiles * K * 16
    dump = row_tiles * col_tiles * rows * 16  # serial CACC export, if unstaged
    return {
        "mac_ideal_128": ceil(USEFUL_MAC / 128),
        "row_broadcast_compute": compute,
        "row_broadcast_plus_dump": compute + dump,
        "rows": rows,
    }


def packed_traffic(scheme):
    act = QUANT_SCHEMES[scheme]["act_bits"]
    w = QUANT_SCHEMES[scheme]["w_bits"]
    a_bytes = ceil(M * K * act / 8)
    w_bytes = ceil(K * N * w / 8)
    bias_bytes = N * 4
    out_bytes = M * N * 4
    return {"a_bytes": a_bytes, "w_bytes": w_bytes, "bias_bytes": bias_bytes,
            "out_bytes": out_bytes, "total_bytes": a_bytes + w_bytes + bias_bytes + out_bytes}


def time_ms(cycles):
    return 1000.0 * cycles / CLK_HZ


# --------------------------------------------------------------------------
# Independent correctness reference: small integer GEMM, tiled vs direct.
# --------------------------------------------------------------------------
def _direct_gemm(a, b, bias):
    return [[sum(a[i][z] * b[z][j] for z in range(len(b))) + bias[j]
             for j in range(len(b[0]))] for i in range(len(a))]


def _tiled_gemm(a, b, bias, rows, cols, strategy):
    m, k, n = len(a), len(b), len(b[0])
    out = [[0] * n for _ in range(m)]
    for i, j in tile_order(m, n, rows, cols, strategy):
        vr, vc = min(rows, m - i * rows), min(cols, n - j * cols)
        for r in range(vr):
            for c in range(vc):
                acc = bias[j * cols + c]
                for z in range(k):
                    acc += a[i * rows + r][z] * b[z][j * cols + c]
                out[i * rows + r][j * cols + c] = acc
    return out


def _lcg(state):
    return (1103515245 * state + 12345) % (2**31)


def self_check():
    """Bit-exact small GEMM + accounting invariants; returns a summary dict."""
    state = 20261018
    cases = [(5, 7, 9), (9, 5, 7), (1, 3, 1), (16, 4, 16), (13, 8, 6)]
    comparisons = 0
    for m, k, n in cases:
        a = [[0] * k for _ in range(m)]
        b = [[0] * n for _ in range(k)]
        for i in range(m):
            for z in range(k):
                state = _lcg(state)
                a[i][z] = state % 15 - 7
        for z in range(k):
            for j in range(n):
                state = _lcg(state)
                b[z][j] = state % 11 - 5
        bias = [0] * n
        for j in range(n):
            state = _lcg(state)
            bias[j] = state % 19 - 9
        expected = _direct_gemm(a, b, bias)
        for rows, cols in ((2, 3), (4, 4), (8, 16)):
            for strategy in STRATEGIES:
                if _tiled_gemm(a, b, bias, rows, cols, strategy) != expected:
                    raise AssertionError((m, k, n, rows, cols, strategy))
                comparisons += 1

    # accounting invariants against the published tiling table
    checks = []
    published = {
        (8, 16, "reload"): 326_144,
        (8, 16, "col_keep_b"): 123_392,
        (8, 16, "full_a_keep_b"): 23_232,
        (16, 16, "row_keep_a"): 125_440,
    }
    for (rows, cols, strategy), want in published.items():
        t = count_traffic(M, K, N, rows, cols, strategy)
        got = t["a_read"] + t["b_read"] + t["bias_read"]
        if got != want:
            raise AssertionError(f"traffic {rows}x{cols}/{strategy}: {got} != {want}")
        checks.append(f"{rows}x{cols}/{strategy} input={got}")

    # capacity invariant: chosen macros cover each buffer
    for rows, cols in ((8, 16), (16, 16)):
        for strategy in STRATEGIES:
            buf = onchip_buffers(M, K, N, rows, cols, strategy)
            for key, words in buf.items():
                plan = macro_options(words)
                cap = SRAM_MACRO_WORDS["16bitx4096"] * plan["fewest_macros"]
                if cap < words:
                    raise AssertionError(f"capacity {key} {rows}x{cols}")

    # INT32 accumulator feasibility (conservative software bound)
    if not (INT32_MIN <= -ACCUM_CONSERVATIVE_A12W10 <= INT32_MAX
            and INT32_MIN <= ACCUM_CONSERVATIVE_A12W10 <= INT32_MAX):
        raise AssertionError("INT32 accumulator bound")
    return {"exact_gemm_comparisons": comparisons,
            "traffic_checks": checks,
            "int32_bound_ok": ACCUM_CONSERVATIVE_A12W10 <= INT32_MAX,
            "result": "PASS"}


# --------------------------------------------------------------------------
def analyse(config):
    rows, cols = config["rows"], config["cols"]
    scheme = config["scheme"]
    strategy = config["strategy"]
    traffic = count_traffic(M, K, N, rows, cols, strategy)
    buffers = onchip_buffers(M, K, N, rows, cols, strategy)
    words = {name: words for name, words in buffers.items()}
    macro_plans = {name: macro_options(w) for name, w in words.items()}
    packed = packed_traffic(scheme)
    ideal = arithmetic_ideal(rows)

    total_input_words = (traffic["a_read"] + traffic["b_read"] + traffic["bias_read"])
    container_bytes = (
        traffic["a_read"] * 2 + traffic["b_read"] * 2 +
        traffic["bias_read"] * 2 + traffic["output_write"] * 2
    )
    staged = STAGED_MODEL_CYCLES_PER_LAYER.get(rows)
    l2 = E203_FULL_LAYER if (rows == E203_FULL_LAYER["rows"]
                             and cols == E203_FULL_LAYER["cols"]) else None
    synth = SYNTH_AREA_UNITS.get(rows)

    return {
        "name": config["name"], "note": config["note"],
        "shape": [M, K, N], "tile": [rows, cols], "scheme": scheme,
        "strategy": strategy,
        "quant_worst_nrmse_pct": QUANT_SCHEMES[scheme]["worst_nrmse_pct"],
        "quant_boundary": VALIDATION_BOUNDARY,
        "traffic_16bit_words": traffic,
        "input_read_words": total_input_words,
        "container_bytes": container_bytes,
        "packed_bytes": packed,
        "container_moved_MiB": 12 * container_bytes / 2**20,
        "packed_moved_MiB": 12 * packed["total_bytes"] / 2**20,
        "onchip_words": words,
        "macro_plans": macro_plans,
        "cycles": {
            "L0_arithmetic": ideal,
            "L0_io_floor_16bit_words": ceil(container_bytes / 2),
            "L0_io_floor_128bit_beats": ceil(container_bytes / 16),
            "L1_staged_model": staged,
            "L2_e203_full_layer": l2,
        },
        "param_load": param_load(),
        "area_L3": synth,
        "output_stage_area_L3": OUTPUT_STAGE_AREA_UNITS,
    }


def param_load():
    """Per-layer parameter-load cost and 12-layer W reload total."""
    a_words = M * K
    w_words = K * N
    bias_words = N * 2  # INT32
    out_words = M * N * 2
    return {
        "per_layer_words": {"a": a_words, "w": w_words, "bias": bias_words, "out": out_words},
        "per_layer_total_words": a_words + w_words + bias_words + out_words,
        "per_layer_cycles_1word": a_words + w_words + bias_words + out_words,
        "per_layer_cycles_8word_128bit": ceil((a_words + w_words + bias_words + out_words) / 8),
        "w_reload_12layers_words": 12 * w_words,
        "w_container_12layers_KiB": 12 * w_words * 2 / 1024,
    }


def shared_sram_holds_all_layers():
    per_layer = {"a_KiB": M * K * 2 / 1024, "w_KiB": K * N * 2 / 1024,
                 "bias_KiB": N * 4 / 1024, "out_KiB": M * N * 4 / 1024}
    total = sum(per_layer.values())
    w_12 = 12 * per_layer["w_KiB"]
    return {
        "shared_sram_KiB": SHARED_SRAM_BYTES / 1024,
        "one_layer_all_operands_KiB": total,
        "w_12_layers_16bit_KiB": w_12,
        "w_12_layers_packed_KiB": {s: v["packed_w_kib_12"] for s, v in QUANT_SCHEMES.items()},
        "w_12_plus_a_bias_out_KiB": w_12 + per_layer["a_KiB"] * 12 + total - per_layer["w_KiB"],
    }


def build_report():
    return {
        "task": "FFN1 first linear layer only: Y[99,256]=X[99,64]@W[64,256]+b[256]",
        "clock_hz": CLK_HZ,
        "evidence_levels": {
            "L0_arithmetic": "parallelism / I/O floors, no schedule or platform timing",
            "L1_rtl_core": "Verilator staged task model; A/W/bias and bus are testbench-timed",
            "L2_e203_soc": "E203 NICE-like 208-tile model; abstract 128-bit bus + simulated SRAM",
            "L3_synthesis": "Nangate45 same-library relative cell area; no SRAM macro, no STA",
        },
        "self_check": self_check(),
        "shared_sram": shared_sram_holds_all_layers(),
        "configs": [analyse(c) for c in CONFIGS],
    }


def render_markdown(report):
    lines = ["# FFN1 whole-task trade-off (generated by ffn1_perf_storage_tradeoff.py)", ""]
    sc = report["self_check"]
    lines.append(f"Self-check: {sc['result']}, {sc['exact_gemm_comparisons']} exact GEMM "
                 f"comparisons, INT32 bound ok={sc['int32_bound_ok']}")
    lines.append("")
    lines.append("| config | tile | scheme | NRMSE % | input words | container B | L0 compute | L1 staged cyc | L2 E203 cyc |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c in report["configs"]:
        l1 = c["cycles"]["L1_staged_model"]["cycles"] if c["cycles"]["L1_staged_model"] else "-"
        l2 = c["cycles"]["L2_e203_full_layer"]["cycles"] if c["cycles"]["L2_e203_full_layer"] else "n/a"
        lines.append(
            f"| {c['name']} | {c['tile'][0]}x{c['tile'][1]} | {c['scheme']} | "
            f"{c['quant_worst_nrmse_pct']:.4f} | {c['input_read_words']:,} | "
            f"{c['container_bytes']:,} | {c['cycles']['L0_arithmetic']['row_broadcast_compute']:,} | "
            f"{l1} | {l2} |")
    lines.append("")
    lines.append(f"Parameter load per layer: {report['configs'][0]['param_load']['per_layer_total_words']:,} "
                 f"16-bit words ({report['configs'][0]['param_load']['per_layer_cycles_1word']:,} cyc @1/cyc, "
                 f"{report['configs'][0]['param_load']['per_layer_cycles_8word_128bit']:,} cyc @128-bit).")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, help="write the full JSON report here")
    p.add_argument("--markdown", type=Path, help="write a markdown summary table here")
    args = p.parse_args()
    try:
        report = build_report()
    except AssertionError as exc:
        p.error(f"self-check failed: {exc}")
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    md = render_markdown(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md)
    print(md, end="")
    if args.json:
        print(f"\nPASS wrote {args.json}")


if __name__ == "__main__":
    main()
