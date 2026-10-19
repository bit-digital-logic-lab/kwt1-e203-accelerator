#!/usr/bin/env python3
"""MODEL-ONLY whole-task event-schedule model for FFN1.

This is a software discrete-event scheduler, NOT RTL and NOT a measurement.
It exists to extend the RTL-proven ONE-LAYER schedule (hardware/experiments/
streaming_mac/tb/tb_kwt_streaming_task.sv) to the full 12-layer task, and to
show the schedule structure (compute / CACC export / output drain / reload).

Inputs (all explicitly labelled):
  * per-tile compute, CACC export (capture) and output drain are 16-bit/word
    bus models calibrated to the RTL-measured one-layer counts,
  * A/W/bias reload per layer is a word-count assumption (no E203/bus proof),
  * 150 MHz figures are conditional arithmetic on the model, not an STA result.

Resources modelled per layer: COMPUTE (one tile at a time), CAPTURE (CACC read
+ stage write), DRAIN (output bus). The single-CACC policy serialises
COMPUTE -> CAPTURE (the measured baseline). The dual-CACC policy lets CAPTURE
of tile T run concurrently with COMPUTE of tile T+1.

Run:
    python3 hardware/experiments/streaming_mac/model/event_schedule.py \
        --baseline-rtl 154240 --dual-rtl 125603
"""
import argparse
import json
import math
from pathlib import Path

M, K, N = 99, 64, 256
ROWS, COLS = 16, 16
WORDS_PER_RESULT = 2          # 16-bit slots per INT32 result
OUT_READY_DUTY = 2           # store bus accepts one result every 2 cycles


def tiles():
    row_tiles = math.ceil(M / ROWS)
    col_tiles = math.ceil(N / COLS)
    # TB order: for col tile, for row tile.
    out = []
    for ct in range(col_tiles):
        for rt in range(row_tiles):
            rows_v = min(ROWS, M - rt * ROWS)
            cols_v = min(COLS, N - ct * COLS)
            out.append({"rt": rt, "ct": ct, "valid": rows_v * cols_v})
    return out


def schedule(tile_list, compute, capture, dual):
    """Cycle-accurate event schedule for one layer.

    compute[t] / capture[t] are per-tile durations; drain[t] = valid*DUTY.
    Returns total cycles and the stage accounting.
    """
    core_free = [0, 0]                      # accumulator page availability
    compute_free = 0                        # COMPUTE resource (one tile at a time)
    stage_free = 0                          # output-stage buffer free
    bus_free = 0                            # DRAIN resource (output bus)
    total = 0
    cap_cycles = 0
    stall_cycles = 0

    for i, t in enumerate(tile_list):
        page = (i % 2) if dual else 0
        c_start = max(core_free[page], compute_free)
        c_end = c_start + compute
        compute_free = c_end
        # Capture of a tile can start only after its compute AND after the
        # stage buffer is free (previous capture+drain released the stage).
        ready = max(c_end, stage_free)
        if ready > c_end:
            stall_cycles += ready - c_end
        cap_start = ready
        cap_end = cap_start + capture
        core_free[page] = cap_end            # page occupied until exported
        stage_free = cap_end
        cap_cycles += capture
        dr_end = cap_end + t["valid"] * OUT_READY_DUTY
        bus_free = dr_end
        total = max(total, c_end, cap_end, dr_end)
    return {"total": total, "capture_cycles": cap_cycles,
            "stall_cycles": stall_cycles}


def build_model(baseline_rtl, dual_rtl):
    tl = tiles()
    n = len(tl)
    initial_load = M * K + K * COLS + 2 * N          # 7872 words @1/cycle
    # Calibrate flat per-tile compute so the SINGLE-CACC model matches the
    # RTL-measured baseline one-layer count. Compute is schedule-independent.
    # baseline total = load + n*(compute+capture) + last_drain + overhead
    # solve for compute given a measured baseline, keeping capture = ROWS*COLS.
    capture_tile = ROWS * COLS
    last_valid = tl[-1]["valid"]
    # model baseline (no overhead) with compute unknown:
    # total = initial_load + sum_i(compute) + sum_i(capture) + last_drain
    # (drain overlaps next compute; the last drain cannot overlap)
    # -> compute = (baseline_rtl - initial_load - sum_capture - last_drain) / n
    sum_capture = captotal = n * capture_tile
    last_drain = last_valid * OUT_READY_DUTY
    compute_tile = (baseline_rtl - initial_load - sum_capture - last_drain) / n

    def layer(dual):
        s = schedule(tl, compute_tile, capture_tile, dual)
        return {"layer_cycles": initial_load + s["total"],
                "initial_load_cycles": initial_load,
                "compute_cycles": int(round(compute_tile * n)),
                "capture_cycles": s["capture_cycles"],
                "stall_cycles": s["stall_cycles"],
                "last_tile_drain_cycles": last_drain,
                "per_tile_compute": compute_tile}

    base = layer(False)
    dual = layer(True)
    return {
        "label": "MODEL-ONLY event schedule; NOT RTL, NOT measured",
        "genuine_prediction": "dual-CACC one-layer cycles are predicted by the "
                              "model (compute calibrated only from the baseline RTL)",
        "calibrated_from": {"baseline_rtl_one_layer_cycles": baseline_rtl},
        "rtl_reference": {"baseline": baseline_rtl, "dual": dual_rtl},
        "tiles": n,
        "initial_load_words": initial_load,
        "per_tile_compute_calibrated": round(compute_tile, 3),
        "per_tile_capture": capture_tile,
        "one_layer": {"baseline_model": base["layer_cycles"],
                      "dual_model": dual["layer_cycles"],
                      "baseline_rtl": baseline_rtl,
                      "dual_rtl": dual_rtl,
                      "baseline_fit_pct": round(100.0 * (base["layer_cycles"] - baseline_rtl)
                                                / baseline_rtl, 3),
                      "dual_prediction_error_pct": round(100.0 * (dual["layer_cycles"] - dual_rtl)
                                                         / dual_rtl, 3)},
        "twelve_layers": {
            "assumption": "12 identical FFN1 layers back-to-back, each charging the "
                          "same per-layer load; inter-layer E203/bus overhead = 0 "
                          "(not modelled, not proven).",
            "baseline_model_cycles": 12 * base["layer_cycles"],
            "dual_model_cycles": 12 * dual["layer_cycles"],
            "speedup": round(base["layer_cycles"] / dual["layer_cycles"], 4),
            "baseline_model_ms_at_150mhz": round(12 * base["layer_cycles"] / 150e6 * 1e3, 4),
            "dual_model_ms_at_150mhz": round(12 * dual["layer_cycles"] / 150e6 * 1e3, 4),
        },
        "not_proven": ["E203 command/reprogram latency", "shared-bus arbitration",
                       "target SRAM macro ports/latency", "150 MHz timing closure",
                       "12-layer quantised numerics"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-rtl", type=int, default=154240)
    ap.add_argument("--dual-rtl", type=int, default=125603)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "model_12layer.json")
    args = ap.parse_args()
    model = build_model(args.baseline_rtl, args.dual_rtl)
    args.out.write_text(json.dumps(model, indent=2) + "\n")
    print(json.dumps(model, indent=2))
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
