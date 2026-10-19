#!/usr/bin/env python3
"""Cycle-accurate shared-bus / stage event model for the bias-overlap schedule.

Two things are deliberately separated:

1. ``baseline`` and ``dual`` are calibrated arithmetic/event reproductions of
   the accepted RTL task harnesses.  Constants were fitted to the accepted RTL
   (full FFN1) and are then reused unchanged:
     * initial load = A words + bias slots + first W page words
     * per-tile wall = 2 clear + ROWS A reads + K*COLS MAC + 5 tail
     * output-stage capture = 259 cycles, drain = 2 cycles per valid INT32
       (the 16-bit bus accepts one slot per cycle, a result needs two slots)
   The full FFN1 model reproduces 154240 (baseline) and 125603 (dual) exactly.

2. ``bias_overlap`` is a cycle-level event simulation of the new one-descriptor
   schedule: one 16-bit shared bus, one grant per cycle, fixed priority
   OUT > BIAS > A > W, A row-tile streaming, W page prefetch, bias group
   prefetch, dual CACC pages, capture/drain output stage.  This is a MODEL;
   the RTL measurement is reported separately by run_bias_overlap.py.

No model path claims two requesters move data on the shared bus in one cycle.
Local SRAM ports (compute A/W/CACC reads) are not on the shared bus and are
allowed to operate concurrently with one bus transfer.
"""
import argparse
import json
import math
from pathlib import Path

ROWS = 16
COLS = 16
CAPTURE_LATENCY = 259


def ceil_div(a, b):
    return (a + b - 1) // b


def compute_tile_cycles(k, rows=ROWS, gaps=False):
    """Wall cycles for one tile, including 2 clear + ROWS A reads + tail 5."""
    mac = k * COLS
    extra = ((mac + 16) // 17) if gaps else 0   # TB gap at index 0,17,34,...
    return 2 + rows + mac + 5 + extra


def tile_geometry(m, n, rows=ROWS, cols=COLS):
    rt = ceil_div(m, rows)
    ct = ceil_div(n, cols)
    return rt, ct, rt * ct


def tile_outputs(m, n, idx, rows=ROWS, cols=COLS):
    rt, ct, _ = tile_geometry(m, n, rows, cols)
    r = idx % rt
    c = idx // rt
    rv = min(rows, m - r * rows)
    cv = min(cols, n - c * cols)
    return max(0, rv) * max(0, cv)


def initial_load(m, k, n):
    # A resident + all bias values + first W page (same serial preload as the
    # accepted harnesses).
    return m * k + 2 * n + k * COLS


def baseline_cycles(m, k, n, rows=ROWS, gaps=False):
    """Accepted single-CACC staged task: compute then serial capture."""
    _, _, tiles = tile_geometry(m, n, rows)
    comp = compute_tile_cycles(k, rows, gaps)
    t = initial_load(m, k, n)
    stage_free = 0
    for idx in range(tiles):
        compute_end = t + comp
        cap_start = max(compute_end, stage_free)
        # A capture that had to wait one cycle for the stage adds a handshake
        # cycle in the accepted harness.
        cap_lat = CAPTURE_LATENCY + (1 if cap_start > compute_end else 0)
        cap_done = cap_start + cap_lat
        stage_free = cap_done + 2 * tile_outputs(m, n, idx, rows)
        t = cap_done
    return stage_free


def dual_cycles(m, k, n, rows=ROWS, gaps=False):
    """Accepted dual-CACC task: capture of tile T under compute of tile T+1."""
    _, _, tiles = tile_geometry(m, n, rows)
    comp = compute_tile_cycles(k, rows, gaps)
    t = initial_load(m, k, n)
    compute_end = t + comp
    stage_free = 0
    for idx in range(1, tiles):
        cap_start = max(compute_end + 1, stage_free)
        cap_done = cap_start + CAPTURE_LATENCY
        stage_free = cap_done + 2 * tile_outputs(m, n, idx - 1, rows)
        compute_end = cap_start + comp
    # Tail: export the last tile and drain it.
    cap_start = max(compute_end + 1, stage_free)
    return cap_start + CAPTURE_LATENCY + 2 * tile_outputs(m, n, tiles - 1, rows)


def bias_overlap_cycles(m, k, n, rows=ROWS, preload_tiles=None, bias_delay=0,
                        out_bp=1, bus_gap=0, gaps=False, trace=False):
    """Cycle-level shared-bus event model of the new schedule.

    One shared 16-bit slot is granted per enabled cycle.  OUT > BIAS > A > W.
    The model returns total cycles from task start (reset release) to the last
    accepted output.  ``out_bp`` gates output slot requests (1 = one request
    every cycle, the normal 2-slots-per-result behaviour).
    """
    rt, ct, tiles = tile_geometry(m, n, rows)
    if preload_tiles is None:
        preload_tiles = rt
    preload_tiles = max(1, min(preload_tiles, rt))
    a_total = m * k
    a_preload = min(preload_tiles * rows, m) * k
    comp = compute_tile_cycles(k, rows, gaps)

    # loader state
    a_loaded = 0
    w_group, w_idx, w_active = 0, 0, False
    w_valid = [False, False]
    w_busy = [False, False]
    w_group_of = [0, 0]
    w_done = [False] * ct
    bias_group, bias_idx, bias_half, bias_active = 0, 0, 0, False
    bias_valid = [False] * ct

    # compute
    comp_tile = 0
    comp_busy = False
    comp_remaining = 0
    compute_started = False
    compute_done = False
    compute_col = 0
    cacc_busy = [False, False]
    pending = [None, None]       # (tile, ct, outputs)
    exp_tile = 0
    stage_state = 0              # 0 idle, 1 capture, 2 drain
    stage_remaining = 0
    drain_slots = 0
    drain_outputs = 0

    cyc = 0
    stalls = {"load": 0, "cacc": 0, "bias": 0, "stage": 0, "out_bp": 0}
    bus = {"out": 0, "bias": 0, "a": 0, "w": 0, "idle": 0}

    def a_row_ready(r):
        need = min((r + 1) * rows, m)
        return a_loaded >= need * k

    def w_page_ready(c):
        p = c % 2
        return w_valid[p] and w_group_of[p] == c

    def w_page_free(p):
        if w_busy[p]:
            return False
        if not w_valid[p]:
            return True
        return w_done[w_group_of[p]]

    while True:
        cyc += 1
        bus_enable = (bus_gap <= 1) or (cyc % bus_gap != 0)

        # Starters (a request may be asserted in the cycle the job starts).
        if not w_active and w_group < ct and w_page_free(w_group % 2):
            w_active = True
            w_group_of[w_group % 2] = w_group
            w_valid[w_group % 2] = False
            w_idx = 0
        if not bias_active and bias_group < ct:
            allowed = (compute_col + 1) if compute_started else 0
            if bias_group <= allowed:
                bias_active = True
                bias_idx = 0
                bias_half = 0

        req_out = (stage_state == 2 and drain_slots > 0 and
                   (out_bp <= 1 or cyc % out_bp == 0))
        req_bias = bias_active and bias_group < ct and cyc >= bias_delay
        req_a = (a_loaded < a_preload) or (compute_started and a_loaded < a_total)
        req_w = w_active
        if stage_state == 2 and drain_slots > 0 and not req_out:
            stalls["out_bp"] += 1

        # One-hot fixed priority.
        if not bus_enable:
            grant = None
        elif req_out:
            grant = "out"
        elif req_bias:
            grant = "bias"
        elif req_a:
            grant = "a"
        elif req_w:
            grant = "w"
        else:
            grant = None
        if grant is None:
            bus["idle"] += 1

        if grant == "out":
            bus["out"] += 1
            drain_slots -= 1
            if drain_slots == 0:
                stage_state = 0
        elif grant == "bias":
            bus["bias"] += 1
            if bias_half:
                bias_half = 0
                if bias_idx == COLS - 1:
                    bias_valid[bias_group] = True
                    bias_active = False
                    bias_group += 1
                else:
                    bias_idx += 1
            else:
                bias_half = 1
        elif grant == "a":
            bus["a"] += 1
            a_loaded += 1
        elif grant == "w":
            bus["w"] += 1
            w_idx += 1
            if w_idx == k * COLS:
                w_valid[w_group % 2] = True
                w_active = False
                w_group += 1

        # Compute engine.
        if not compute_done:
            if comp_busy:
                comp_remaining -= 1
                if comp_remaining <= 0:
                    comp_busy = False
                    p = comp_tile % 2
                    pending[p] = (comp_tile, compute_col,
                                  tile_outputs(m, n, comp_tile, rows))
                    if comp_tile % rt == rt - 1:
                        w_done[compute_col] = True
                        w_busy[compute_col % 2] = False
                    comp_tile += 1
            elif comp_tile < tiles:
                c, r = comp_tile // rt, comp_tile % rt
                if not compute_started:
                    if a_loaded >= a_preload and a_row_ready(0) and w_page_ready(0):
                        compute_started = True
                        compute_col = 0
                        cacc_busy[0] = True
                        w_busy[0] = True
                        comp_busy = True
                        comp_remaining = comp
                    else:
                        stalls["load"] += 1
                elif cacc_busy[comp_tile % 2]:
                    stalls["cacc"] += 1
                elif not a_row_ready(r) or not w_page_ready(c):
                    stalls["load"] += 1
                else:
                    compute_col = c
                    cacc_busy[comp_tile % 2] = True
                    w_busy[c % 2] = True
                    comp_busy = True
                    comp_remaining = comp
            else:
                compute_done = True

        # Export engine: start the oldest pending tile if stage idle + bias ok.
        if stage_state == 0 and exp_tile < tiles:
            p = exp_tile % 2
            if pending[p] is not None and pending[p][0] == exp_tile:
                _, pc, nout = pending[p]
                if not bias_valid[pc]:
                    stalls["bias"] += 1
                else:
                    pending[p] = None
                    stage_state = 1
                    stage_remaining = CAPTURE_LATENCY
                    drain_outputs = nout
                    drain_slots = 2 * nout
                    # CACC page is freed at capture_done, not at start.
        elif stage_state == 0 and exp_tile >= tiles:
            pass
        elif stage_state != 0 and exp_tile < tiles:
            p = exp_tile % 2
            if pending[p] is not None and pending[p][0] == exp_tile:
                stalls["stage"] += 1

        # Stage capture/drain progress.
        if stage_state == 1:
            stage_remaining -= 1
            if stage_remaining <= 0:
                p = exp_tile % 2
                cacc_busy[p] = False
                exp_tile += 1
                if drain_slots > 0:
                    stage_state = 2
                else:
                    stage_state = 0
        # DRAIN is progressed by OUT grants above.

        if trace and (cyc % 10000 == 0):
            print(f"  cycle {cyc}: a={a_loaded} w={w_group} bias={bias_group} "
                  f"tile={comp_tile} exp={exp_tile} stage={stage_state}")

        if (compute_done and exp_tile == tiles and stage_state == 0 and
                drain_slots == 0):
            break

    return {
        "cycles": cyc,
        "initial_load_cycles": None,  # filled by the RTL; model keeps bus view
        "compute_wall_cycles": None,
        "tail_cycles": None,
        "a_preload_tiles": preload_tiles,
        "bias_delay": bias_delay,
        "out_bp": out_bp,
        "bus_gap": bus_gap,
        "stalls": stalls,
        "bus": bus,
        "mac_cycles_model": comp * tiles,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "model_results.json")
    ap.add_argument("--rtl-results", type=Path, default=None,
                    help="optional results.json from the RTL runner")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    accepted = {
        "baseline_full": 154240,
        "dual_full": 125603,
    }
    cases = [
        ("tiny", 3, 4, 5, False, 462, 463),
        ("ragged", 10, 7, 19, True, 1262, 1261),
        ("twotile", 32, 8, 16, True, 2118, 2119),
        ("overflow", 8, 64, 16, True, 3192, 3191),
        ("full", 99, 64, 256, False, 154240, 125603),
    ]
    out = {"label": "MODEL ONLY - calibrated to accepted RTL, not a new RTL run",
           "constants": {"rows": ROWS, "cols": COLS,
                         "capture_latency": CAPTURE_LATENCY,
                         "compute_tile": "2 clear + ROWS A reads + K*COLS MAC + 5 tail",
                         "drain": "2 slots (cycles) per valid INT32 result",
                         "bus": "one 16-bit slot/cycle; OUT>BIAS>A>W"},
           "baseline_dual": [], "bias_overlap_sweep": [], "bias_overlap_cases": []}

    for name, m, k, n, gaps, rtl_base, rtl_dual in cases:
        b = baseline_cycles(m, k, n, gaps=gaps)
        d = dual_cycles(m, k, n, gaps=gaps)
        out["baseline_dual"].append({
            "case": name, "m": m, "k": k, "n": n, "gaps": gaps,
            "baseline_model": b, "baseline_rtl": rtl_base,
            "baseline_delta": b - rtl_base,
            "dual_model": d, "dual_rtl": rtl_dual,
            "dual_delta": d - rtl_dual,
        })
        print(f"{name:8s} baseline model={b} rtl={rtl_base} d={b-rtl_base:+d} | "
              f"dual model={d} rtl={rtl_dual} d={d-rtl_dual:+d}")

    # Preload sweep for full FFN1 (the measured RTL cases are reported later).
    for pre in range(1, 8):
        r = bias_overlap_cycles(99, 64, 256, preload_tiles=pre)
        out["bias_overlap_sweep"].append({"a_preload_tiles": pre, **r})
        print(f"bias-overlap full preload={pre}: model={r['cycles']} "
              f"stalls={r['stalls']} bus={r['bus']}")

    # Delayed / back-pressure model cases for the full shape.
    for label, kwargs in [
        ("delayed_bias_6000", dict(preload_tiles=3, bias_delay=6000)),
        ("delayed_bias_20000", dict(preload_tiles=3, bias_delay=20000)),
        ("out_backpressure_3", dict(preload_tiles=3, out_bp=3)),
        ("bus_gap_7", dict(preload_tiles=3, bus_gap=7)),
    ]:
        r = bias_overlap_cycles(99, 64, 256, **kwargs)
        out["bias_overlap_cases"].append({"case": label, **r})
        print(f"{label}: model={r['cycles']} stalls={r['stalls']}")

    if args.rtl_results and args.rtl_results.exists():
        rtl = json.loads(args.rtl_results.read_text())
        out["rtl_measurements"] = {
            "baseline_full": rtl.get("comparison", {}).get("baseline_full"),
            "accepted_dual_full": rtl.get("comparison", {}).get("accepted_dual_full"),
            "bias_overlap_full": rtl.get("comparison", {}).get("bias_overlap_full"),
            "bias_overlap_delayed": rtl.get("comparison", {}).get("bias_overlap_delayed"),
            "bias_overlap_out_bp": rtl.get("comparison", {}).get("bias_overlap_out_bp"),
        }

    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
