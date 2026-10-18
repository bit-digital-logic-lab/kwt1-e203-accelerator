#!/usr/bin/env python3
"""Parse the power-activity VCDs and verify they are usable activity evidence.

Checks (per tile VCD):
  * the waveform is non-empty and has a value-change section;
  * the clock toggles;
  * the modeled accelerator/tile/array/cacc/DMA scopes have transitions;
  * the observed window start/end match the E203 NICE request and the final
    NICE response, and the derived task cycle count matches the simulation log.

It does NOT convert activity to power or energy. Without a PDK, interconnect
power model, or SRAM macro power model those figures are recorded as
"not_computable".
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


# Module-scope substrings that must show switching activity.
KEY_GROUPS = {
    "accelerator_nice_fsm": ".accelerator.",
    "tile_system_load_compute_writeback": ".accelerator.tile_system.",
    "compute_engine": ".accelerator.tile_system.tile.",
    "product_array": ".accelerator.tile_system.tile.array.",
    "cacc": ".accelerator.tile_system.tile.cacc.",
    "dma": ".accelerator.tile_system.dma.",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class VcdReader:
    """Streaming VCD reader that keeps only what the checks need."""

    def __init__(self, path):
        self.path = path
        self.vars = {}          # id -> {"path", "name", "width"}
        self.scope = []
        self.toggles = {}       # id -> change count
        self.last = {}          # id -> last raw value string
        self.series = {}        # id -> list of (time, value)
        self.watch = set()
        self.clock_period = None
        self.max_time = 0
        self.value_changes = 0
        self.had_enddefinitions = False
        self._clock_prev_time = None
        self._clock_min_delta = None

    def _full_path(self, name):
        return ".".join(self.scope + [name])

    def parse(self):
        time = 0
        with self.path.open("r", encoding="ascii", errors="replace") as fh:
            in_defs = True
            for raw in fh:
                line = raw.rstrip("\n")
                if in_defs:
                    if line.startswith("$scope"):
                        parts = line.split()
                        if len(parts) >= 3:
                            self.scope.append(parts[2])
                    elif line.startswith("$upscope"):
                        if self.scope:
                            self.scope.pop()
                    elif line.startswith("$var"):
                        parts = line.split()
                        if len(parts) >= 5:
                            width = int(parts[2])
                            ident = parts[3]
                            name = parts[4]
                            full = self._full_path(name)
                            self.vars[ident] = {"path": full, "name": name,
                                                "width": width}
                            self.toggles[ident] = 0
                    elif line.startswith("$enddefinitions"):
                        in_defs = False
                        self.had_enddefinitions = True
                    continue
                if not line:
                    continue
                if line[0] == "#":
                    try:
                        time = int(line[1:])
                    except ValueError:
                        pass
                    if time > self.max_time:
                        self.max_time = time
                    continue
                self.value_changes += 1
                token = line.split()
                if not token:
                    continue
                first = token[0]
                if first[0] in "01xzXZ":
                    ident = first[1:]
                    value = first[0]
                elif first[0] in "bBrR":
                    if len(token) < 2:
                        continue
                    ident = token[1]
                    value = first[1:] if first[0] in "bB" else first
                else:
                    continue
                self.toggles[ident] = self.toggles.get(ident, 0) + 1
                self.last[ident] = value
                self._record(time, ident, value)

    def _record(self, time, ident, value):
        info = self.vars.get(ident)
        if info is None:
            return
        name = info["name"]
        if name == "clk" and info["path"].endswith("tb_kwt_power_activity.clk"):
            if self._clock_prev_time is not None:
                delta = time - self._clock_prev_time
                if delta > 0 and (self._clock_min_delta is None or delta < self._clock_min_delta):
                    self._clock_min_delta = delta
            self._clock_prev_time = time
        if ident in self.watch:
            self.series.setdefault(ident, []).append((time, value))

    def watch_names(self, names):
        for ident, info in self.vars.items():
            if info["name"] in names:
                self.watch.add(ident)

    def series_for(self, name):
        for ident, info in self.vars.items():
            if info["name"] == name and ident in self.series:
                return self.series[ident]
        return []

    def series_by_path_suffix(self, suffix):
        for ident, info in self.vars.items():
            if info["path"].endswith(suffix) and ident in self.series:
                return self.series[ident]
        return []

    def id_of(self, name):
        for ident, info in self.vars.items():
            if info["name"] == name:
                return ident
        return None


def first_both_one(vcd, sig_a, sig_b):
    sa = {t: v for t, v in vcd.series_for(sig_a)}
    sb = {t: v for t, v in vcd.series_for(sig_b)}
    a = b = "0"
    for t in sorted(set(sa) | set(sb)):
        if t in sa:
            a = sa[t]
        if t in sb:
            b = sb[t]
        if a == "1" and b == "1":
            return t
    return None


def last_both_one(vcd, sig_a, sig_b):
    sa = dict(vcd.series_for(sig_a))
    sb = dict(vcd.series_for(sig_b))
    last = None
    a = b = "0"
    for t in sorted(set(sa) | set(sb)):
        if t in sa:
            a = sa[t]
        if t in sb:
            b = sb[t]
        if a == "1" and b == "1":
            last = t
    return last


def first_one(vcd, name):
    for t, v in vcd.series_for(name):
        if v == "1":
            return t
    return None


def decode_bin(value):
    text = value.lstrip("bB")
    if not text or set(text) - set("01"):
        return None
    return int(text, 2)


# kwt_tile_system_model state_t: IDLE, LOAD_A, WAIT_A, LOAD_B, WAIT_B,
# LOAD_BIAS, WAIT_BIAS, BIAS_REQ, BIAS_CAPTURE, RUN_TILE, WAIT_TILE,
# STORE_OUT, WAIT_STORE, RESPONSE
def phase_of_state(code):
    if code is None:
        return None
    if 1 <= code <= 8:
        return "load"
    if 9 <= code <= 10:
        return "compute"
    if 11 <= code <= 12:
        return "writeback"
    return None


def phase_windows(vcd):
    transitions = []
    for time, value in vcd.series_by_path_suffix(".accelerator.tile_system.state"):
        transitions.append((time, phase_of_state(decode_bin(value))))
    transitions.sort()
    windows = {}
    prev_phase = None
    prev_start = None
    for time, phase in transitions:
        if phase != prev_phase:
            if prev_phase in ("load", "compute", "writeback"):
                if prev_phase in windows:
                    windows[prev_phase][1] = time
                else:
                    windows[prev_phase] = [prev_start, time]
            prev_phase = phase
            prev_start = time
    return windows


def parse_log(path):
    text = path.read_text(errors="replace")
    result = {"pass": "PASS" in text, "text_tail": text.strip().splitlines()[-1] if text.strip() else ""}
    m = re.search(r"task_cycles=(\d+)", text)
    if m:
        result["task_cycles"] = int(m.group(1))
    m = re.search(r"total_cycles=(\d+)", text)
    if m:
        result["total_cycles"] = int(m.group(1))
    m = re.search(r"outputs=(\d+)", text)
    if m:
        result["outputs"] = int(m.group(1))
    m = re.search(r"padding=(\d+)", text)
    if m:
        result["padding"] = int(m.group(1))
    m = re.search(r"first_req_cycle=(-?\d+)", text)
    if m:
        result["first_req_cycle"] = int(m.group(1))
    m = re.search(r"last_rsp_cycle=(-?\d+)", text)
    if m:
        result["last_rsp_cycle"] = int(m.group(1))
    return result


def analyze(vcd_path, log_path):
    vcd = VcdReader(vcd_path)
    vcd.parse()
    vcd.watch_names({
        "nice_req_valid", "nice_req_ready", "nice_rsp_valid", "nice_rsp_ready",
        "marker_first_req", "marker_last_rsp", "state",
    })
    # Re-parse to capture watch series (ids discovered only after parse).
    vcd.toggles = {k: 0 for k in vcd.vars}
    vcd.series = {}
    vcd._clock_prev_time = None
    vcd._clock_min_delta = None
    vcd.parse()

    clk_id = vcd.id_of("clk")
    clk_toggles = vcd.toggles.get(clk_id, 0) if clk_id else 0
    half = vcd._clock_min_delta
    period = (2 * half) if half else None

    scope_toggles = {group: 0 for group in KEY_GROUPS}
    toggled_sigs = []
    untoggled_sigs = []
    for ident, info in vcd.vars.items():
        n = vcd.toggles.get(ident, 0)
        path = info["path"]
        for group, token in KEY_GROUPS.items():
            if token in path:
                scope_toggles[group] += n
        (toggled_sigs if n > 0 else untoggled_sigs).append(path)

    # Window via explicit tb markers, cross-checked with raw NICE handshakes.
    w_start_marker = first_one(vcd, "marker_first_req")
    w_end_marker = first_one(vcd, "marker_last_rsp")
    w_start_req = first_both_one(vcd, "nice_req_valid", "nice_req_ready")
    w_end_rsp = last_both_one(vcd, "nice_rsp_valid", "nice_rsp_ready")

    start = w_start_marker if w_start_marker is not None else w_start_req
    end = w_end_marker if w_end_marker is not None else w_end_rsp
    task_cycles_vcd = None
    if start is not None and end is not None and period:
        task_cycles_vcd = (end - start) // period + 1

    log = parse_log(log_path) if log_path.exists() else {}
    window_start_ok = (start is not None and w_start_req is not None and
                       abs(start - w_start_req) <= period if period else False)
    window_end_ok = (end is not None and w_end_rsp is not None and
                     abs(end - w_end_rsp) <= period if period else False)
    cycles_ok = (task_cycles_vcd is not None and
                 "task_cycles" in log and
                 abs(task_cycles_vcd - log["task_cycles"]) <= 1)

    phases = phase_windows(vcd)
    phase_order_ok = False
    if {"load", "compute", "writeback"} <= set(phases) and start is not None and end is not None:
        load_s = phases["load"][0]
        comp_s = phases["compute"][0]
        wb_e = phases["writeback"][1]
        phase_order_ok = (start <= load_s <= comp_s <= wb_e <= end)

    non_empty = (vcd.had_enddefinitions and len(vcd.vars) > 0 and
                 vcd.value_changes > 0 and clk_toggles >= 2)
    key_modules_ok = all(count > 0 for count in scope_toggles.values())

    checks = {
        "waveform_non_empty": bool(non_empty),
        "clock_toggles": bool(clk_toggles >= 2),
        "key_modules_toggle": bool(key_modules_ok),
        "window_start_matches_first_nice_request": bool(window_start_ok),
        "window_end_matches_final_nice_response": bool(window_end_ok),
        "load_compute_writeback_phases_present": bool(
            {"load", "compute", "writeback"} <= set(phases)),
        "load_compute_writeback_in_window_order": bool(phase_order_ok),
        "task_cycles_match_log": bool(cycles_ok),
        "log_pass": bool(log.get("pass", False)),
    }

    return {
        "vcd": str(vcd_path),
        "vcd_bytes": vcd_path.stat().st_size,
        "vcd_sha256": sha256(vcd_path),
        "log": str(log_path),
        "log_summary": log,
        "timescale_units_max_time": vcd.max_time,
        "clock_half_period_time_units": half,
        "clock_period_time_units": period,
        "clock_toggles": clk_toggles,
        "signals_defined": len(vcd.vars),
        "signals_toggled": len(toggled_sigs),
        "toggled_coverage_pct": round(100.0 * len(toggled_sigs) / max(len(vcd.vars), 1), 3),
        "coverage_scope": ("curated control/bus signals of the modeled E203 NICE path; "
                           "large behavioral SRAM arrays and the unannotated CPU datapath "
                           "are excluded from the dump, so this is not full-design coverage"),
        "value_changes": vcd.value_changes,
        "scope_toggles": scope_toggles,
        "phase_windows_time_units": phases,
        "window_start_time": start,
        "window_end_time": end,
        "window_start_from_raw_nice_req": w_start_req,
        "window_end_from_raw_nice_rsp": w_end_rsp,
        "task_cycles_from_vcd": task_cycles_vcd,
        "untoggled_signals": sorted(untoggled_sigs)[:50],
        "checks": checks,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True,
                    help="directory holding *_tile???.vcd and .log files")
    ap.add_argument("--out", type=Path, default=None,
                    help="summary JSON path (default: <results>/summary.json)")
    args = ap.parse_args()

    vcds = sorted(args.results.glob("*_tile*.vcd"))
    if not vcds:
        print(f"no VCD files in {args.results}", file=sys.stderr)
        return 1

    entries = []
    ok = True
    for vcd_path in vcds:
        log_path = vcd_path.with_suffix(".log")
        entry = analyze(vcd_path, log_path)
        entries.append(entry)
        failed = [name for name, value in entry["checks"].items() if not value]
        print(f"[{vcd_path.name}] bytes={entry['vcd_bytes']} "
              f"clk_toggles={entry['clock_toggles']} period={entry['clock_period_time_units']} "
              f"task_cycles_vcd={entry['task_cycles_from_vcd']} "
              f"task_cycles_log={entry['log_summary'].get('task_cycles')} "
              f"coverage={entry['toggled_coverage_pct']}%")
        for group, count in entry["scope_toggles"].items():
            print(f"    {group}: {count} toggles")
        pw = entry["phase_windows_time_units"]
        phase_str = ", ".join(f"{k}={v[1]-v[0]+1}u" for k, v in pw.items())
        print(f"    window=({entry['window_start_time']}..{entry['window_end_time']}) "
              f"phases: {phase_str}")
        if failed:
            ok = False
            print(f"    FAILED checks: {', '.join(failed)}")

    summary = {
        "evidence": "E203 NICE-like 8x16 A12W10 activity VCD",
        "power_energy": {
            "value": "not_computable",
            "reason": ("No competition PDK, no interconnect power model, and no "
                       "SRAM macro power model are available. Unannotated circuits "
                       "and unmodeled SRAM cannot be counted as zero activity. This "
                       "artifact is switching activity (VCD) only."),
        },
        "entries": entries,
        "verdict": "PASS" if ok else "FAIL",
    }
    out = args.out or (args.results / "summary.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"summary -> {out} : {summary['verdict']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
