#!/usr/bin/env python3
"""Reproducible Verilator differential/boundary regression for KWT-1 FFN1.

Scope (bachelor-required work: 12 independent FFN1 layers, each a
99x64 @ 64x256 + bias linear layer):

* Builds the real tile engine ``kwt_pingpong_cacc_engine`` (8x16 lanes,
  INT16 A/W, INT32 accumulator/bias/output) with Verilator once.
* For every one of the frozen 12 A12W10 layers it drives three tiles:
  first tile, a middle tile, and the last (partial) row tile with output
  padding.
* Adds explicit boundary cases: K=1 reduction, column-mask output padding,
  ReLU output masking, reset injected mid-compute, and a ping-pong
  back-pressure handshake.
* Expected values come from an independent NumPy integer GEMM computed
  from the same frozen weights/bias and stimulus, never from the DUT.
* A negative injection corrupts one expected element and proves the
  comparator reports a mismatch (the self-check can fail).

Activations are a deterministic synthetic pattern (documented below). The
weights and biases are the frozen per-layer A12W10 parameters; no network,
MFCC dataset or software model is required.

Run:  python3 run_extended_verilator.py [--out DIR] [--verilator PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROWS = 8
COLS = 16
ACC_W = 32
M, K, N = 99, 64, 256
DEFAULT_PARAMS = REPO / "software/results/ffn1_a12w10_static_sym.npz"

RTL = REPO / "hardware/rtl"
SOURCES = [
    RTL / "pulse_array/kwt_product_array.sv",
    RTL / "pulse_array/kwt_cacc.sv",
    RTL / "sram/kwt_operand_pingpong.sv",
    RTL / "pulse_array/kwt_pingpong_cacc_engine.sv",
    HERE / "tb_kwt_extended_regression.sv",
]
TOP = "tb_kwt_extended_regression"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def synthetic_activations(layer: int) -> np.ndarray:
    """Deterministic 12-bit activation stimulus for one layer.

    This is a labelled synthetic pattern, not a real audio activation; it only
    needs to be inside the A12W10 activation range so the integer model and the
    RTL exercise the same well-defined arithmetic.
    """
    rows = np.arange(M, dtype=np.int64)[:, None]
    cols = np.arange(K, dtype=np.int64)[None, :]
    return ((rows * 97 + cols * 53 + layer * 29) % 4095 - 2047).astype(np.int64)


def build(verilator: str, build_dir: Path) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    obj = build_dir / "obj"
    command = [
        verilator, "--binary", "--timing", "--assert", "-Wall", "-Wno-fatal",
        "--top-module", TOP, f"-I{RTL / 'sram'}", "--Mdir", str(obj),
        *[str(src) for src in SOURCES],
    ]
    log = build_dir / "build.log"
    with log.open("w") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.stderr.write("Verilator build failed:\n" + log.read_text())
        raise SystemExit(result.returncode)
    binary = obj / f"V{TOP}"
    if not binary.exists():
        raise SystemExit(f"missing Verilator binary {binary}")
    return binary


def write_hex(path: Path, values, bits: int) -> None:
    mask = (1 << bits) - 1
    digits = bits // 4
    with path.open("w", encoding="ascii") as handle:
        for value in np.asarray(values).reshape(-1):
            handle.write(f"{int(value) & mask:0{digits}x}\n")


def make_case(name, layer, qw, qb, qx, row0, col0, row_mask, col_mask,
              relu, mode, klen=K):
    klen = int(klen)
    a_words = np.zeros((klen, ROWS), dtype=np.int64)
    for r in range(ROWS):
        if row0 + r < M:
            a_words[:, r] = qx[row0 + r, :klen]
    b_words = np.zeros((klen, COLS), dtype=np.int64)
    for c in range(COLS):
        if col0 + c < N:
            b_words[:, c] = qw[:klen, col0 + c]
    bias_words = np.zeros(COLS, dtype=np.int64)
    for c in range(COLS):
        if col0 + c < N:
            bias_words[c] = qb[col0 + c]

    expected = np.zeros((ROWS, COLS), dtype=np.int64)
    for r in range(ROWS):
        if not (row_mask >> r) & 1:
            continue
        for c in range(COLS):
            if not (col_mask >> c) & 1:
                continue
            if row0 + r >= M or col0 + c >= N:
                continue
            total = int(np.dot(a_words[:, r], b_words[:, c])) + int(bias_words[c])
            if relu and total < 0:
                total = 0
            expected[r, c] = total
    # a_words is [k][r]; flatten k-major as the RTL address expects.
    case = {
        "name": name, "layer": layer, "mode": mode, "klen": klen,
        "row0": row0, "col0": col0, "row_mask": row_mask, "col_mask": col_mask,
        "relu": int(relu),
        "a": a_words.reshape(-1), "b": b_words.reshape(-1), "bias": bias_words,
        "expected": expected,
    }
    return case


def build_cases(params: Path):
    with np.load(params, allow_pickle=False) as data:
        cases = []
        names = []
        for layer in range(1, 13):
            stem = f"layer{layer:02}"
            qw = data[stem + "_weight"].astype(np.int64)   # (64,256)
            qb = data[stem + "_bias"].astype(np.int64)     # (256,)
            qx = synthetic_activations(layer)              # (99,64)
            cases.append(make_case(f"L{layer:02}_first", layer, qw, qb, qx,
                                   0, 0, 0xFF, 0xFFFF, False, "normal"))
            cases.append(make_case(f"L{layer:02}_middle", layer, qw, qb, qx,
                                   48, 128, 0xFF, 0xFFFF, False, "normal"))
            cases.append(make_case(f"L{layer:02}_last", layer, qw, qb, qx,
                                   96, 240, 0x07, 0xFFFF, False, "normal"))
            names.append(stem)
        # Boundary cases anchored on layer 01 frozen parameters.
        qw = data["layer01_weight"].astype(np.int64)
        qb = data["layer01_bias"].astype(np.int64)
        qx = synthetic_activations(1)
        cases.append(make_case("B_k1", 1, qw, qb, qx, 0, 0, 0xFF, 0xFFFF,
                               False, "normal", klen=1))
        cases.append(make_case("B_colpad", 1, qw, qb, qx, 0, 0, 0xFF, 0x000F,
                               False, "normal"))
        cases.append(make_case("B_relu", 1, qw, qb, qx, 0, 0, 0xFF, 0xFFFF,
                               True, "normal"))
        cases.append(make_case("B_reset", 1, qw, qb, qx, 0, 0, 0xFF, 0xFFFF,
                               False, "reset"))
        cases.append(make_case("B_pingpong", 1, qw, qb, qx, 0, 0, 0xFF, 0xFFFF,
                               False, "pingpong"))
    return cases, names


def write_stimulus(case, work: Path):
    write_hex(work / "a.hex", case["a"], 16)
    write_hex(work / "b.hex", case["b"], 16)
    write_hex(work / "bias.hex", case["bias"], 32)


def run_case(binary: Path, case, work: Path):
    write_stimulus(case, work)
    out = work / "out.hex"
    plusargs = [
        f"+A={work/'a.hex'}", f"+B={work/'b.hex'}", f"+BIAS={work/'bias.hex'}",
        f"+OUT={out}", f"+KLEN={case['klen']}",
        f"+ROWMASK={case['row_mask']:x}", f"+COLMASK={case['col_mask']:x}",
        f"+RELU={case['relu']}", f"+MODE={case['mode']}",
        f"+CASE={case['name']}",
    ]
    actuals = []
    if case["mode"] == "pingpong":
        out2 = work / "out2.hex"
        plusargs.append(f"+OUT2={out2}")
        actuals.append(out2)
    actuals.insert(0, out)
    result = subprocess.run([str(binary), *plusargs], capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(f"simulation failed for {case['name']}")
    arrays = []
    for path in actuals:
        words = [int(line, 16) for line in path.read_text().split()]
        if len(words) != ROWS * COLS:
            raise SystemExit(f"{case['name']}: expected {ROWS*COLS} outputs, got {len(words)}")
        arrays.append(np.array(words, dtype=np.uint32).view(np.int32).astype(np.int64)
                      .reshape(ROWS, COLS))
    return arrays


def compare(actual: np.ndarray, expected: np.ndarray):
    diff = actual != expected
    mismatches = int(diff.sum())
    first = None
    if mismatches:
        r, c = np.argwhere(diff)[0]
        first = (int(r), int(c), int(actual[r, c]), int(expected[r, c]))
    return mismatches, first


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--out", type=Path, default=Path("/tmp/kwt-verilator-extended"))
    parser.add_argument("--verilator", default=shutil.which("verilator") or "verilator")
    parser.add_argument("--inject-output", nargs=3, metavar=("CASE", "ROW", "COL"),
                        default=None,
                        help="negative injection: corrupt one DUT output before "
                             "comparison so the detector must fail")
    args = parser.parse_args()

    if not args.params.exists():
        parser.error(f"frozen parameter package not found: {args.params}")
    args.out.mkdir(parents=True, exist_ok=True)

    version = subprocess.run([args.verilator, "--version"], capture_output=True,
                             text=True).stdout.strip()
    cases, layer_stems = build_cases(args.params)
    nonzero_expected = int(sum(int(np.count_nonzero(case["expected"])) for case in cases))
    if nonzero_expected == 0:
        raise SystemExit("degenerate stimulus: no nonzero expected outputs")
    binary = build(args.verilator, args.out / "build")

    results = []
    stored_actual = None
    checked = 0
    for case in cases:
        work = args.out / case["name"]
        work.mkdir(parents=True, exist_ok=True)
        arrays = run_case(binary, case, work)
        if args.inject_output is not None:
            iname, irow, icol = args.inject_output
            if iname == case["name"]:
                arrays[0][int(irow), int(icol)] += 1
        mismatches, first = compare(arrays[0], case["expected"])
        checked += arrays[0].size
        if case["mode"] == "pingpong":
            mm2, first2 = compare(arrays[1], case["expected"])
            checked += arrays[1].size
            if mm2 and first is None:
                mismatches, first = mm2, first2
        results.append({"name": case["name"], "layer": case["layer"],
                        "mode": case["mode"], "mismatches": mismatches})
        if stored_actual is None:
            stored_actual = (arrays[0].copy(), case["expected"].copy())
        status = "ok" if mismatches == 0 else f"MISMATCH {first}"
        print(f"[{'PASS' if mismatches == 0 else 'FAIL'}] {case['name']:14s} "
              f"K={case['klen']:2d} row0={case['row0']:2d} col0={case['col0']:3d} "
              f"relu={case['relu']} mode={case['mode']:8s} {status}", flush=True)
        if mismatches:
            raise SystemExit(f"differential mismatch in {case['name']}: {first}")

    # Negative injection: corrupt one expected element and require detection.
    actual, expected = stored_actual
    injected = expected.copy()
    injected[3, 5] += 1
    injected_mismatches, injected_first = compare(actual, injected)
    if injected_mismatches == 0:
        raise SystemExit("negative injection was NOT detected (self-check broken)")

    summary = {
        "verilator": version,
        "params": str(args.params),
        "params_sha256": sha256(args.params),
        "layers": layer_stems,
        "cases": len(cases),
        "checked_outputs": checked,
        "nonzero_expected_outputs": nonzero_expected,
        "mismatches": 0,
        "negative_injection": {
            "detected": True,
            "mismatches": injected_mismatches,
            "first": injected_first,
        },
        "results": results,
        "limitations": [
            "Tile-engine differential test only; no E203 SoC, DMA, shared bus,",
            "SRAM macro, synthesis, timing, area or power are exercised.",
            "Activations are a deterministic synthetic pattern, not real audio.",
            "The frozen weights/bias are the project's A12W10 parameters.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"PASS {len(cases)} cases, {checked} outputs, "
          f"0 mismatches; negative injection detected "
          f"({injected_mismatches} mismatches) at {injected_first}")
    print(f"verilator: {version}")
    print(f"summary: {args.out/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
