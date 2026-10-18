#!/usr/bin/env python3
"""Run tile-level edge cases against the provisional system model.

The suite deliberately avoids a 12-layer/208-tile regression.  It uses one
frozen real FFN1 layer from the shared vector directory and checks:

* the last row tile (3 valid rows + 5 masked padding rows),
* consecutive commands without an intervening reset,
* a reset injected while the tile engine is computing,
* a negative injection that corrupts one independently computed golden value
  and requires the testbench to report a mismatch.

The golden used by the HDL testbench is recomputed here from the frozen
A/W/bias files with exact integer arithmetic.  It is also cross-checked against
the separately generated expected32 file; neither the golden nor the expected
value is copied from the DUT.  The shared vectors are read-only: this script
stages ASCII copies in a temporary directory, which also avoids Icarus
problems with non-ASCII source paths.
"""

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[4]
TB_DIR = pathlib.Path(__file__).resolve().parent
RTL = REPO / "hardware/rtl"
SRAM = RTL / "sram"
M, K, N = 99, 64, 256
ROW_TILES, COL_TILES = 13, 16
TILE_ELEMS = 8 * 16
MAX_CASES = 64  # must match tb_kwt_tile_edge_cases.sv


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_vector_root():
    configured = os.environ.get("KWT_VECTOR_ROOT")
    if configured:
        return pathlib.Path(configured)
    # The primary worktree is the first entry in `git worktree list`.
    try:
        text = subprocess.check_output(
            ["git", "-C", str(REPO), "worktree", "list", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        text = ""
    for line in text.splitlines():
        if line.startswith("worktree "):
            return (pathlib.Path(line[len("worktree "):]) /
                    "hardware/architecture/results/real_e203_validation0_vectors")
    return REPO / "hardware/architecture/results/real_e203_validation0_vectors"


def read_hex(path):
    return [int(token, 16) for token in path.read_text(encoding="ascii").split()]


def signed(value, bits):
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def load_layer(root, layer):
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing vector manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((item for item in manifest.get("layers", [])
                  if item.get("layer") == layer), None)
    if entry is None:
        raise SystemExit(f"layer {layer} not present in {manifest_path}")
    stem = f"layer{layer:02}"
    names = {"a16": f"{stem}_a16.hex", "w16": f"{stem}_w16.hex",
             "bias32": f"{stem}_bias32.hex",
             "expected32": f"{stem}_expected32.hex"}
    for key, filename in names.items():
        path = root / filename
        record = entry["files"][key]
        if not path.is_file():
            raise SystemExit(f"missing vector file: {path}")
        actual = sha256(path)
        if actual != record["sha256"]:
            raise SystemExit(f"SHA256 mismatch for {path}: {actual} != {record['sha256']}")
    return manifest, entry, {key: read_hex(root / filename) for key, filename in names.items()}


def build_cases():
    cases = []
    # Last row tile: rows 96..98 are valid, rows 99..103 are padding.
    for ct in range(COL_TILES):
        cases.append(("last", 12, ct, 0x07))
    # Back-to-back full tiles with no reset between commands.
    for rt, ct in ((0, 0), (5, 9), (0, 0)):
        cases.append(("continuous", rt, ct, 0xFF))
    reset_index = len(cases)
    # This command is aborted by reset and is not compared.
    cases.append(("reset", 2, 3, 0xFF))
    # First command after reset; also the last row tile.
    cases.append(("recovery", 12, 15, 0x07))
    return cases, reset_index


def pack_case(rt, ct, mask):
    tile = rt * COL_TILES + ct
    return (tile | (rt << 16) | (ct << 24) | (mask << 32))


def compute_golden(data, cases):
    a16 = [signed(value, 16) for value in data["a16"]]
    w16 = [signed(value, 16) for value in data["w16"]]
    bias32 = [signed(value, 32) for value in data["bias32"]]
    expected32 = [signed(value, 32) for value in data["expected32"]]
    if len(a16) != ROW_TILES * K * 8 or len(w16) != COL_TILES * K * 16:
        raise SystemExit("unexpected frozen vector dimensions")
    if len(bias32) != N or len(expected32) != M * N:
        raise SystemExit("unexpected frozen bias/expected dimensions")
    # The manifest states rows 3..7 of the last A tile are zero.
    for r in range(3, 8):
        for k in range(K):
            if a16[(12 * K + k) * 8 + r] != 0:
                raise SystemExit("last A tile padding row is not zero")

    golden = []
    checked_valid = 0
    checked_padding = 0
    cross_checked = 0
    for kind, rt, ct, mask in cases:
        for r in range(8):
            row = rt * 8 + r
            for c in range(16):
                col = ct * 16 + c
                if row < M and (mask >> r) & 1:
                    value = bias32[col]
                    for k in range(K):
                        value += a16[(rt * K + k) * 8 + r] * w16[(ct * K + k) * 16 + c]
                    if not -(1 << 31) <= value < (1 << 31):
                        raise SystemExit(f"golden overflow at tile {rt},{ct} [{r},{c}]")
                    if value != expected32[row * N + col]:
                        raise SystemExit(
                            f"independent golden differs from expected32 at "
                            f"[{row},{col}]: {value} != {expected32[row * N + col]}")
                    cross_checked += 1
                    checked_valid += 1
                else:
                    value = 0
                    checked_padding += 1
                golden.append(value & 0xFFFFFFFF)
    return golden, checked_valid, checked_padding, cross_checked


def padded_hex(values, width, total):
    lines = [f"{value:0{width}x}" for value in values]
    lines += ["0" * width] * max(0, total - len(lines))
    return "\n".join(lines) + "\n"


def stage_files(work, data, cases, golden):
    (work / "a.hex").write_text(
        padded_hex([v & 0xFFFF for v in data["a16"]], 4, 13 * K * 8), encoding="ascii")
    (work / "w.hex").write_text(
        padded_hex([v & 0xFFFF for v in data["w16"]], 4, 16 * K * 16), encoding="ascii")
    (work / "bias.hex").write_text(
        padded_hex([v & 0xFFFFFFFF for v in data["bias32"]], 8, N), encoding="ascii")
    (work / "golden.hex").write_text(
        padded_hex([v & 0xFFFFFFFF for v in golden], 8, MAX_CASES * TILE_ELEMS),
        encoding="ascii")
    (work / "cases.hex").write_text(
        padded_hex([pack_case(rt, ct, mask) for _, rt, ct, mask in cases], 16, MAX_CASES),
        encoding="ascii")


def compile_model(work):
    image = work / "tile_edge_cases"
    sources = [
        RTL / "pulse_array/kwt_pe.sv",
        RTL / "pulse_array/kwt_pulse_array.sv",
        RTL / "pulse_array/kwt_postprocess.sv",
        RTL / "pulse_array/kwt_tile_engine.sv",
        RTL / "pulse_array/kwt_product_array.sv",
        RTL / "pulse_array/kwt_cacc.sv",
        RTL / "pulse_array/kwt_pingpong_cacc_engine.sv",
        SRAM / "kwt_operand_pingpong.sv",
        SRAM / "kwt_tile_buffers.sv",
        RTL / "dma/kwt_dma.sv",
        RTL / "dma/kwt_ext_memory_model.sv",
        RTL / "control/kwt_tile_system_model.sv",
        TB_DIR / "tb_kwt_tile_edge_cases.sv",
    ]
    command = [os.environ.get("IVERILOG", "iverilog"), "-g2012", "-I", str(SRAM),
               "-s", "tb_kwt_tile_edge_cases", "-o", str(image)]
    command += [str(source) for source in sources]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode or not image.is_file():
        hint = ""
        if "unresolved wire array" in result.stderr or not image.is_file():
            hint = ("\nnote: this Icarus build cannot elaborate the frozen RTL; "
                    "use Icarus 14 (for example the oss-cad-suite iverilog).")
        raise SystemExit("iverilog failed:\n" + result.stdout + result.stderr + hint)
    return image


def run_case(work, image, case_count, reset_index, golden_name="golden.hex"):
    command = [
        os.environ.get("VVP", "vvp"), str(image),
        f"+A={work / 'a.hex'}", f"+W={work / 'w.hex'}",
        f"+BIAS={work / 'bias.hex'}", f"+GOLDEN={work / golden_name}",
        f"+CASES={work / 'cases.hex'}",
        f"+CASE_COUNT={case_count}", f"+RESET_INDEX={reset_index}",
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    return result.returncode, result.stdout + result.stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=pathlib.Path, default=default_vector_root())
    parser.add_argument("--layer", type=int, default=1, choices=range(1, 13))
    args = parser.parse_args()
    root = args.vectors.resolve()
    if not root.is_dir():
        parser.error(f"vector directory not found: {root} (set KWT_VECTOR_ROOT or --vectors)")

    manifest, entry, data = load_layer(root, args.layer)
    cases, reset_index = build_cases()
    golden, valid, padding, cross_checked = compute_golden(data, cases)
    print(f"vectors={root}")
    print(f"layer={args.layer:02} manifest_sample={manifest.get('sample_path')}")
    print(f"independent_golden: valid={valid} padding={padding} "
          f"expected32_cross_checks={cross_checked}")
    print(f"cases={len(cases)} reset_index={reset_index} "
          f"last_tile_valid={16*3*16} last_tile_padding={16*5*16}")

    temp_parent = pathlib.Path("/tmp/opencode")
    if not temp_parent.is_dir():
        temp_parent = pathlib.Path(tempfile.gettempdir())
    with tempfile.TemporaryDirectory(prefix="kwt-tile-edges-", dir=temp_parent) as temp:
        work = pathlib.Path(temp)
        stage_files(work, data, cases, golden)
        image = compile_model(work)
        positive_rc, positive_log = run_case(work, image, len(cases), reset_index)
        print("---- positive run ----")
        print(positive_log, end="" if positive_log.endswith("\n") else "\n")
        if positive_rc != 0 or "PASS tile edge cases:" not in positive_log:
            print(f"FAIL positive exit={positive_rc}", file=sys.stderr)
            return 1

        # Negative injection: corrupt one independently computed valid golden
        # value in the final checked case and require a reported mismatch.
        last = len(cases) - 1
        index = last * TILE_ELEMS + 2 * 16 + 15  # row 2, column 15 of last case
        bad_golden = list(golden)
        bad_golden[index] = (bad_golden[index] + 1) & 0xFFFFFFFF
        (work / "golden_bad.hex").write_text(
            padded_hex(bad_golden, 8, MAX_CASES * TILE_ELEMS), encoding="ascii")
        negative_rc, negative_log = run_case(
            work, image, len(cases), reset_index, "golden_bad.hex")
        print("---- negative-injection run (must fail) ----")
        print(negative_log, end="" if negative_log.endswith("\n") else "\n")
        if negative_rc == 0 or "MISMATCH" not in negative_log:
            print(f"FAIL negative injection was not detected exit={negative_rc}",
                  file=sys.stderr)
            return 1
        print(f"NEGATIVE_INJECTION_DETECTED golden_index={index} exit={negative_rc}")

    print(f"PASS tile edge suite: cases={len(cases)} valid={valid} padding={padding} "
          f"continuous_tasks={len(cases)-2} reset_tasks=1 "
          f"negative_injection=detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
