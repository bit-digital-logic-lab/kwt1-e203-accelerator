#!/usr/bin/env python3
"""Schedule the multisample A12W10 E203 real-layer validation.

Boundary: this script only writes under the worktree validation directory and
an ASCII temporary root.  Shared sample/parameter/weight archives and the
existing run_e203.sh are read-only.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

M, K, N, LAYERS = 99, 64, 256, 12
ROTATION = {0: [1, 4, 7, 10], 40: [2, 5, 8, 11], 440: [3, 6, 9, 12]}
MANDATORY = [1, 6, 12]
TILE_EXPECT = {0: (128, 0), 207: (48, 80)}
RUNNING_RE = re.compile(r"^Running E203 (layer\d+)$", re.MULTILINE)
PASS_RE = re.compile(
    r"^PASS frozen real-sample E203 FFN1 tiles: start=(\d+) count=(\d+) "
    r"outputs=(\d+) padding=(\d+) task_cycles=(\d+) total_cycles=(\d+)$",
    re.MULTILINE)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_layers(index):
    return sorted(set(ROTATION[index]) | set(MANDATORY))


def run(command, log_path, env):
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                 text=True, env=env)
    return process.returncode, log_path.read_text(encoding="utf-8")


def parse_log(text):
    running = RUNNING_RE.findall(text)
    passes = []
    for match in PASS_RE.finditer(text):
        start, count, outputs, padding, task_cycles, total_cycles = map(int, match.groups())
        passes.append({"start": start, "count": count, "outputs": outputs,
                       "padding": padding, "task_cycles": task_cycles,
                       "total_cycles": total_cycles})
    return running, passes


def stage_sampling(selected, vector_dir, sampling_dir):
    if sampling_dir.exists():
        shutil.rmtree(sampling_dir)
    sampling_dir.mkdir(parents=True)
    for layer in selected:
        stem = f"layer{layer:02d}"
        for key in ("a16", "w16", "bias32", "expected32"):
            source = vector_dir / f"{stem}_{key}.hex"
            if not source.is_file():
                raise FileNotFoundError(source)
            (sampling_dir / source.name).symlink_to(source)


def add_sample_hashes(vector_root, indices, archive):
    import numpy as np
    with np.load(archive, allow_pickle=False) as data:
        features = data["features"]
        partitions = data["partitions"]
        validation_ids = np.flatnonzero(partitions == "validation")
    for index in indices:
        directory = vector_root / f"sample_idx{index:03d}"
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sample_id = int(validation_ids[index])
        manifest["sample_feature_sha256"] = hashlib.sha256(
            features[sample_id].tobytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--shared-root", type=Path,
                        default=Path("/home/yang/Desktop/ic设计竞赛"))
    parser.add_argument("--python", type=Path,
                        default=Path("/home/yang/Desktop/ic设计竞赛/software/.venv/bin/python"))
    parser.add_argument("--tmp-root", type=Path,
                        default=Path("/tmp/opencode/multisample_validation"))
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 40, 440])
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--extra-full-layer", action="store_true",
                        help="4th sample idx=80, layer01 only, all 208 tiles")
    args = parser.parse_args()

    samples = args.shared_root / "software/.build/quant_samples_12labels_40.npz"
    params = args.shared_root / "software/results/ffn1_a12w10_static_sym_12labels_40.npz"
    weights = args.shared_root / "software/weights/kwt1_fp32.npz"
    for path in (samples, params, weights, params.with_suffix(".json")):
        if not path.is_file():
            raise SystemExit(f"missing shared read-only input: {path}")

    vector_root = args.tmp_root / "vectors"
    run_root = args.tmp_root / "runs"
    vector_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = "/home/yang/yosys-sta/oss-cad-suite/bin:" + env.get("PATH", "")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("OMP_NUM_THREADS", "4")

    prepare = args.repo / "hardware/rtl/control/tb/prepare_real_e203_vectors.py"
    runner = args.repo / "hardware/rtl/control/tb/run_e203.sh"

    if not args.skip_generate:
        for index in args.indices:
            out = vector_root / f"sample_idx{index:03d}"
            command = [
                str(args.python), str(prepare),
                "--samples", str(samples), "--params", str(params),
                "--weights", str(weights), "--validation-index", str(index),
                "--device", "cpu", "--out", str(out),
            ]
            log_path = run_root / f"generate_idx{index:03d}.log"
            print(f"generate idx={index}")
            code, text = run(command, log_path, env)
            if code != 0:
                print(text, file=sys.stderr)
                return code
    add_sample_hashes(vector_root, args.indices, samples)

    check = here / "check_vectors.py"
    check_log = run_root / "vector_check.log"
    command = [str(args.python), str(check), "--vector-root", str(vector_root),
               "--samples", str(samples), "--params", str(params),
               "--weights", str(weights), "--indices", *map(str, args.indices),
               "--out", str(run_root / "vector_checks.json")]
    code, text = run(command, check_log, env)
    print(text, end="")
    if code != 0:
        print("vector independent check failed; stopping before E203", file=sys.stderr)
        return code
    checks = json.loads((run_root / "vector_checks.json").read_text(encoding="utf-8"))

    results = {"format": "KWT-1 A12W10 multisample E203 sample matrix v1",
               "evidence": "Icarus NICE-like E203 + simulation bus/SRAM model",
               "shared_inputs": {
                   "sample_archive": str(samples), "sample_archive_sha256": sha256(samples),
                   "parameter_archive": str(params), "parameter_archive_sha256": sha256(params),
                   "fp32_weight_archive": str(weights),
                   "fp32_weight_archive_sha256": sha256(weights)},
               "samples": []}
    run_records = []
    for index, check_record in zip(args.indices, checks["samples"]):
        vector_dir = vector_root / f"sample_idx{index:03d}"
        selected = selected_layers(index)
        sampling_dir = args.tmp_root / "sampling" / f"sample_idx{index:03d}"
        stage_sampling(selected, vector_dir, sampling_dir)
        sample_record = {
            "validation_index": index,
            "label": check_record["label"],
            "sample_path": check_record["sample_path"],
            "sample_feature_sha256": check_record["sample_feature_sha256"],
            "rotation_layers": ROTATION[index],
            "selected_layers": selected,
            "full_12_layer_vectors_checked": len(check_record["layers"]) == LAYERS,
            "worst_layer_nrmse_pct": check_record["worst_layer_nrmse_pct"],
            "runs": [],
        }
        for tile in (0, 207):
            log_path = run_root / f"sample_idx{index:03d}_tile{tile:03d}.log"
            command = ["bash", str(runner), "--real-layers", str(sampling_dir),
                       str(tile), "1"]
            print(f"E203 idx={index} tile={tile} layers={selected}")
            code, text = run(command, log_path, env)
            running, passes = parse_log(text)
            expected_outputs, expected_padding = TILE_EXPECT[tile]
            expected_running = [f"layer{layer:02d}" for layer in selected]
            ok = (code == 0 and running == expected_running and len(passes) == len(selected)
                  and all(item["start"] == tile and item["count"] == 1
                          and item["outputs"] == expected_outputs
                          and item["padding"] == expected_padding for item in passes)
                  and "FATAL" not in text and "mismatch" not in text)
            record = {
                "tile": tile, "exit_code": code, "pass_count": len(passes),
                "expected_pass_count": len(selected), "running_layers": running,
                "passes": passes, "ok": ok, "log": str(log_path),
                "log_sha256": sha256(log_path),
            }
            sample_record["runs"].append(record)
            run_records.append((index, tile, record))
            if not ok:
                results["samples"].append(sample_record)
                results["first_failure"] = record
                (run_root / "e203_results.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
                print(text, file=sys.stderr)
                print(f"FAIL idx={index} tile={tile} exit={code} "
                      f"passes={len(passes)}/{len(selected)}; stopped", file=sys.stderr)
                return 1
        results["samples"].append(sample_record)

    # Full 12-layer rotation matrix.  Only rows that actually ran can be PASS.
    matrix = []
    for sample in results["samples"]:
        index = sample["validation_index"]
        ran = {}
        failed = set()
        for record in sample["runs"]:
            tile = record["tile"]
            for layer in sample["selected_layers"]:
                ran[(layer, tile)] = record["exit_code"] == 0 and record["ok"]
        for layer in range(1, LAYERS + 1):
            cells = []
            for tile in (0, 207):
                if (layer, tile) in ran:
                    cells.append("PASS" if ran[(layer, tile)] else "FAIL")
                else:
                    cells.append("not_run")
            matrix.append({"validation_index": index, "label": sample["label"],
                           "layer": layer,
                           "primary_rotation": layer in ROTATION[index],
                           "tile000": cells[0], "tile207": cells[1]})
    results["matrix"] = matrix
    results["ok"] = all(record["ok"] for _, _, record in run_records)

    if args.extra_full_layer:
        index = 80
        out = vector_root / f"sample_idx{index:03d}"
        command = [
            str(args.python), str(prepare),
            "--samples", str(samples), "--params", str(params),
            "--weights", str(weights), "--validation-index", str(index),
            "--layers", "1", "--device", "cpu", "--out", str(out),
        ]
        code, text = run(command, run_root / "generate_extra_idx080.log", env)
        if code != 0:
            print(text, file=sys.stderr)
            return code
        add_sample_hashes(vector_root, [index], samples)
        sampling_dir = args.tmp_root / "sampling" / f"sample_idx{index:03d}_layer01"
        stage_sampling([1], out, sampling_dir)
        log_path = run_root / "sample_idx080_layer01_full208.log"
        print("E203 extra sample idx=80 layer01 tiles=0..207")
        command = ["bash", str(runner), "--real-layers", str(sampling_dir), "0", "208"]
        code, text = run(command, log_path, env)
        running, passes = parse_log(text)
        ok = (code == 0 and running == ["layer01"] and len(passes) == 1
              and passes[0]["start"] == 0 and passes[0]["count"] == 208
              and passes[0]["outputs"] == M * N and passes[0]["padding"] == 1280
              and "FATAL" not in text and "mismatch" not in text)
        results["extra_full_layer"] = {
            "validation_index": index, "layer": 1, "tile_start": 0, "tile_count": 208,
            "exit_code": code, "pass_count": len(passes), "passes": passes,
            "ok": ok, "log": str(log_path), "log_sha256": sha256(log_path)}
        if not ok:
            print(text, file=sys.stderr)
            results["ok"] = False

    (run_root / "e203_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (run_root / "e203_matrix.tsv").open("w", encoding="utf-8") as destination:
        destination.write("validation_index\tlabel\tlayer\tprimary_rotation\ttile000\ttile207\n")
        for row in matrix:
            destination.write(f"{row['validation_index']}\t{row['label']}\t{row['layer']}\t"
                              f"{str(row['primary_rotation']).lower()}\t"
                              f"{row['tile000']}\t{row['tile207']}\n")
    print(f"results: {run_root / 'e203_results.json'}")
    return 0 if results["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
