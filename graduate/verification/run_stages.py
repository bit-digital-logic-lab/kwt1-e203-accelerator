#!/usr/bin/env python3
"""Staged verification runner for the graduate (研究生组) KWT-1 tasks.

The runner does **not** decide PASS from the existence of files.  A stage is
PASS only after its evidence command actually ran, returned exit code 0 and the
captured output contained a real comparison result (a contract line or an
explicit pass marker with a checked count).  Anything else is FAIL.  A stage
whose design module is absent is SKIP, with the missing paths recorded.

Stages (see README.md for the full contract):

    attention   single-head attention module, 99x99 scores, dim 64   (RTL)
    encoder1    one PostNorm encoder layer                            (RTL)
    encoder12   twelve encoder layers + per-layer parameter switch    (RTL)
    e203        E203 drives the accelerator (NICE), simplified SoC    (SoC)
    synth       Yosys structural synthesis evidence                   (synth)
    smoke_*     checks that are runnable on the integration baseline

Evidence classes are kept separate on purpose:

    software    numeric error of the software model (NRMSE / max abs)
    rtl         bit-exact RTL vs an independent golden
    soc         E203 / bus / DMA integration
    synth       elaboration, structural counts, mapped area

Use ``--design-root`` to run the same stages against another worktree/branch.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import Callable, Iterable, Sequence

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[1]  # .../graduate/verification -> repo root

CONTRACT_RE = re.compile(
    r"KWT-GRAD-VERIFY\s+(?P<stage>[\w.-]+)\s+checked=(?P<checked>\d+)\s+failed=(?P<failed>\d+)"
)
RESULT_RATIO_RE = re.compile(r"result:\s*(?P<passed>\d+)\s*/\s*(?P<total>\d+)\s+passed")

# Toolchain locations that are known to elaborate the frozen RTL correctly.
OSS_CAD_BIN = "/home/yang/yosys-sta/oss-cad-suite/bin"
IVERILOG_MIN_MAJOR = 12  # Icarus 11 cannot resolve unpacked arrays in this RTL.
YOSYS_MIN_MINOR = 20


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: pathlib.Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def tool_version(executable: str, *args: str) -> str:
    try:
        proc = subprocess.run([executable, *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    text = (proc.stdout + proc.stderr).strip().splitlines()
    return text[0] if text else "unknown"


def find_iverilog(override: str | None) -> tuple[str | None, str]:
    candidates: list[str] = []
    if override:
        candidates.append(override)
    env = os.environ.get("IVERILOG")
    if env:
        candidates.append(env)
    candidates.append(os.path.join(OSS_CAD_BIN, "iverilog"))
    path_hit = shutil.which("iverilog")
    if path_hit:
        candidates.append(path_hit)
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if not os.path.exists(candidate):
            continue
        version = tool_version(candidate, "-V")
        match = re.search(r"version\s+(\d+)", version)
        if match and int(match.group(1)) >= IVERILOG_MIN_MAJOR:
            return candidate, version
    return None, "Icarus Verilog >= %d not found (Icarus 11 mis-elaborates the unpacked arrays)" % IVERILOG_MIN_MAJOR


def find_yosys(override: str | None) -> tuple[str | None, str]:
    candidates: list[str] = []
    if override:
        candidates.append(override)
    env = os.environ.get("YOSYS")
    if env:
        candidates.append(env)
    candidates.append(os.path.join(OSS_CAD_BIN, "yosys"))
    path_hit = shutil.which("yosys")
    if path_hit:
        candidates.append(path_hit)
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if not os.path.exists(candidate):
            continue
        version = tool_version(candidate, "-V")
        match = re.search(r"Yosys\s+(\d+)\.(\d+)", version)
        if match and int(match.group(2)) >= YOSYS_MIN_MINOR:
            return candidate, version
    return None, "Yosys >= 0.%d not found (Yosys 0.9 cannot parse the SystemVerilog RTL)" % YOSYS_MIN_MINOR


def git_commit(root: pathlib.Path) -> str | None:
    try:
        proc = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


# --------------------------------------------------------------------------- #
# stage model
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Stage:
    name: str
    kind: str  # software | rtl | soc | synth
    gate: list[pathlib.Path]          # design modules; missing -> SKIP
    harness: list[pathlib.Path]       # evidence scripts; missing -> FAIL
    command: list[str]
    sample: str
    compare: str
    golden: str
    validators: list[Callable[[str], tuple[bool, str, int]]]
    forbid: list[str] = dataclasses.field(default_factory=list)
    inputs: list[pathlib.Path] = dataclasses.field(default_factory=list)
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    timeout: int = 900
    requires: list[str] = dataclasses.field(default_factory=list)
    skip_note: str = ""


# validators ----------------------------------------------------------------- #
def contract_validator(stage_name: str, minimum: int = 1):
    def check(output: str) -> tuple[bool, str, int]:
        best: tuple[bool, str, int] | None = None
        for match in CONTRACT_RE.finditer(output):
            if match.group("stage") != stage_name:
                continue
            checked = int(match.group("checked"))
            failed = int(match.group("failed"))
            if checked >= minimum and failed == 0:
                result = (True, f"contract checked={checked} failed={failed}", checked)
            else:
                result = (False, f"contract checked={checked} failed={failed}", checked)
            best = result
        if best is None:
            return False, f"no 'KWT-GRAD-VERIFY {stage_name}' contract line", 0
        return best
    return check


def result_ratio_validator(minimum: int = 1):
    def check(output: str) -> tuple[bool, str, int]:
        matches = list(RESULT_RATIO_RE.finditer(output))
        if not matches:
            return False, "no 'result: N/N passed' line", 0
        match = matches[-1]
        passed, total = int(match.group("passed")), int(match.group("total"))
        if total < minimum or passed != total:
            return False, f"result {passed}/{total} passed", passed
        return True, f"{passed}/{total} vectors passed", passed
    return check


def regex_all_validator(patterns: list[str], description: str):
    compiled = [re.compile(p) for p in patterns]

    def check(output: str) -> tuple[bool, str, int]:
        missing = [p.pattern for p in compiled if not p.search(output)]
        if missing:
            return False, "missing marker(s): " + ", ".join(missing), 0
        return True, description, 1
    return check


# --------------------------------------------------------------------------- #
# classification (pure; exercised by faults/test_harness_faults.py)
# --------------------------------------------------------------------------- #
def classify(returncode: int, output: str, stage: Stage) -> tuple[str, str]:
    """Return (status, reason) for an executed stage."""
    if returncode != 0:
        return "FAIL", f"evidence command exited {returncode}"
    for pattern in stage.forbid:
        if re.search(pattern, output):
            return "FAIL", f"forbidden marker matched: {pattern}"
    checked_total = 0
    for validator in stage.validators:
        ok, detail, checked = validator(output)
        if not ok:
            return "FAIL", detail
        checked_total += checked
    if checked_total < 1:
        return "FAIL", "no checked comparison was reported"
    return "PASS", f"checked={checked_total}"


# --------------------------------------------------------------------------- #
# stage registry
# --------------------------------------------------------------------------- #
def build_stages(root: pathlib.Path, tools: dict) -> list[Stage]:
    def rel(*parts: str) -> pathlib.Path:
        return root.joinpath(*parts)

    stages: list[Stage] = []

    # ---- attention -------------------------------------------------------- #
    stages.append(Stage(
        name="attention",
        kind="rtl",
        gate=[rel("hardware/rtl/attention/kwt_attention.sv")],
        harness=[rel("hardware/rtl/attention/tb/run.sh")],
        command=["bash", str(rel("hardware/rtl/attention/tb/run.sh"))],
        sample="KWT-1 single-head attention: Q/K/V projection, 99x99 scores, "
               "row softmax, probability-weighted values, output projection (dim 64)",
        compare="RTL vs an independent golden authored in the attention branch",
        golden="independent golden required from the harness (never derived from DUT output)",
        validators=[contract_validator("attention")],
        forbid=[r"\bFAIL\b", r"MISMATCH"],
        inputs=[rel("hardware/rtl/attention/kwt_attention.sv")],
        env={"KWT_GRAD_DESIGN_ROOT": str(root)},
        requires=["iverilog"],
    ))

    # ---- single encoder layer -------------------------------------------- #
    stages.append(Stage(
        name="encoder1",
        kind="rtl",
        gate=[rel("hardware/rtl/encoder/kwt_encoder_layer.sv")],
        harness=[rel("hardware/rtl/encoder/tb/run_single_layer.sh")],
        command=["bash", str(rel("hardware/rtl/encoder/tb/run_single_layer.sh"))],
        sample="One PostNorm encoder layer: attention + 2 residual adds + 2 LayerNorm + "
               "64->256->64 FFN + GELU",
        compare="RTL vs the design software numeric model, per-layer intermediate outputs",
        golden="independent golden required from the harness (never derived from DUT output)",
        validators=[contract_validator("encoder1")],
        forbid=[r"\bFAIL\b", r"MISMATCH"],
        inputs=[rel("hardware/rtl/encoder/kwt_encoder_layer.sv")],
        env={"KWT_GRAD_DESIGN_ROOT": str(root)},
        requires=["iverilog"],
    ))

    # ---- twelve encoder layers ------------------------------------------- #
    stages.append(Stage(
        name="encoder12",
        kind="rtl",
        gate=[rel("hardware/rtl/encoder/kwt_encoder_layer.sv"),
              rel("hardware/rtl/encoder/kwt_encoder_top12.sv")],
        harness=[rel("hardware/rtl/encoder/tb/run_twelve_layers.sh")],
        command=["bash", str(rel("hardware/rtl/encoder/tb/run_twelve_layers.sh"))],
        sample="All 12 encoder layers with layer-to-layer data passing and per-layer "
               "parameter update",
        compare="RTL vs the software numeric model across all 12 layers",
        golden="independent golden required from the harness (never derived from DUT output)",
        validators=[contract_validator("encoder12")],
        forbid=[r"\bFAIL\b", r"MISMATCH"],
        inputs=[rel("hardware/rtl/encoder/kwt_encoder_layer.sv"),
                rel("hardware/rtl/encoder/kwt_encoder_top12.sv")],
        env={"KWT_GRAD_DESIGN_ROOT": str(root)},
        requires=["iverilog"],
    ))

    # ---- E203 driven accelerator ----------------------------------------- #
    stages.append(Stage(
        name="e203",
        kind="soc",
        gate=[rel("hardware/rtl/control/tb/run_e203.sh")],
        harness=[rel("hardware/rtl/control/tb/run_e203.sh")],
        command=["bash", str(rel("hardware/rtl/control/tb/run_e203.sh"))],
        sample="Real E203 CPU executes the KWT program and drives the accelerator over "
               "the NICE interface (simplified SoC model, 128 outputs)",
        compare="E203 observed NICE request/response and outputs, cross-checked by the TB "
                "against the frozen tile golden",
        golden="independent tile golden compiled into tb_kwt_real_e203.sv",
        validators=[regex_all_validator(
            [r"PASS real E203 NICE: requests=[1-9]\d* responses=[1-9]\d* "
             r"outputs=[1-9]\d* cycles=\d+"],
            "E203 NICE run passed")],
        forbid=[r"\bFAIL\b", r"MISMATCH"],
        inputs=[rel("hardware/rtl/control/tb/run_e203.sh"),
                rel("hardware/rtl/control/tb/tb_kwt_real_e203.sv"),
                rel("hardware/rtl/control/tb/e203_kwt_program.S")],
        env={"KWT_GRAD_DESIGN_ROOT": str(root), "KWT_E203_TMP_BASE": "/tmp"},
        requires=["iverilog", "riscv64-linux-gnu-gcc", "e203_upstream"],
        skip_note="needs Icarus >= 12, riscv64-linux-gnu-gcc and 资料/e203_hbirdv2_full",
    ))

    # ---- synthesis evidence ---------------------------------------------- #
    stages.append(Stage(
        name="synth",
        kind="synth",
        gate=[rel("hardware/rtl/pulse_array/kwt_product_array.sv"),
              rel("hardware/rtl/pulse_array/kwt_cacc.sv"),
              rel("hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv")],
        harness=[HERE / "synth" / "yosys_evidence.py"],
        command=[sys.executable, str(HERE / "synth" / "yosys_evidence.py"),
                 "--design-root", str(root), "--yosys", str(tools["yosys"] or "yosys")],
        sample="kwt_row_broadcast_tile compute core: 128 (here 8) multipliers, CACC banks, "
               "bias add; optional standard-cell mapping of kwt_cacc",
        compare="Yosys elaboration + structural cell counts; optional mapped area with an "
                "example library (NOT a timing/area signoff)",
        golden="structural expectations ($mul >= 1, $mem_v2 >= 1, check -assert clean)",
        validators=[contract_validator("synth")],
        forbid=[r"ERROR:", r"Found and reported [1-9]\d* problems"],
        inputs=[rel("hardware/rtl/pulse_array/kwt_product_array.sv"),
                rel("hardware/rtl/pulse_array/kwt_cacc.sv"),
                rel("hardware/rtl/pulse_array/kwt_row_broadcast_tile.sv")],
        env={"KWT_GRAD_DESIGN_ROOT": str(root)},
        requires=["yosys"],
    ))

    # ---- currently runnable smoke ---------------------------------------- #
    stages.append(Stage(
        name="smoke_rtl_vector",
        kind="rtl",
        gate=[rel("hardware/testbenchplus/common/run_vectors.py")],
        harness=[rel("hardware/testbenchplus/common/run_vectors.py")],
        command=[sys.executable, str(rel("hardware/testbenchplus/common/run_vectors.py"))],
        sample="testbenchplus example vectors (layer0_fp32, layer0_int16); explicit, "
               "author-supplied expected matrices, not real KWT-1 layer data",
        compare="integer: bit-exact; fp32: NRMSE <= 3% and max abs error, against the "
                "vector 'expected' golden",
        golden="independent 'expected' matrices in vectors/*/case.json",
        validators=[result_ratio_validator(minimum=1)],
        forbid=[r"\bFAIL\b", r"MISMATCH", r"NaN"],
        inputs=[rel("hardware/testbenchplus/common/run_vectors.py"),
                rel("hardware/testbenchplus/common/tb_array_vector.sv"),
                rel("hardware/rtl/pulse_array/kwt_pe.sv"),
                rel("hardware/rtl/pulse_array/kwt_pulse_array.sv"),
                rel("hardware/testbenchplus/vectors/layer0_fp32/case.json"),
                rel("hardware/testbenchplus/vectors/layer0_int16/case.json")],
        env={"KWT_GRAD_DESIGN_ROOT": str(root)},
        requires=["iverilog"],
    ))

    stages.append(Stage(
        name="smoke_golden_faults",
        kind="rtl",
        gate=[rel("hardware/testbenchplus/common/negative_injection_check.py")],
        harness=[rel("hardware/testbenchplus/common/negative_injection_check.py")],
        command=[sys.executable,
                 str(rel("hardware/testbenchplus/common/negative_injection_check.py"))],
        sample="fault injection on both testbenchplus vectors: original golden must pass, "
               "golden with one element +1 must be rejected with non-zero exit",
        compare="harness self-check: corrupted independent golden must produce FAIL",
        golden="the same vectors/*/case.json goldens, deliberately corrupted",
        validators=[regex_all_validator(
            [r"PASS negative injection .*detected with exit=[1-9]\d*",
             r"PASS testbenchplus negative-injection self-check \(\d+ vectors\)"],
            "corrupted goldens detected")],
        forbid=[r"was not detected"],
        inputs=[rel("hardware/testbenchplus/common/negative_injection_check.py"),
                rel("hardware/testbenchplus/common/run_vectors.py"),
                rel("hardware/testbenchplus/vectors/layer0_fp32/case.json"),
                rel("hardware/testbenchplus/vectors/layer0_int16/case.json")],
        env={"KWT_GRAD_DESIGN_ROOT": str(root)},
        requires=["iverilog"],
    ))

    stages.append(Stage(
        name="smoke_software_golden",
        kind="software",
        gate=[rel("software/offline_encoder_input.py"),
              rel("software/weights/kwt1_fp32.npz"),
              rel("software/validation/real_input_chain_20261017/sample_00_mfcc.npy"),
              rel("software/validation/real_input_chain_20261017/sample_00_encoder.npy")],
        harness=[HERE / "golden" / "check_encoder_input.py"],
        command=[sys.executable, str(HERE / "golden" / "check_encoder_input.py"),
                 "--design-root", str(root)],
        sample="validation sample 00 (four/750e3e75_nohash_0.wav, label _unknown_): "
               "98x40 MFCC -> 99x64 layer-0 encoder input",
        compare="FP32 max abs / NRMSE between the design convert() result, an independent "
                "numpy reference and the stored TF-checkpoint golden",
        golden="stored sample_00_encoder.npy produced by the TensorFlow checkpoint path",
        validators=[contract_validator("encoder_input")],
        forbid=[r"FAIL"],
        inputs=[rel("software/offline_encoder_input.py"),
                rel("software/weights/kwt1_fp32.npz"),
                rel("software/validation/real_input_chain_20261017/sample_00_mfcc.npy"),
                rel("software/validation/real_input_chain_20261017/sample_00_encoder.npy")],
        env={"KWT_GRAD_DESIGN_ROOT": str(root)},
        requires=["python-numpy"],
    ))

    return stages


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def run_stage(stage: Stage, tools: dict, out_dir: pathlib.Path) -> dict:
    record: dict = {
        "stage": stage.name,
        "kind": stage.kind,
        "status": "SKIP",
        "reason": "",
        "command": stage.command,
        "command_shell": " ".join(_shell_quote(part) for part in stage.command),
        "exit_code": None,
        "duration_s": 0.0,
        "inputs": {},
        "sample_identity": stage.sample,
        "comparison": stage.compare,
        "golden": stage.golden,
        "tools": {key: tools.get(key) for key in ("python", "iverilog", "iverilog_version",
                                                   "yosys", "yosys_version",
                                                   "riscv64-linux-gnu-gcc")},
        "stdout_tail": "",
        "stderr_tail": "",
    }

    # module gate -> SKIP when the design does not contain the module yet
    missing_gate = [p for p in stage.gate if not p.exists()]
    if missing_gate:
        record["reason"] = "design module(s) absent: " + ", ".join(
            str(p) for p in missing_gate)
        if stage.skip_note:
            record["reason"] += "; " + stage.skip_note
        return record

    # tool gate -> SKIP when the environment cannot run the evidence
    missing_tools = [name for name in stage.requires if not tools.get(name)]
    if missing_tools:
        record["reason"] = "required tool(s)/data unavailable: " + ", ".join(missing_tools)
        if stage.skip_note:
            record["reason"] += "; " + stage.skip_note
        return record

    # harness gate -> FAIL: module exists but no evidence harness
    missing_harness = [p for p in stage.harness if not p.exists()]
    if missing_harness:
        record["status"] = "FAIL"
        record["reason"] = "module present but evidence harness missing: " + ", ".join(
            str(p) for p in missing_harness)
        return record

    record["inputs"] = {str(p): sha256_file(p) for p in stage.inputs}
    record["gate_sha256"] = {str(p): sha256_file(p) for p in stage.gate}

    env = dict(os.environ)
    env.update(stage.env)
    if tools.get("iverilog_dir"):
        env["PATH"] = tools["iverilog_dir"] + os.pathsep + env.get("PATH", "")
    if tools.get("iverilog"):
        env["IVERILOG"] = tools["iverilog"]
        env["VVP"] = os.path.join(tools["iverilog_dir"] or "", "vvp")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{stage.name}.log"
    started = time.time()
    try:
        proc = subprocess.run(stage.command, cwd=str(stage.env.get("KWT_GRAD_DESIGN_ROOT", ".")),
                              env=env, capture_output=True, text=True, timeout=stage.timeout)
        returncode, output = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        output = (exc.stdout or "") + (exc.stderr or "") if isinstance(exc.stdout, str) else ""
        output += f"\n[runner] timeout after {stage.timeout}s"
    except OSError as exc:
        returncode, output = 127, f"[runner] failed to execute: {exc}"
    record["duration_s"] = round(time.time() - started, 3)
    record["exit_code"] = returncode
    log_path.write_text(output, encoding="utf-8", errors="replace")
    record["log"] = str(log_path)
    tail = output.strip().splitlines()
    record["stdout_tail"] = "\n".join(tail[-25:])
    record["stderr_tail"] = ""

    status, reason = classify(returncode, output, stage)
    # the process succeeded but produced no comparison -> record separately
    record["status"] = status
    record["reason"] = reason
    return record


def _shell_quote(text: str) -> str:
    if re.fullmatch(r"[\w@%+=:,./-]+", text):
        return text
    return "'" + text.replace("'", "'\\''") + "'"


# --------------------------------------------------------------------------- #
# font-end
# --------------------------------------------------------------------------- #
def resolve_tools(args: argparse.Namespace, root: pathlib.Path) -> dict:
    iverilog, iverilog_version = find_iverilog(args.iverilog_bin)
    yosys, yosys_version = find_yosys(args.yosys_bin)
    e203_upstream = root / "资料/e203_hbirdv2_full"
    tools = {
        "python": sys.executable,
        "iverilog": iverilog,
        "iverilog_version": iverilog_version,
        "iverilog_dir": os.path.dirname(iverilog) if iverilog else None,
        "yosys": yosys,
        "yosys_version": yosys_version,
        "riscv64-linux-gnu-gcc": shutil.which("riscv64-linux-gnu-gcc"),
        "e203_upstream": str(e203_upstream) if e203_upstream.is_dir() else None,
        "python-numpy": None,
    }
    try:
        import numpy  # noqa: F401
        tools["python-numpy"] = sys.executable
    except ImportError:
        pass
    return tools


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design-root", type=pathlib.Path, default=DEFAULT_ROOT,
                        help="repository/design root to test (default: this worktree)")
    parser.add_argument("--stage", action="append", default=[],
                        help="stage name or 'all' (repeatable); default: all")
    parser.add_argument("--out", type=pathlib.Path,
                        default=HERE / "results/latest",
                        help="evidence output directory (default: graduate/verification/results/latest)")
    parser.add_argument("--timeout", type=int, default=None, help="per-stage timeout seconds")
    parser.add_argument("--iverilog-bin", default=None)
    parser.add_argument("--yosys-bin", default=None)
    parser.add_argument("--strict", action="store_true", help="exit non-zero on SKIP as well")
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    args = parser.parse_args(argv)

    root = args.design_root.resolve()
    if not root.is_dir():
        parser.error(f"--design-root is not a directory: {root}")
    tools = resolve_tools(args, root)
    stages = build_stages(root, tools)

    if args.list:
        for stage in stages:
            print(f"{stage.name:22s} {stage.kind:8s} requires={','.join(stage.requires) or '-'}")
        return 0

    wanted = {name.lower() for name in args.stage} or {"all"}
    selected = [s for s in stages if "all" in wanted or s.name.lower() in wanted]
    unknown = wanted - {"all"} - {s.name.lower() for s in stages}
    if unknown:
        parser.error("unknown stage(s): " + ", ".join(sorted(unknown)))
    if not selected:
        parser.error("no stage selected")

    if args.timeout is not None:
        for stage in selected:
            stage.timeout = args.timeout

    out_dir = args.out.resolve()
    results = []
    for stage in selected:
        print(f"== {stage.name} ({stage.kind}) ==", flush=True)
        record = run_stage(stage, tools, out_dir)
        results.append(record)
        print(f"   {record['status']}: {record['reason']}", flush=True)

    summary = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "design_root": str(root),
        "design_commit": git_commit(root),
        "runner": str(HERE / "run_stages.py"),
        "verdicts": {r["stage"]: r["status"] for r in results},
        "counts": {status: sum(1 for r in results if r["status"] == status)
                   for status in ("PASS", "SKIP", "FAIL")},
        "records": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"\nresults -> {out_dir}")
    print("counts:", summary["counts"])

    failures = summary["counts"]["FAIL"]
    skips = summary["counts"]["SKIP"]
    if failures:
        return 1
    if args.strict and skips:
        return 2
    return 0


def render_markdown(summary: dict) -> str:
    lines = [
        "# Graduate verification stage run",
        "",
        f"- generated (UTC): {summary['generated_utc']}",
        f"- design root: `{summary['design_root']}`",
        f"- design commit: `{summary['design_commit']}`",
        f"- counts: {summary['counts']}",
        "",
        "| stage | class | status | reason | exit | duration s |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for record in summary["records"]:
        lines.append(
            "| {stage} | {kind} | **{status}** | {reason} | {exit_code} | {duration_s} |".format(
                stage=record["stage"], kind=record["kind"], status=record["status"],
                reason=record["reason"].replace("|", "\\|"),
                exit_code=record["exit_code"], duration_s=record["duration_s"]))
    lines += ["", "## Per-stage evidence", ""]
    for record in summary["records"]:
        lines.append(f"### {record['stage']} ({record['kind']}) — {record['status']}")
        lines.append("")
        lines.append(f"- reason: {record['reason']}")
        lines.append(f"- command: `{record['command_shell']}`")
        lines.append(f"- exit code: {record['exit_code']}")
        lines.append(f"- duration s: {record['duration_s']}")
        lines.append(f"- sample identity: {record['sample_identity']}")
        lines.append(f"- comparison: {record['comparison']}")
        lines.append(f"- golden: {record['golden']}")
        if record.get("inputs"):
            lines.append("- input hashes:")
            for path, digest in record["inputs"].items():
                lines.append(f"  - `{path}` sha256={digest}")
        if record.get("gate_sha256"):
            lines.append("- design module hashes:")
            for path, digest in record["gate_sha256"].items():
                lines.append(f"  - `{path}` sha256={digest}")
        if record.get("log"):
            lines.append(f"- log: `{record['log']}`")
        if record["stdout_tail"]:
            lines.append("- output tail:")
            lines.append("")
            lines.append("```")
            lines.append(record["stdout_tail"])
            lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


if __name__ == "__main__":
    sys.exit(main())
