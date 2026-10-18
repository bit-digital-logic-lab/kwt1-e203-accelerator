#!/usr/bin/env python3
"""Read-only fact checker for the KWT-1 next-stage design review.

It never writes to the repository or the shared results. It only:
  * calls `git` for metadata (merge-base / diff / show),
  * reads local files and hashes them.

Usage (from the worktree root):
    python3 docs/check_next_stage_facts.py \
        --repo /home/yang/Desktop/ic设计竞赛 \
        --design-doc /home/yang/kwt-worktrees/design-integration/docs/KWT1_整体设计决策_20261018.md

Exit code 0 when every *structural expectation* below holds; 1 otherwise.
Observations that merely document a known inconsistency are reported as
FINDING and do not change the exit code.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

BASE = "d94fb6f"
WORKER_BRANCHES = [
    "codex/kwt-architecture",
    "codex/kwt-driver",
    "codex/kwt-hook-fix",
    "codex/kwt-packaging",
    "codex/kwt-preprocess",
    "codex/kwt-quant",
    "codex/kwt-synth",
    "codex/kwt-verification",
]
ALL_BRANCHES = WORKER_BRANCHES + ["codex/kwt-design-integration"]


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, kind: str, name: str, detail: str) -> None:
        self.rows.append((kind, name, detail))

    def dump(self) -> int:
        fails = sum(1 for k, _, _ in self.rows if k == "FAIL")
        for kind, name, detail in self.rows:
            print(f"[{kind}] {name}  {detail}")
        print("-" * 72)
        print(f"SUMMARY: checks={len(self.rows)} FAIL={fails}")
        return 1 if fails else 0


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=Path("/home/yang/Desktop/ic设计竞赛"))
    parser.add_argument("--design-doc", type=Path, default=None)
    args = parser.parse_args()
    repo: Path = args.repo
    report = Report()

    # 1. base commit and per-branch merge-base.
    for branch in ALL_BRANCHES:
        try:
            mb = git(repo, "merge-base", branch, BASE).strip()
        except RuntimeError:
            report.add("FAIL", f"分支 {branch}", "不存在或无法解析 merge-base")
            continue
        if mb.startswith(BASE):
            report.add("PASS", f"分支基座 {branch}", f"merge-base={mb[:12]}")
        else:
            report.add("FAIL", f"分支基座 {branch}",
                       f"merge-base={mb[:12]} != {BASE}")

    # 2. WORKER_REPORT.md is added by every branch -> add/add collision.
    base_has_report = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{BASE}:WORKER_REPORT.md"],
        capture_output=True).returncode == 0
    report.add("PASS" if not base_has_report else "FAIL",
               "基线是否已含 WORKER_REPORT.md",
               "无（各分支各自新增）" if not base_has_report else "已含")
    blobs = {}
    for branch in ALL_BRANCHES:
        try:
            blob = git(repo, "rev-parse", f"{branch}:WORKER_REPORT.md").strip()
        except RuntimeError:
            blob = ""
        blobs[branch] = blob
    distinct = {b for b in blobs.values() if b}
    if len(distinct) == len(ALL_BRANCHES):
        report.add("FINDING", "WORKER_REPORT.md 冲突面",
                   f"{len(ALL_BRANCHES)} 个分支各写根文件、blob 互不相同 -> 顺序 merge 必产 add/add 冲突")
    else:
        report.add("INFO", "WORKER_REPORT.md 冲突面", f"distinct={len(distinct)}")

    # 3. main working run_e203.sh identical to base (no real conflict there).
    work = repo / "hardware/rtl/control/tb/run_e203.sh"
    try:
        base_bytes = subprocess.run(
            ["git", "-C", str(repo), "show", f"{BASE}:hardware/rtl/control/tb/run_e203.sh"],
            capture_output=True).stdout
    except Exception as exc:  # pragma: no cover - defensive
        base_bytes = b""
        report.add("FAIL", "run_e203.sh 基线", str(exc))
    if work.exists() and base_bytes:
        same = sha256_bytes(work.read_bytes()) == sha256_bytes(base_bytes)
        report.add("FINDING" if same else "INFO", "run_e203.sh 主目录工作副本 vs 基线",
                   "逐字节相同（选定基线 d94fb6f 下无文本冲突）" if same
                   else "不同（需人工核对）")

    # 4. old vs corrected E203 12-layer exit status.
    res = repo / "hardware/architecture/results"
    old_status = res / "real_e203_validation0_finalizer.status"
    new_status = res / "real_e203_validation0_ascii_finalizer.status"
    for label, path, want in (("旧(中文路径)运行", old_status, "SUMMARIZE_EXIT 1"),
                              ("修正(ASCII)运行", new_status, "SUMMARIZE_EXIT 0")):
        if path.exists():
            text = path.read_text().strip()
            ok = want in text
            report.add("PASS" if ok else "FAIL", f"{label} finalizer", text.replace("\n", " | "))
        else:
            report.add("INFO", f"{label} finalizer", "文件缺失")
    for label, sub, want in (("旧", "real_e203_validation0_logs", "1"),
                             ("修正", "real_e203_validation0_ascii_logs", "0")):
        exits = sorted((res / sub).glob("layer*.exit"))
        vals = {p.read_text().strip() for p in exits}
        ok = len(exits) == 12 and vals == {want}
        report.add("PASS" if ok else "FAIL", f"{label}运行 12 层 .exit",
                   f"count={len(exits)} values={sorted(vals)}")

    # 5. A12W10 conservative bound: manifest value vs design-doc claim.
    manifest = res / "real_e203_validation0_vectors/manifest.json"
    if manifest.exists():
        text = manifest.read_text()
        bounds = [int(v) for v in re.findall(
            r'"conservative_abs_accumulator_bound":\s*(\d+)', text)]
        if bounds:
            report.add("PASS", "E203 向量 manifest A12W10 上界",
                       f"per-layer max={max(bounds)} (n={len(bounds)})")
    if args.design_doc and args.design_doc.exists():
        doc = args.design_doc.read_text()
        has_a = "67,780,536" in doc
        has_b = "67,826,716" in doc
        if has_a and has_b:
            report.add("FINDING", "设计报告 A12W10 上界自相矛盾",
                       "同文同时出现 67,780,536 与 67,826,716")
        else:
            report.add("INFO", "设计报告 A12W10 上界", f"a={has_a} b={has_b}")

    # 6. E203 testbench: ITCM load capped at 64 B, CPU system bus stubbed.
    tb = repo / "hardware/rtl/control/tb/tb_kwt_real_layer_e203.sv"
    if tb.exists():
        text = tb.read_text()
        capped = "[0:63]" in text and "i < 64" in text
        report.add("FINDING" if capped else "INFO", "E203 TB 程序装载上限",
                   "仅 64 B ITCM（装不下编译后的 C 驱动）" if capped else "未检出 64 B 上限")
    cpu = repo / "hardware/rtl/control/tb/e203_cpu_instance.svh"
    if cpu.exists():
        text = cpu.read_text()
        stubbed = ".mem_icb_cmd_ready(1'b0)" in text
        report.add("FINDING" if stubbed else "INFO", "CPU 系统总线",
                   "mem_icb 置零：CPU 无到共享 RAM 的数据通路" if stubbed
                   else "未见 mem_icb 置零")

    # 7. design-doc branch table omits synth and miscounts.
    if args.design_doc and args.design_doc.exists():
        doc = args.design_doc.read_text()
        mentions_synth = "kwt-synth" in doc
        report.add("FINDING" if not mentions_synth else "INFO",
                   "设计报告整合顺序是否含 synth 分支",
                   "未出现 codex/kwt-synth（遗漏综合分支）" if not mentions_synth else "已提及")

    return report.dump()


if __name__ == "__main__":
    sys.exit(main())
