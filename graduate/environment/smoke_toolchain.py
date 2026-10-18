#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""研究生组第 4-7 项：工具链 smoke test（真实执行，记录真实退出码）。

与 ``check_environment.py`` 的分工：
- check_environment.py 只做"存在性/版本/哈希"的只读盘点；
- 本脚本真实跑最小用例，证明工具链能用，且项目 RTL 能被本项目工具读到。

每个用例记录命令行、退出码、关键输出与判定。工具缺失时状态为
``skipped_missing_tool``，绝不写 PASS；用例本身失败时状态为 ``fail``。

输出 ``smoke_test.json`` / ``smoke_test.md`` / ``smoke_test.log``。

用法::

    python3 graduate/environment/smoke_toolchain.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

OSS_CAD = os.environ.get("OSS_CAD_SUITE", "/home/yang/yosys-sta/oss-cad-suite")

COUNTER_V = """\
module counter(input clk, input rst_n, output reg [7:0] q);
  always @(posedge clk or negedge rst_n)
    if (!rst_n) q <= 8'h00; else q <= q + 8'd1;
endmodule
"""

TB_V = """\
module tb;
  reg clk = 0, rst_n = 0;
  wire [7:0] q;
  counter u(.clk(clk), .rst_n(rst_n), .q(q));
  always #5 clk = ~clk;
  initial begin
    #12 rst_n = 1;
    #100 $display("SMOKE_Q=%0d", q);
    $finish;
  end
endmodule
"""

TINY_C = """\
int f(int a, int b) { return a * b + 7; }
int main(void) { return f(3, 4); }
"""

REPO_RTL_CANDIDATES = [
    "hardware/rtl/sram/kwt_operand_pingpong.sv",
    "hardware/rtl/sram/kwt_tile_buffers.sv",
]


def which(cmd: str) -> str | None:
    if os.sep in cmd:
        return cmd if Path(cmd).is_file() and os.access(cmd, os.X_OK) else None
    return shutil.which(cmd)


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 60) -> dict:
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
            timeout=timeout, errors="replace",
        )
        return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "error": None}
    except subprocess.TimeoutExpired:
        return {"exit_code": None, "stdout": "", "stderr": "", "error": "timeout"}
    except FileNotFoundError:
        return {"exit_code": None, "stdout": "", "stderr": "", "error": "not found"}
    except Exception as exc:  # pragma: no cover
        return {"exit_code": None, "stdout": "", "stderr": "", "error": str(exc)}


def tail(text: str, n: int = 6) -> list[str]:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return lines[-n:]


def make_record(test_id: str, name: str, tool: str) -> dict:
    return {
        "id": test_id,
        "name": name,
        "tool": tool,
        "status": "not_run",
        "commands": [],
        "exit_codes": [],
        "evidence": [],
        "detail": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix="kwt_env_smoke.", dir="/tmp/opencode"))
    log_lines: list[str] = []
    results: list[dict] = []

    def log(msg: str) -> None:
        log_lines.append(msg)

    log(f"# smoke work dir: {work}")
    log(f"# repo: {repo}")

    (work / "counter.v").write_text(COUNTER_V, encoding="utf-8")
    (work / "tb.v").write_text(TB_V, encoding="utf-8")
    (work / "tiny.c").write_text(TINY_C, encoding="utf-8")

    # ---- 1. Verilator 独立编译+仿真 -------------------------------------- #
    rec = make_record("verilator_sim", "Verilator 最小计数器编译并运行", "verilator")
    vbin = which("verilator")
    if not vbin:
        rec["status"] = "skipped_missing_tool"
        rec["detail"] = "verilator 不在 PATH"
    else:
        rec["commands"].append(f"verilator --binary --timing -Wno-fatal -j 0 -o sim counter.v tb.v")
        c = run([vbin, "--binary", "--timing", "-Wno-fatal", "-j", "0", "-o", "sim", "counter.v", "tb.v"], cwd=work)
        rec["exit_codes"].append(c["exit_code"])
        log(f"[verilator compile] exit={c['exit_code']}\n{tail(c['stdout']+c['stderr'], 10)}")
        if c["exit_code"] == 0:
            sim = work / "obj_dir" / "sim"
            rec["commands"].append("./obj_dir/sim")
            r = run([str(sim)], cwd=work)
            rec["exit_codes"].append(r["exit_code"])
            out = r["stdout"] + r["stderr"]
            log(f"[verilator run] exit={r['exit_code']}\n{tail(out, 5)}")
            rec["evidence"] = tail(out, 3)
            rec["status"] = "pass" if (r["exit_code"] == 0 and "SMOKE_Q=" in out) else "fail"
            rec["detail"] = None if rec["status"] == "pass" else "仿真未输出 SMOKE_Q 或退出码非 0"
        else:
            rec["status"] = "fail"
            rec["evidence"] = tail(c["stdout"] + c["stderr"], 6)
            rec["detail"] = "Verilator 编译失败"
    results.append(rec)

    # ---- 2. Verilator 对项目 RTL 做 lint --------------------------------- #
    rec = make_record("verilator_repo_lint", "Verilator 对仓库内 SRAM RTL 做 lint", "verilator")
    repo_rtl = [p for p in REPO_RTL_CANDIDATES if (repo / p).is_file()]
    if not vbin:
        rec["status"] = "skipped_missing_tool"
        rec["detail"] = "verilator 不在 PATH"
    elif not repo_rtl:
        rec["status"] = "skipped_missing_input"
        rec["detail"] = "仓库内未找到候选 RTL"
    else:
        target = repo_rtl[0]
        rec["commands"].append(f"verilator --lint-only -Wno-fatal -Ihardware/rtl/sram {target}")
        c = run(
            [vbin, "--lint-only", "-Wno-fatal", "-Wall", "-Ihardware/rtl/sram", target],
            cwd=repo,
        )
        rec["exit_codes"].append(c["exit_code"])
        log(f"[verilator repo lint] exit={c['exit_code']} target={target}\n{tail(c['stdout']+c['stderr'], 8)}")
        rec["evidence"] = tail(c["stdout"] + c["stderr"], 3)
        rec["status"] = "pass" if c["exit_code"] == 0 else "fail"
        rec["detail"] = None if rec["status"] == "pass" else f"lint 退出码 {c['exit_code']}"
    results.append(rec)

    # ---- 3. Icarus 独立编译+仿真 ----------------------------------------- #
    rec = make_record("iverilog_sim", "Icarus Verilog 最小计数器编译并运行", "iverilog")
    ibin = which("iverilog")
    vvp = which("vvp")
    if not ibin or not vvp:
        rec["status"] = "skipped_missing_tool"
        rec["detail"] = "iverilog 或 vvp 不在 PATH"
    else:
        rec["commands"].append("iverilog -g2012 -o isim counter.v tb.v")
        c = run([ibin, "-g2012", "-o", "isim", "counter.v", "tb.v"], cwd=work)
        rec["exit_codes"].append(c["exit_code"])
        log(f"[iverilog compile] exit={c['exit_code']}\n{tail(c['stdout']+c['stderr'], 6)}")
        if c["exit_code"] == 0:
            rec["commands"].append("vvp isim")
            r = run([vvp, "isim"], cwd=work)
            rec["exit_codes"].append(r["exit_code"])
            out = r["stdout"] + r["stderr"]
            log(f"[iverilog run] exit={r['exit_code']}\n{tail(out, 5)}")
            rec["evidence"] = tail(out, 3)
            rec["status"] = "pass" if (r["exit_code"] == 0 and "SMOKE_Q=" in out) else "fail"
            rec["detail"] = None if rec["status"] == "pass" else "仿真未输出 SMOKE_Q 或退出码非 0"
        else:
            rec["status"] = "fail"
            rec["evidence"] = tail(c["stdout"] + c["stderr"], 6)
            rec["detail"] = "iverilog 编译失败"
    results.append(rec)

    # ---- 4. Yosys 独立综合 ----------------------------------------------- #
    rec = make_record("yosys_synth", "Yosys 最小计数器综合并统计", "yosys")
    ybin = which(f"{OSS_CAD}/bin/yosys") or which("yosys")
    if not ybin:
        rec["status"] = "skipped_missing_tool"
        rec["detail"] = "yosys 不在 PATH"
    else:
        rec["commands"].append(f"{ybin} -p 'read_verilog counter.v; synth -top counter; stat'")
        c = run([ybin, "-p", "read_verilog counter.v; synth -top counter; stat"], cwd=work)
        rec["exit_codes"].append(c["exit_code"])
        out = c["stdout"] + c["stderr"]
        log(f"[yosys synth] exit={c['exit_code']} bin={ybin}\n{tail(out, 8)}")
        rec["evidence"] = [ln for ln in out.splitlines() if "cells" in ln.lower()][-2:] or tail(out, 3)
        rec["status"] = "pass" if c["exit_code"] == 0 else "fail"
        rec["detail"] = None if rec["status"] == "pass" else f"综合退出码 {c['exit_code']}"
    results.append(rec)

    # ---- 5. Yosys 读项目 SystemVerilog RTL ------------------------------- #
    rec = make_record("yosys_repo_read", "Yosys(oss-cad-suite) 读入仓库内 SRAM SystemVerilog RTL", "yosys")
    oss_yosys = which(f"{OSS_CAD}/bin/yosys")
    if not oss_yosys:
        rec["status"] = "skipped_missing_tool"
        rec["detail"] = "oss-cad-suite yosys 不存在"
    elif not repo_rtl:
        rec["status"] = "skipped_missing_input"
        rec["detail"] = "仓库内未找到候选 RTL"
    else:
        target = repo_rtl[0]
        top = Path(target).stem
        cmd = ["-p", f"read_verilog -sv -Ihardware/rtl/sram {target}; hierarchy -top {top} -check; proc"]
        rec["commands"].append(f"{oss_yosys} " + " ".join(cmd))
        c = run([oss_yosys] + cmd, cwd=repo)
        rec["exit_codes"].append(c["exit_code"])
        log(f"[yosys repo read] exit={c['exit_code']} target={target}\n{tail(c['stdout']+c['stderr'], 8)}")
        rec["evidence"] = tail(c["stdout"] + c["stderr"], 3)
        rec["status"] = "pass" if c["exit_code"] == 0 else "fail"
        rec["detail"] = None if rec["status"] == "pass" else f"读入退出码 {c['exit_code']}"
    results.append(rec)

    # ---- 6. iEDA STA/综合工具可执行 -------------------------------------- #
    rec = make_record("ieda_version", "iEDA 可执行（STA/综合流程）", "iEDA")
    ieda = which("/home/yang/yosys-sta/bin/iEDA") or which("iEDA")
    if not ieda:
        rec["status"] = "skipped_missing_tool"
        rec["detail"] = "iEDA 不在 PATH"
    else:
        rec["commands"].append(f"{ieda} -version")
        c = run([ieda, "-version"], cwd=work)
        rec["exit_codes"].append(c["exit_code"])
        out = (c["stdout"] + c["stderr"]).strip()
        log(f"[iEDA version] exit={c['exit_code']} out={out!r}")
        rec["evidence"] = [f"exit={c['exit_code']}", f"stdout/stderr={out!r}"]
        # iEDA 这类二进制常直接返回 0 且不打印版本；只有非 0/异常才算 fail。
        if c["exit_code"] == 0:
            rec["status"] = "pass"
            rec["detail"] = "可执行且退出码 0；无版本字符串输出（如实记录，不代表 STA 已跑通）"
        else:
            rec["status"] = "fail"
            rec["detail"] = f"退出码 {c['exit_code']} error={c['error']}"
    results.append(rec)

    # ---- 7. RV32 交叉编译 + 目标架构核对 --------------------------------- #
    rec = make_record("rv32_compile", "RV32 交叉编译最小 C 并核对 ELF 架构", "rv32_gcc")
    gcc = None
    for cand in ("riscv64-linux-gnu-gcc", "riscv32-unknown-elf-gcc", "riscv64-unknown-elf-gcc"):
        if which(cand):
            gcc = which(cand)
            break
    objdump = which("riscv64-linux-gnu-objdump")
    if not gcc:
        rec["status"] = "skipped_missing_tool"
        rec["detail"] = "未找到 riscv gcc"
    else:
        rec["commands"].append(f"{gcc} -march=rv32imac -mabi=ilp32 -O2 -c tiny.c -o tiny.o")
        c = run([gcc, "-march=rv32imac", "-mabi=ilp32", "-O2", "-c", "tiny.c", "-o", "tiny.o"], cwd=work)
        rec["exit_codes"].append(c["exit_code"])
        log(f"[rv32 compile] exit={c['exit_code']}\n{tail(c['stdout']+c['stderr'], 6)}")
        if c["exit_code"] == 0 and objdump:
            rec["commands"].append(f"{objdump} -f tiny.o")
            r = run([objdump, "-f", "tiny.o"], cwd=work)
            rec["exit_codes"].append(r["exit_code"])
            out = r["stdout"] + r["stderr"]
            log(f"[rv32 objdump] exit={r['exit_code']}\n{tail(out, 5)}")
            rec["evidence"] = [ln for ln in out.splitlines() if "architecture" in ln or "elf32" in ln]
            ok = r["exit_code"] == 0 and any("riscv:rv32" in ln for ln in out.splitlines())
            rec["status"] = "pass" if ok else "fail"
            rec["detail"] = None if ok else "ELF 架构不是 riscv:rv32"
        elif c["exit_code"] == 0:
            rec["status"] = "pass"
            rec["detail"] = "已生成目标文件（未找到 objdump）"
        else:
            rec["status"] = "fail"
            rec["evidence"] = tail(c["stdout"] + c["stderr"], 6)
            rec["detail"] = "RV32 编译失败"
    results.append(rec)

    log_path = outdir / "smoke_test.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    counts = {"pass": 0, "fail": 0, "skipped_missing_tool": 0, "skipped_missing_input": 0, "not_run": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo),
        "work_dir": str(work),
        "note": "所有状态来自真实命令退出码；缺失工具记为 skipped_missing_tool，不写 PASS。",
        "counts": counts,
        "tests": results,
    }
    (outdir / "smoke_test.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# 工具链 smoke test（真实退出码）", ""]
    lines.append(f"- 生成时间（UTC）：{report['generated_at']}")
    lines.append(f"- 工作目录：`{work}`")
    lines.append(f"- 汇总：pass={counts['pass']} fail={counts['fail']} "
                 f"skipped_missing_tool={counts['skipped_missing_tool']} "
                 f"skipped_missing_input={counts['skipped_missing_input']}")
    lines.append("")
    lines.append("| 用例 | 工具 | 状态 | 退出码 | 证据 |")
    lines.append("| --- | --- | :-: | --- | --- |")
    for r in results:
        ev = " / ".join(r["evidence"])[:120].replace("|", "/")
        lines.append(f"| {r['name']} | {r['tool']} | **{r['status']}** | {r['exit_codes']} | {ev} |")
    lines.append("")
    (outdir / "smoke_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[smoke] json -> {outdir / 'smoke_test.json'}")
    print(f"[smoke] pass={counts['pass']} fail={counts['fail']} "
          f"skipped_missing_tool={counts['skipped_missing_tool']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
