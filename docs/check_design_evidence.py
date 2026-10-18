#!/usr/bin/env python3
"""KWT-1 本科组整体设计：独立证据核对脚本（只读）。

核验对象（均可在任意 worktree 运行，用参数指定主目录结果路径）：

1. 12 层 ASCII 路径重跑的 E203 全层证据：
   - 每层 `layerNN.exit == 0`
   - 每层日志恰好一条全层 PASS，且 start/count/outputs/padding/cycles 与约定一致
   - 日志无 FATAL / mismatch / error / `xxxx`
   - 各层日志 SHA-256 与 `*_ascii_summary.json` 记录逐字一致
   - 摘要合计 checked_outputs=304128、checked_padding=15360、total_task_cycles=10056336
   - `*_ascii_finalizer.status` 为 `SUMMARIZE_EXIT 0`
2. 被纠正的旧运行（中文路径）仅作对照，必须仍显示 0/12 失败：
   - 旧 12 个 `.exit` 全为 1，旧日志含 `got=xxxxxxxx`
3. 报告引用的格式边界：
   - 向量 manifest 张量形状 X[99,64] / W[64,256] / bias[256] / Y[99,256]
   - hex 行数 a16=6656、w16=16384、bias32=256、expected32=25344
   - 向量文件 SHA-256 与 manifest 一致
   - 摘要 worst_layer_nrmse_pct 与 manifest 一致；协议含 A12W10
   - INT32 保守累加上界 < 2^31-1
4. 关键文件存在性：脚本、RTL、驱动、报告产物。

脚本只读，不写入共享数据，不修改任何文件。退出码：0=全部通过；1=存在 FAIL。

用法：
    python3 docs/check_design_evidence.py \
        --results-dir /home/yang/Desktop/ic设计竞赛/hardware/architecture/results
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

INT32_MAX = 2 ** 31 - 1
PASS_RE = re.compile(
    r"^PASS frozen real-sample E203 FFN1 tiles: "
    r"start=(\d+) count=(\d+) outputs=(\d+) padding=(\d+) "
    r"task_cycles=(\d+) total_cycles=(\d+)$",
    re.MULTILINE,
)
EXPECTED_PER_LAYER = dict(start=0, count=208, outputs=25344, padding=1280,
                          task_cycles=838028, total_cycles=838041)
EXPECTED_TOTALS = dict(checked_outputs=304128, checked_padding=15360,
                       total_task_cycles=10056336)
HEX_LINES = {"a16": 6656, "w16": 16384, "bias32": 256, "expected32": 25344}
EXPECTED_SHAPES = {"X": [99, 64], "W": [64, 256], "bias": [256], "Y": [99, 256]}


class Checks:
    def __init__(self):
        self.rows = []
        self.fail = 0
        self.warn = 0

    def add(self, level, name, detail=""):
        self.rows.append((level, name, detail))
        if level == "FAIL":
            self.fail += 1
        elif level == "WARN":
            self.warn += 1

    def ok(self, name, cond, detail=""):
        self.add("PASS" if cond else "FAIL", name, detail)
        return cond

    def report(self):
        width = max(len(n) for _, n, _ in self.rows) if self.rows else 0
        for level, name, detail in self.rows:
            line = f"[{level}] {name.ljust(width)}"
            if detail:
                line += "  " + detail
            print(line)
        print("-" * 72)
        npass = sum(1 for r in self.rows if r[0] == "PASS")
        print(f"SUMMARY: PASS={npass} WARN={self.warn} FAIL={self.fail}")
        return self.fail == 0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def check_current_ascii(c, results: Path, repo: Path):
    logs = results / "real_e203_validation0_ascii_logs"
    summary_path = results / "real_e203_validation0_ascii_summary.json"
    status_path = results / "real_e203_validation0_ascii_finalizer.status"
    runlog_path = results / "real_e203_validation0_ascii_summary_run.log"
    vectors = results / "real_e203_validation0_vectors"

    status = status_path.read_text() if status_path.is_file() else ""
    c.ok("ascii/finalizer.status == SUMMARIZE_EXIT 0",
         "SUMMARIZE_EXIT 0" in status, status.strip().replace("\n", " | "))
    runlog = runlog_path.read_text() if runlog_path.is_file() else ""
    c.ok("ascii/summary_run.log 报告 12 个 PASS 任务",
         "PASS 12 full E203 tasks" in runlog and "304128 outputs" in runlog
         and "15360 padding" in runlog, runlog.strip()[:120])

    ok_summary = summary_path.is_file()
    if not c.ok("ascii/summary.json 存在", ok_summary):
        return
    s = read_json(summary_path)

    layers = s.get("layers", [])
    c.ok("ascii/摘要含 12 层且层号为 1..12",
         [x.get("layer") for x in layers] == list(range(1, 13)),
         f"layers={len(layers)}")

    # 逐层日志、退出码、PASS、SHA-256
    for entry in layers:
        layer = entry.get("layer")
        stem = f"layer{layer:02d}"
        exit_path = logs / f"{stem}.exit"
        log_path = logs / f"{stem}.log"
        exit_ok = exit_path.is_file() and exit_path.read_text().strip() == "0"
        c.ok(f"ascii/{stem}.exit == 0", exit_ok,
             exit_path.read_text().strip() if exit_path.is_file() else "missing")
        if not log_path.is_file():
            c.add("FAIL", f"ascii/{stem}.log 存在", "missing")
            continue
        text = log_path.read_text()
        matches = PASS_RE.findall(text)
        want = (str(EXPECTED_PER_LAYER["start"]), str(EXPECTED_PER_LAYER["count"]),
                str(EXPECTED_PER_LAYER["outputs"]), str(EXPECTED_PER_LAYER["padding"]),
                str(EXPECTED_PER_LAYER["task_cycles"]), str(EXPECTED_PER_LAYER["total_cycles"]))
        c.ok(f"ascii/{stem}.log 恰好一条约定 PASS",
             len(matches) == 1 and matches[0] == want,
             f"matches={len(matches)}")
        bad = [w for w in ("FATAL", "mismatch", "error", "xxxx") if w.lower() in text.lower()]
        c.ok(f"ascii/{stem}.log 无错误标记", not bad, ",".join(bad))
        c.ok(f"ascii/{stem}.log SHA-256 与摘要一致",
             sha256(log_path) == entry.get("log_sha256"), entry.get("log_sha256", "")[:16])

    # 摘要合计
    for key, want in EXPECTED_TOTALS.items():
        c.ok(f"ascii/摘要 {key} == {want}", s.get(key) == want, str(s.get(key)))

    # 向量 manifest 与格式边界
    mpath = vectors / "manifest.json"
    if not c.ok("ascii/向量 manifest.json 存在", mpath.is_file()):
        return
    m = read_json(mpath)
    c.ok("边界/manifest 张量形状 X[99,64] W[64,256] bias[256] Y[99,256]",
         m.get("tensor_shapes") == EXPECTED_SHAPES, json.dumps(m.get("tensor_shapes")))
    c.ok("边界/数值协议含 A12W10 与 INT32",
         "A12W10" in m.get("numerical_protocol", "") and "INT32" in m.get("numerical_protocol", ""),
         m.get("numerical_protocol", ""))
    c.ok("边界/摘要 worst_layer_nrmse_pct 与 manifest 一致",
         s.get("worst_layer_nrmse_pct") == m.get("worst_layer_nrmse_pct"),
         str(s.get("worst_layer_nrmse_pct")))

    manifest_layers = {x["layer"]: x for x in m.get("layers", [])}
    c.ok("边界/manifest 含 12 层", sorted(manifest_layers) == list(range(1, 13)))
    hex_ok = True
    detail = []
    for layer, entry in manifest_layers.items():
        for kind, want_lines in HEX_LINES.items():
            rec = entry.get("files", {}).get(kind)
            if rec is None:
                hex_ok = False
                detail.append(f"layer{layer:02d}:{kind} missing")
                continue
            p = Path(rec["path"])
            if not p.is_file():
                hex_ok = False
                detail.append(f"layer{layer:02d}:{kind} file missing")
                continue
            lines = sum(1 for _ in p.open("rb"))
            if lines != want_lines:
                hex_ok = False
                detail.append(f"layer{layer:02d}:{kind} lines={lines}")
            if sha256(p) != rec.get("sha256"):
                hex_ok = False
                detail.append(f"layer{layer:02d}:{kind} sha mismatch")
    c.ok("边界/全部向量 hex 行数与 SHA-256 正确", hex_ok, "; ".join(detail[:4]))

    # INT32 边界
    bounds = [x.get("conservative_abs_accumulator_bound", 0) for x in m.get("layers", [])]
    max_abs = [x.get("max_abs_expected_int32", 0) for x in m.get("layers", [])]
    c.ok("边界/A12W10 保守累加上界 < 2^31-1",
         bounds and max(bounds) < INT32_MAX,
         f"max={max(bounds) if bounds else 'n/a'} ({max(bounds)/INT32_MAX*100:.2f}% of INT32)")
    c.ok("边界/实测最大 |expected INT32| < 2^31-1",
         max_abs and max(max_abs) < INT32_MAX, f"max={max(max_abs) if max_abs else 'n/a'}")


def check_superseded_old(c, results: Path):
    """旧的中文路径运行必须仍显示 0/12 失败，用于证明纠正的是'当前状态'描述。"""
    old_logs = results / "real_e203_validation0_logs"
    status_path = results / "real_e203_validation0_finalizer.status"
    if not old_logs.is_dir():
        c.add("WARN", "superseded/旧运行目录缺失", str(old_logs))
        return
    status = status_path.read_text() if status_path.is_file() else ""
    c.ok("superseded/旧 finalizer.status == SUMMARIZE_EXIT 1",
         "SUMMARIZE_EXIT 1" in status, status.strip().replace("\n", " | "))
    exits = []
    for i in range(1, 13):
        p = old_logs / f"layer{i:02d}.exit"
        exits.append(p.read_text().strip() if p.is_file() else "missing")
    c.ok("superseded/旧 12 层退出码全为 1（0/12 失败）",
         all(e == "1" for e in exits), ",".join(exits))
    fatals = 0
    for i in range(1, 13):
        p = old_logs / f"layer{i:02d}.log"
        if p.is_file() and "got=xxxxxxxx" in p.read_text():
            fatals += 1
    c.ok("superseded/旧日志出现 got=xxxxxxxx", fatals == 12, f"{fatals}/12")


def check_other_boundaries(c, repo: Path):
    """报告引用的其他格式/边界产物与关键文件存在性。"""
    # 240 条 RTL（计算核边界）摘要
    p240 = repo / "hardware/architecture/quantized_validation240_a12w10_rtl_summary.json"
    if p240.is_file():
        s = read_json(p240)
        c.ok("边界/240 条 A12W10 RTL 摘要 samples=240 且 per-layer=240*25344",
             s.get("samples") == 240 and
             all(x.get("checked_outputs") == 240 * 25344 for x in s.get("layers", [])),
             f"worst={s.get('worst_nrmse_pct')}")
        c.ok("边界/240 条 RTL pass_3pct 为真", s.get("pass_3pct") is True)
    else:
        c.add("WARN", "边界/240 条 RTL 摘要缺失", str(p240))

    # 16 行 staged 计算核边界：154,240 拍/层（L1，测试台定时）
    p16 = repo / "hardware/architecture/staged_task_row16_summary.json"
    if p16.is_file():
        cases = read_json(p16)
        full = [x for x in cases if x.get("name") == "full"]
        c.ok("边界/16 行 staged 计算核 full = 154240 拍（L1，非 E203）",
             bool(full) and full[0].get("cycles") == 154240,
             f"cycles={full[0].get('cycles') if full else 'n/a'}")
    else:
        c.add("WARN", "边界/16 行 staged 摘要缺失", str(p16))

    # 关键文件存在性
    required = [
        "hardware/rtl/control/tb/run_e203.sh",
        "hardware/rtl/control/tb/tb_kwt_real_layer_e203.sv",
        "hardware/rtl/control/tb/summarize_real_e203.py",
        "hardware/rtl/control/tb/e203_kwt_full_layer.S",
        "hardware/rtl/control/kwt_tile_system_model.sv",
        "hardware/rtl/control/kwt_nice_system_model.sv",
        "hardware/rtl/pulse_array/kwt_pingpong_cacc_engine.sv",
        "hardware/rtl/sram/kwt_operand_pingpong.sv",
        "hardware/rtl/dma/kwt_dma.sv",
        "hardware/rtl/dma/kwt_dma_2d.sv",
        "software/driver/kwt_nice_driver.c",
        "software/driver/kwt_nice_driver.h",
        "software/driver/test_kwt_nice_driver.c",
        "hardware/architecture/run_quantized_ffn1_batch_rtl.py",
        "docs/check_design_evidence.py",
    ]
    missing = [r for r in required if not (repo / r).is_file()]
    c.ok("存在性/关键脚本与 RTL 文件齐全", not missing, ",".join(missing) or "all present")

    # 证据分层声明：摘要把硬件边界写成 E203 + 8x16（与 16 行 L1 区分）
    sp = repo / "hardware/architecture/claude_review_check.md"
    c.ok("存在性/Claude 评审核实文档存在", sp.is_file(), str(sp.name))

    # A14W12（仅软件证据，本 worktree 未合并 RTL 产物）
    a14 = list((repo / "software/results").glob("*a14w12*")) + \
          list((repo / "software/results").glob("*A14W12*"))
    if a14:
        c.add("INFO", "A14W12/本 worktree 存在软件证据文件", ",".join(p.name for p in a14[:3]))
    else:
        c.add("INFO", "A14W12/本 worktree 无 A14W12 RTL 或软件产物",
              "A14W12 只有 quant 分支的软件证据，不能当作已实现硬件")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path,
                    default=Path("/home/yang/Desktop/ic设计竞赛/hardware/architecture/results"),
                    help="主目录 architecture/results（只读）")
    ap.add_argument("--repo", type=Path, default=None,
                    help="被核对仓库根（默认脚本所在 worktree 根）")
    args = ap.parse_args()

    repo = args.repo or Path(__file__).resolve().parent.parent
    results = args.results_dir

    print(f"repo    = {repo}")
    print(f"results = {results}")
    print("=" * 72)

    c = Checks()
    c.ok("results 目录存在", results.is_dir(), str(results))
    if results.is_dir():
        check_current_ascii(c, results, repo)
        check_superseded_old(c, results)
    check_other_boundaries(c, repo)

    passed = c.report()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
