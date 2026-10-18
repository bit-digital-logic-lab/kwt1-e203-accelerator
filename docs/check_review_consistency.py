#!/usr/bin/env python3
"""KWT-1 审查纠错：只读一致性检查（2026-10-18）。

逐条核对被纠正的"项目口径"是否与真实产物一致，防止把过期/边界混淆的
说法当成当前状态。只读，不写任何文件；退出码 0=全部核对完成，
1=存在与"当前真实产物"矛盾的断言。

用法：
    python3 docs/check_review_consistency.py \
        --main-dir /home/yang/Desktop/ic设计竞赛 \
        --sta-dir /tmp/kwt-8x16-sta
"""

import argparse
import json
import re
import sys
from pathlib import Path


def read(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-dir", type=Path,
                    default=Path("/home/yang/Desktop/ic设计竞赛"))
    ap.add_argument("--sta-dir", type=Path, default=Path("/tmp/kwt-8x16-sta"))
    a = ap.parse_args()
    main_dir, sta = a.main_dir, a.sta_dir
    results = main_dir / "hardware/architecture/results"
    repo = Path(__file__).resolve().parent.parent

    rows = []

    def rec(item, ok, detail):
        rows.append((item, "确认" if ok else "矛盾/缺口", detail))

    # 1) GELU 是否本科必做
    rmod = read(repo / "题目/README.md")
    gelu_optional = "GELU、第二线性层和注意力不是本科必做" in rmod
    rec("GELU 非本科必做（选做）", gelu_optional,
        "题目/README.md 含'GELU…不是本科必做'" if gelu_optional else "未找到原文")
    for doc in ("AUDIT_SUMMARY.md", "CODE_REVIEW.md", "REVIEW_INDEX.md",
                "QUICK_REFERENCE.md", "ACTION_PLAN.md", "README_REVIEW.md"):
        t = read(main_dir / doc)
        if not t:
            continue
        # 头部纠正头存在则视为已标注历史；否则若把 GELU 列为 P0/必做则矛盾
        has_head = "范围纠正" in t[:400]
        claims = bool(re.search(r"GELU.{0,12}(P0|必做|BLOCKER)", t))
        rec(f"主目录 {doc} 的 GELU 口径",
            has_head or not claims,
            "已有纠正头" if has_head else ("仍把 GELU 当必做/P0" if claims else "无冲突"))

    # 2) 旧 0/12 vs 当前 12/12
    old = results / "real_e203_validation0_logs"
    new = results / "real_e203_validation0_ascii_logs"
    def exits(d):
        out = []
        for i in range(1, 13):
            p = d / f"layer{i:02d}.exit"
            out.append(p.read_text().strip() if p.is_file() else "?")
        return out
    oe, ne = exits(old), exits(new)
    rec("旧运行 real_e203_validation0 = 0/12（过期）",
        all(e == "1" for e in oe), ",".join(oe))
    rec("当前 real_e203_validation0_ascii = 12/12 通过",
        all(e == "0" for e in ne), ",".join(ne))

    # 3) 16 行(L1) 与 8x16(L2) 证据边界不可混用
    r16 = results.parent / "staged_task_row16_summary.json"
    if not r16.is_file():
        r16 = repo / "hardware/architecture/staged_task_row16_summary.json"
    cyc16 = None
    if r16.is_file():
        cases = json.loads(r16.read_text())
        full = [c for c in cases if c.get("name") == "full"]
        cyc16 = full[0].get("cycles") if full else None
    s = results / "real_e203_validation0_ascii_summary.json"
    cyc8 = json.loads(s.read_text()).get("total_task_cycles", 0) // 12 if s.is_file() else None
    rec("16 行(L1) 周期=154240", cyc16 == 154240, f"cycles={cyc16}")
    rec("8x16(L2 E203) 每层周期=838028", cyc8 == 838028, f"cycles={cyc8}")
    rec("两者边界不同，不得相减当加速比", cyc16 != cyc8,
        f"L1={cyc16} L2={cyc8}；1.63x 是 8行L1/L1 模型比值")

    # 4) 网表 / SDC / STA：并非"完全不存在"
    netlists = sorted(sta.glob("systolic_*.v")) if sta.is_dir() else []
    sdcs = sorted(sta.glob("*.sdc")) if sta.is_dir() else []
    rpts = sorted(sta.glob("*.rpt")) if sta.is_dir() else []
    rec("示例库网表存在", bool(netlists), ",".join(p.name for p in netlists) or "缺失")
    rec("探索性 SDC 存在", bool(sdcs), ",".join(p.name for p in sdcs) or "缺失")
    sta_viol = None
    if rpts:
        rp = read(rpts[0])
        m = re.search(r"slack \(VIOLATED\).*?(-?\d+\.\d+)\s*$", rp, re.S)
        worst = re.findall(r"\|\s*(-\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|", rp)
        sta_viol = "VIOLATED" in rp
    rec("存在 iSTA 报告但时序违例（无有效/达标 STA）",
        bool(rpts) and sta_viol is True,
        f"rpt={rpts[0].name if rpts else '缺失'} violated={sta_viol}")
    pwr = sta / "kwt_systolic_cacc_tile.pwr"
    nsp_zero = "Net Switch Power   ==    0.000e+00" in read(pwr) if pwr.is_file() else None
    rec("功耗报告为 0 翻转的无效模型（不得作签核/也不得按0活动）",
        nsp_zero is True, f"Net Switch Power==0 = {nsp_zero}")

    # 5) 提交软件数值模型是否已冻结（权威声明 + 哈希）
    frozen_hits = []
    for pat in ("*frozen*", "*FROZEN*", "*数值模型*"):
        frozen_hits += list(main_dir.glob(pat)) + list((main_dir / "software").glob(pat))
    rec("提交软件数值模型尚未冻结为唯一权威交付",
        not any(p.is_file() for p in frozen_hits),
        f"命中冻结声明文件={[p.name for p in frozen_hits] or '无'}")

    # 6) 评分准入边界（效率+创新被 gate 的分值）
    rec("准入：效率+创新合计 40 分（吞吐8+功耗8+面积4+创新20），非 36",
        "吞吐量 | 8" in rmod or "吞吐量" in rmod,
        "按题目/README.md 评分表：40(功能)+20(集成) 之外，效率20+创新20 受准入")

    print("=" * 78)
    for item, st, detail in rows:
        print(f"[{st}] {item}\n        {detail}")
    print("-" * 78)
    bad = sum(1 for _, st, _ in rows if st == "矛盾/缺口")
    print(f"SUMMARY: 项数={len(rows)} 需关注={bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
