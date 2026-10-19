#!/usr/bin/env python3
"""p027_precision -- render the results table (markdown + tsv) from results.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent


def main():
    res = json.loads((TASK / "results.json").read_text())
    rows = []
    head = ("| scheme | A bits (x/a/r1/ln1/g/r2) | W attn/w1/w2 | worst layer | "
            "pooled y NRMSE | holdout worst | weight KiB/12L | MAC bit-products |")
    sep = "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"
    lines = [f"# p027_precision summary (state={res['state']}, selected={res['selected']})",
             "", head, sep]
    tsv = ["name\tstage_bits\tweights\tworst_layer\tworst_y_pct\tholdout_pct\t"
           "weight_kib\tmac_bit_products\ttarget_met"]
    for r in res["candidates"]:
        if r.get("state") != "EVALUATED":
            lines.append(f"| {r['name']} | - | - | - | {r.get('state')} | - | - | - |")
            continue
        sb = r["cfg"]["stage_bits"]
        a = "/".join(str(sb[k]) for k in ("x", "a", "r1", "ln1", "g", "r2"))
        w = f"{r['cfg']['w_attn']}/{r['cfg']['w1']}/{r['cfg']['w2']}"
        h = r.get("holdout_train", {})
        hp = f"{h['worst_y_nrmse_pct']:.4f}%" if h else "-"
        wb = r["cost"]["twelve_layers"]["weight_kib"]
        mp = r["cost"]["twelve_layers"]["mac_bit_products_total"] / 1e9
        lines.append(f"| {r['name']} | {a} | {w} | {r['worst_layer']} | "
                     f"{r['worst_y_nrmse_pct']:.4f}% | {hp} | {wb:.1f} | {mp:.1f}e9 |")
        tsv.append(f"{r['name']}\t{a}\t{w}\t{r['worst_layer']}\t"
                   f"{r['worst_y_nrmse_pct']:.6f}\t"
                   f"{(h.get('worst_y_nrmse_pct') if h else '')}\t{wb:.1f}\t{mp:.3f}\t"
                   f"{r['target_0p27pct']}")
    text = "\n".join(lines) + "\n"
    (TASK / "logs").mkdir(exist_ok=True)
    (TASK / "logs" / "summary.md").write_text(text)
    (TASK / "logs" / "summary.tsv").write_text("\n".join(tsv) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
