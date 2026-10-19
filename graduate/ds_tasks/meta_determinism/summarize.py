#!/usr/bin/env python3
"""Turn the raw acceptance re-run results into a compact summary.json."""
import argparse
import datetime
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", default="")
    ap.add_argument("--exit-codes", nargs="*", type=int, default=[])
    ap.add_argument("--artifact", nargs=3, action="append", default=[],
                    metavar=("PATH", "DECLARED", "ACTUAL"))
    args = ap.parse_args()

    artifacts = [{"path": p, "declared_sha256": d, "actual_sha256": a,
                  "ok": d == a} for p, d, a in args.artifact]
    summary = {
        "head": args.head,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "exit_codes": args.exit_codes,
        "commands_ok": bool(args.exit_codes) and all(c == 0 for c in args.exit_codes),
        "artifacts": artifacts,
        "artifacts_ok": bool(artifacts) and all(a["ok"] for a in artifacts),
        "all_ok": bool(args.exit_codes) and all(c == 0 for c in args.exit_codes)
        and bool(artifacts) and all(a["ok"] for a in artifacts),
    }
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
