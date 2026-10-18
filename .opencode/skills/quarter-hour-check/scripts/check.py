#!/usr/bin/env python3
"""Project check-ins and a deterministic quarter-hour scan."""

import argparse
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
STATE = ROOT / ".ai-checks"
CHECKINS = STATE / "checkins"
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


def now():
    return datetime.now(timezone.utc)


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                 prefix=f".{path.name}.", delete=False) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        tmp = Path(handle.name)
    os.replace(tmp, path)


def update(args):
    if not ID_PATTERN.fullmatch(args.id):
        raise SystemExit("invalid check-in id")
    path = CHECKINS / f"{args.id}.json"
    prior = json.loads(path.read_text()) if path.exists() else {}
    record = {"id": args.id, "state": args.state, "updated_at": now().isoformat(),
              "source": "manual", "note": args.note or prior.get("note", ""),
              "result": args.result or prior.get("result", ""),
              "service": args.service or prior.get("service", ""),
              "pid": args.pid if args.pid is not None else prior.get("pid")}
    atomic_json(path, record)
    print(path)


def service_active(name):
    if not name:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", name):
        return False
    result = subprocess.run(["systemctl", "--user", "is-active", "--quiet", name],
                            check=False, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    return result.returncode == 0


def pid_alive(pid):
    if pid is None:
        return None
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True


def scan(_args):
    observed = now()
    entries = []
    for path in sorted(CHECKINS.glob("*.json")):
        try:
            item = json.loads(path.read_text())
            stamped = item["updated_at"].replace("Z", "+00:00")
            age = max(0, int((observed - datetime.fromisoformat(stamped)).total_seconds()))
            state = item["state"]
            service = service_active(item.get("service"))
            pid = pid_alive(item.get("pid"))
            result = item.get("result", "")
            result_exists = bool(result and Path(result).is_file() and Path(result).stat().st_size)
            running = state in ("running", "busy", "retry")
            reason = ""
            if state in ("failed", "error"):
                reason = "task_failed"
            elif running and service is False:
                reason = "service_stopped"
            elif running and pid is False:
                reason = "process_stopped"
            elif running and age > 1800 and service is not True and pid is not True:
                reason = "stale_over_30m"
            elif state == "complete" and result and not result_exists:
                reason = "result_missing"
            entries.append({"id": item.get("id", path.stem), "state": state,
                            "age_seconds": age, "service_active": service,
                            "pid_alive": pid, "result_exists": result_exists,
                            "needs_attention": bool(reason), "reason": reason,
                            "note": item.get("note", "")})
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            entries.append({"id": path.stem, "needs_attention": True,
                            "reason": f"invalid_checkin: {type(exc).__name__}"})
    report = {"checked_at": observed.isoformat(), "interval_seconds": 900,
              "entries": entries, "attention_count": sum(e["needs_attention"] for e in entries)}
    atomic_json(STATE / "latest.json", report)
    print(f"checked {len(entries)} entries; attention={report['attention_count']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    update_parser = sub.add_parser("update")
    update_parser.add_argument("--id", required=True)
    update_parser.add_argument("--state", choices=("running", "busy", "retry", "idle", "complete", "failed"), required=True)
    update_parser.add_argument("--result")
    update_parser.add_argument("--service")
    update_parser.add_argument("--pid", type=int)
    update_parser.add_argument("--note")
    sub.add_parser("scan")
    args = parser.parse_args()
    (update if args.command == "update" else scan)(args)


if __name__ == "__main__":
    main()
