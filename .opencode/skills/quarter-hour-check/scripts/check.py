#!/usr/bin/env python3
"""Project check-ins and a deterministic quarter-hour scan.

State semantics (documented in the skill and enforced here):

* ``running`` / ``busy`` / ``retry`` describe a task that should have live
  backing.  When a ``service`` and/or ``pid`` is recorded the scan verifies it.
  A service-only check-in has no pid (``None``); an inactive service must still
  be reported as ``service_stopped``.  Absence of a pid never suppresses the
  service check.
* ``idle``  interactive session waiting for input.  No background process is
  expected, so it is never reported as disconnected.
* ``complete``  terminal success.  If a result path is recorded it must exist
  and be non-empty, otherwise ``result_missing``.
* ``failed`` / ``error``  terminal failure, reported as ``task_failed``.

The scan only observes and records state.  It never wakes a model, retries a
job, or edits results; conclusions must come from real logs and exit codes.
Notifications are optional and are emitted only when an entry's attention
reason changes, so a persistent problem is not re-popped every 15 minutes.
A missing ``notify-send`` never changes the scan exit code.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_STATE = ROOT / ".ai-checks"
STATE_ENV = "QUARTER_HOUR_STATE_DIR"
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
LIVE_STATES = ("running", "busy", "retry")


def state_root(args):
    """Resolve the state directory: flag, then env, then the project default."""
    if getattr(args, "state_dir", None):
        return Path(args.state_dir)
    override = os.environ.get(STATE_ENV)
    if override:
        return Path(override)
    return DEFAULT_STATE


def now():
    return datetime.now(timezone.utc)


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        handle.write(text)
        tmp = Path(handle.name)
    os.replace(tmp, path)


def atomic_json(path, data):
    atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def load_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def update(args, state):
    if not ID_PATTERN.fullmatch(args.id):
        raise SystemExit("invalid check-in id")
    path = state / "checkins" / f"{args.id}.json"
    prior = load_json(path, {})
    record = {"id": args.id, "state": args.state, "updated_at": now().isoformat(),
              "source": "manual", "note": args.note or prior.get("note", ""),
              "result": args.result or prior.get("result", ""),
              "service": args.service or prior.get("service", ""),
              "pid": args.pid if args.pid is not None else prior.get("pid")}
    atomic_json(path, record)
    print(path)


def service_active(name):
    """True if the unit is active, False if clearly inactive, None if unknown.

    Unknown is returned when the unit name is unusable or systemctl is missing;
    it must not be reported as a stopped service.
    """
    if not name:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", name):
        return False
    try:
        result = subprocess.run(["systemctl", "--user", "is-active", "--quiet", name],
                                check=False, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
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


def evaluate(item, observed):
    """Return one scan entry, isolating the attention rules for testing."""
    stamped = item["updated_at"].replace("Z", "+00:00")
    age = max(0, int((observed - datetime.fromisoformat(stamped)).total_seconds()))
    state = item["state"]
    service = service_active(item.get("service"))
    pid = pid_alive(item.get("pid"))
    result = item.get("result", "")
    result_exists = bool(result and Path(result).is_file() and Path(result).stat().st_size)
    running = state in LIVE_STATES
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
    return {"id": item.get("id", ""), "state": state, "age_seconds": age,
            "service_active": service, "pid_alive": pid, "result_exists": result_exists,
            "needs_attention": bool(reason), "reason": reason,
            "note": item.get("note", "")}


def send_notification(enabled, entries):
    """Best-effort desktop notification. Never raises, never affects the exit code."""
    if not enabled or not entries:
        return False
    summary = ", ".join(f"{e['id']}:{e['reason']}" for e in entries)
    message = f"quarter-hour-check 需要关注: {summary}"
    try:
        result = subprocess.run(["notify-send", "--urgency=critical",
                                 "quarter-hour-check", message],
                                check=False, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def write_attention_marker(state, observed, attention):
    """Conspicuous on-disk marker so a human notices without a desktop session."""
    marker = state / "ATTENTION.md"
    if not attention:
        try:
            marker.unlink()
        except FileNotFoundError:
            pass
        return
    lines = [f"# quarter-hour-check attention ({observed.isoformat()})", ""]
    for entry in attention:
        note = f" — {entry['note']}" if entry.get("note") else ""
        lines.append(f"- {entry['id']}: {entry['reason']}{note}")
    atomic_write(marker, "\n".join(lines) + "\n")


def scan(args, state):
    observed = now()
    entries = []
    for path in sorted((state / "checkins").glob("*.json")):
        try:
            item = json.loads(path.read_text())
            item.setdefault("id", path.stem)
            entries.append(evaluate(item, observed))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            entries.append({"id": path.stem, "needs_attention": True,
                            "reason": f"invalid_checkin: {type(exc).__name__}",
                            "state": "invalid", "note": ""})
    report = {"checked_at": observed.isoformat(), "interval_seconds": 900,
              "entries": entries, "attention_count": sum(e["needs_attention"] for e in entries)}
    atomic_json(state / "latest.json", report)

    attention = [e for e in entries if e.get("needs_attention")]
    write_attention_marker(state, observed, attention)

    changed = []
    if args.notify:
        notify_path = state / "notification_state.json"
        previous = load_json(notify_path, {}).get("notified", {})
        current = {e["id"]: e["reason"] for e in attention}
        changed = [e for e in attention if previous.get(e["id"]) != e["reason"]]
        notified = send_notification(True, changed)
        atomic_json(notify_path, {"updated_at": observed.isoformat(),
                                  "last_change_at": observed.isoformat() if changed else
                                  load_json(notify_path, {}).get("last_change_at"),
                                  "last_notify_ok": notified,
                                  "notified": current})

    print(f"checked {len(entries)} entries; attention={report['attention_count']}"
          + (f"; new={len(changed)}" if changed else ""))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_state_dir(target):
        target.add_argument("--state-dir",
                            help="override the .ai-checks directory (also QUARTER_HOUR_STATE_DIR)")

    update_parser = sub.add_parser("update")
    update_parser.add_argument("--id", required=True)
    update_parser.add_argument("--state",
                              choices=("running", "busy", "retry", "idle", "complete", "failed"),
                              required=True)
    update_parser.add_argument("--result")
    update_parser.add_argument("--service")
    update_parser.add_argument("--pid", type=int)
    update_parser.add_argument("--note")
    add_state_dir(update_parser)

    scan_parser = sub.add_parser("scan")
    scan_parser.add_argument("--notify", action="store_true",
                             help="send a desktop notification when an attention reason changes")
    add_state_dir(scan_parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    state = state_root(args)
    (update if args.command == "update" else scan)(args, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
