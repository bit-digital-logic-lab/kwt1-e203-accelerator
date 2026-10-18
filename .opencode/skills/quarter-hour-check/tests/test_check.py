"""Tests for the 15-minute status hook (scan semantics, notifications, finalizer).

Run from the repository root:

    python3 -m unittest discover -s .opencode/skills/quarter-hour-check/tests -v

All state is redirected to temporary directories, so the project's real
``.ai-checks`` state is never touched.
"""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

SKILL = Path(__file__).resolve().parents[1]
CHECK_PY = SKILL / "scripts" / "check.py"
FINALIZE_HOOK = SKILL / "scripts" / "finalize_hook.sh"
REPO_ROOT = CHECK_PY.resolve().parents[4]
ASCII_FINALIZER = (REPO_ROOT / "hardware/rtl/control/tb/"
                   "finalize_real_e203_validation0_ascii.sh")


def load_check_module():
    spec = importlib.util.spec_from_file_location("quarter_hour_check", CHECK_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = load_check_module()


def run_main(argv):
    """Run check.main and return (returncode, stdout)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = check.main(argv)
    return code, buffer.getvalue()


class ScanTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name)
        self.result = self.state / "result.txt"
        self.result.write_text("ok\n")

    def update(self, *extra):
        return run_main(["update", "--state-dir", str(self.state), *extra])

    def scan(self, *extra):
        return run_main(["scan", "--state-dir", str(self.state), *extra])

    def latest(self):
        return json.loads((self.state / "latest.json").read_text())

    def entry(self, entry_id):
        for item in self.latest()["entries"]:
            if item["id"] == entry_id:
                return item
        raise AssertionError(f"no entry {entry_id!r} in {self.latest()}")

    def checkin_path(self, entry_id):
        return self.state / "checkins" / f"{entry_id}.json"

    # ---- normal completion -------------------------------------------------
    def test_normal_complete_with_result_is_clean(self):
        self.update("--id", "job", "--state", "running", "--result", str(self.result))
        self.update("--id", "job", "--state", "complete", "--result", str(self.result))
        code, _ = self.scan()
        self.assertEqual(code, 0)
        entry = self.entry("job")
        self.assertFalse(entry["needs_attention"])
        self.assertEqual(entry["reason"], "")
        self.assertTrue(entry["result_exists"])
        self.assertFalse((self.state / "ATTENTION.md").exists())
        self.assertEqual(self.latest()["attention_count"], 0)

    # ---- service-only task must still alarm (Claude's rule is wrong) -------
    def test_service_only_without_pid_still_flags_service_stopped(self):
        with mock.patch.object(check, "service_active", return_value=False):
            self.update("--id", "svc", "--state", "running", "--service", "demo.service")
            self.scan()
        entry = self.entry("svc")
        self.assertIsNone(entry["pid_alive"])          # no pid was ever recorded
        self.assertEqual(entry["service_active"], False)
        self.assertTrue(entry["needs_attention"])
        self.assertEqual(entry["reason"], "service_stopped")
        self.assertTrue((self.state / "ATTENTION.md").is_file())

    def test_service_only_still_running_is_clean(self):
        with mock.patch.object(check, "service_active", return_value=True):
            self.update("--id", "svc", "--state", "running", "--service", "demo.service")
            self.scan()
        self.assertFalse(self.entry("svc")["needs_attention"])

    # ---- pid-only task -----------------------------------------------------
    def test_dead_pid_flags_process_stopped(self):
        proc = subprocess.Popen(["true"])
        proc.wait()
        dead_pid = proc.pid
        if check.pid_alive(dead_pid):
            self.skipTest("pid reused before assertion")
        self.update("--id", "proc", "--state", "running", "--pid", str(dead_pid))
        self.scan()
        entry = self.entry("proc")
        self.assertEqual(entry["pid_alive"], False)
        self.assertEqual(entry["reason"], "process_stopped")

    def test_live_pid_is_clean(self):
        self.update("--id", "proc", "--state", "running", "--pid", str(os.getpid()))
        self.scan()
        self.assertFalse(self.entry("proc")["needs_attention"])

    # ---- idle interactive session -----------------------------------------
    def test_idle_session_is_never_disconnected(self):
        with mock.patch.object(check, "service_active", return_value=False), \
                mock.patch.object(check, "pid_alive", return_value=False):
            self.update("--id", "sess", "--state", "idle",
                        "--service", "demo.service", "--pid", "424242")
            self.scan()
        entry = self.entry("sess")
        self.assertFalse(entry["needs_attention"])
        self.assertEqual(entry["reason"], "")

    # ---- result missing ----------------------------------------------------
    def test_complete_with_missing_result_flags_result_missing(self):
        missing = self.state / "not-created.txt"
        self.update("--id", "job", "--state", "complete", "--result", str(missing))
        self.scan()
        entry = self.entry("job")
        self.assertTrue(entry["needs_attention"])
        self.assertEqual(entry["reason"], "result_missing")

    def test_failed_state_flags_task_failed(self):
        self.update("--id", "job", "--state", "failed", "--note", "boom")
        self.scan()
        self.assertEqual(self.entry("job")["reason"], "task_failed")

    def test_stale_running_task_without_backing_flags(self):
        self.update("--id", "job", "--state", "running")
        path = self.checkin_path("job")
        record = json.loads(path.read_text())
        record["updated_at"] = (check.now() - timedelta(seconds=1900)).isoformat()
        path.write_text(json.dumps(record))
        self.scan()
        self.assertEqual(self.entry("job")["reason"], "stale_over_30m")

    # ---- notification dedup ------------------------------------------------
    def install_fake_notify(self):
        bindir = self.state / "bin"
        bindir.mkdir()
        log = self.state / "notify.log"
        fake = bindir / "notify-send"
        fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{log}"\n')
        fake.chmod(0o755)
        original = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{bindir}:{original}"
        self.addCleanup(os.environ.__setitem__, "PATH", original)
        return log

    def test_notification_fires_once_per_state_change(self):
        log = self.install_fake_notify()
        with mock.patch.object(check, "service_active", return_value=False):
            self.update("--id", "svc", "--state", "running", "--service", "demo.service")
            self.scan("--notify")
            self.scan("--notify")   # identical attention -> no repeat popup
            self.scan("--notify")
        self.assertEqual(log.read_text().count("svc:service_stopped"), 1)
        self.assertTrue((self.state / "ATTENTION.md").is_file())
        notify_state = json.loads((self.state / "notification_state.json").read_text())
        self.assertEqual(notify_state["notified"], {"svc": "service_stopped"})

    def test_recovered_then_repeated_failure_notifies_again(self):
        log = self.install_fake_notify()
        self.update("--id", "job", "--state", "complete", "--result", str(self.result))
        self.result.unlink()
        self.scan("--notify")
        self.assertTrue((self.state / "ATTENTION.md").is_file())
        self.result.write_text("ok\n")
        self.scan("--notify")
        self.assertFalse((self.state / "ATTENTION.md").exists())
        self.result.unlink()
        self.scan("--notify")
        self.assertEqual(log.read_text().count("job:result_missing"), 2)

    def test_plain_scan_does_not_consume_notification_state(self):
        log = self.install_fake_notify()
        with mock.patch.object(check, "service_active", return_value=False):
            self.update("--id", "svc", "--state", "running", "--service", "demo.service")
            self.scan()                     # no --notify: nothing recorded, no popup
            self.assertFalse((self.state / "notification_state.json").exists())
            self.scan("--notify")           # first notify run still fires
            self.scan("--notify")
        self.assertEqual(log.read_text().count("svc:service_stopped"), 1)

    def test_missing_notify_send_does_not_change_exit_code(self):
        missing = self.state / "gone.txt"
        self.update("--id", "job", "--state", "complete", "--result", str(missing))
        original = os.environ.get("PATH", "")
        os.environ["PATH"] = str(self.state)     # no notify-send here
        self.addCleanup(os.environ.__setitem__, "PATH", original)
        code, _ = self.scan("--notify")
        self.assertEqual(code, 0)
        self.assertTrue((self.state / "ATTENTION.md").is_file())


class FinalizeHookTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name)
        self.result = self.state / "report.md"
        self.result.write_text("done\n")

    def run_hook(self, body, check_py=str(CHECK_PY), code=0):
        script = (f"set -uo pipefail\nsource {FINALIZE_HOOK!s}\n"
                  f"check_finalize_init demo {check_py!s}\n{body}\nexit {code}\n")
        env = dict(os.environ, QUARTER_HOUR_STATE_DIR=str(self.state))
        return subprocess.run(["bash", "-c", script], env=env,
                              capture_output=True, text=True)

    def checkin(self):
        return json.loads((self.state / "checkins" / "demo.json").read_text())

    def test_abnormal_exit_registers_failed_and_preserves_code(self):
        result = self.run_hook("", code=7)
        self.assertEqual(result.returncode, 7)
        record = self.checkin()
        self.assertEqual(record["state"], "failed")
        self.assertIn("异常", record["note"])

    def test_success_registers_complete_with_result(self):
        result = self.run_hook(f'check_finalize_state complete "{self.result}" "12 层完成"')
        self.assertEqual(result.returncode, 0)
        record = self.checkin()
        self.assertEqual(record["state"], "complete")
        self.assertEqual(record["result"], str(self.result))

    def test_signal_registers_failed_and_preserves_signal_code(self):
        result = self.run_hook("kill -TERM $$")
        self.assertEqual(result.returncode, 143)
        self.assertEqual(self.checkin()["state"], "failed")

    def test_registration_failure_still_preserves_exit_code(self):
        result = self.run_hook("", check_py="/nonexistent/check.py", code=5)
        self.assertEqual(result.returncode, 5)
        self.assertFalse((self.state / "checkins" / "demo.json").exists())

    def test_terminal_state_is_not_overwritten_by_scan(self):
        self.run_hook(f'check_finalize_state complete "{self.result}" "ok"')
        code, _ = run_main(["scan", "--state-dir", str(self.state)])
        self.assertEqual(code, 0)
        self.assertEqual(self.checkin()["state"], "complete")


class FinalizerWiringTest(unittest.TestCase):
    def test_ascii_finalizer_is_syntactically_valid(self):
        result = subprocess.run(["bash", "-n", str(ASCII_FINALIZER)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ascii_finalizer_wires_the_exit_trap(self):
        text = ASCII_FINALIZER.read_text()
        self.assertIn("finalize_hook.sh", text)
        self.assertIn("check_finalize_init", text)
        self.assertIn("check_finalize_state complete", text)
        self.assertIn("set -uo pipefail", text)


if __name__ == "__main__":
    unittest.main()
