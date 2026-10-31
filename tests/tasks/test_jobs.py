"""共享任务生命周期和 CLI/Web 边界的行为测试。"""
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from market_research.tasks.jobs import JobRunner, JobStore, JobsBusy
from market_research.validation.window import parse_window, resolve_window


class JobTests(unittest.TestCase):
    def setUp(self):
        self.store = JobStore()
        self.runner = JobRunner(self.store, max_jobs=1)

    def test_worker_success_and_failure_release_capacity(self):
        def immediate_thread(target, **kwargs):
            return SimpleNamespace(start=target)
        with patch("market_research.tasks.jobs.Thread", side_effect=immediate_thread), patch("market_research.tasks.jobs.traceback.print_exc"):
            jid, merged = self.runner.submit("110022", ["ridge"], lambda jid: {"answer": 42})
            self.assertFalse(merged)
            self.assertEqual(self.store.get(jid)["result"], {"answer": 42})
            def fail(jid):
                raise ValueError("fit failed")
            failed, _ = self.runner.submit("110022", ["ridge"], fail)
            self.assertEqual(self.store.get(failed)["status"], "error")
            self.assertIn("fit failed", self.store.get(failed)["error"])
            final, _ = self.runner.submit("110022", ["ridge"], lambda jid: {})
            self.assertEqual(self.store.get(final)["status"], "done")

    def test_start_and_creation_failures_release_capacity(self):
        with patch("market_research.tasks.jobs.Thread.start", side_effect=RuntimeError("thread unavailable")):
            with self.assertRaises(RuntimeError):
                self.runner.submit("110022", ["ridge"], lambda jid: {})
        with patch.object(self.store, "create", side_effect=RuntimeError("store unavailable")):
            with self.assertRaisesRegex(RuntimeError, "store unavailable"):
                self.runner.submit("110022", ["ridge"], lambda jid: {})
        self.assertTrue(self.runner._slots.acquire(False))
        self.runner._slots.release()

    def test_concurrent_duplicates_create_only_one_worker(self):
        from concurrent.futures import ThreadPoolExecutor
        with patch("market_research.tasks.jobs.Thread") as thread:
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: self.runner.submit("110022", ["ridge"], lambda jid: {}, key="same"), range(8)))
            self.assertEqual(len({jid for jid, _ in results}), 1)
            self.assertEqual(sum(not merged for _, merged in results), 1)
            self.assertEqual(thread.return_value.start.call_count, 1)
            with self.assertRaises(JobsBusy):
                self.runner.submit("110022", ["ridge"], lambda jid: {}, key="different")

    def test_timeout_cannot_be_overwritten_by_late_success(self):
        with patch("market_research.tasks.jobs.time.time", return_value=100):
            jid = self.store.create("110022", ["ridge"])
        with patch("market_research.tasks.jobs.time.time", return_value=100 + self.store.JOB_TIMEOUT_SEC + 1):
            self.assertEqual(self.store.get(jid)["status"], "error")
            self.store.update(jid, stage="late progress")
            self.store.finish(jid, {"late": True})
            job = self.store.get(jid)
            self.assertEqual(job["stage"], "超时")
            self.assertIsNone(job["result"])

    def test_terminal_state_and_public_snapshot(self):
        jid = self.store.create("110022", ["ridge"])
        self.store.finish(jid, {"ok": True})
        self.store.fail(jid, "late failure")
        snapshot = self.store.get(jid)
        snapshot["status"] = "error"
        self.assertEqual(self.store.get(jid)["status"], "done")
        self.assertNotIn("result", self.store.public(jid))
        cancelled = self.store.create("110022", ["ridge"])
        self.assertTrue(self.store.cancel(cancelled))
        self.store.finish(cancelled, {})
        self.assertEqual(self.store.get(cancelled)["status"], "error")


class WindowTests(unittest.TestCase):
    def test_cli_and_http_share_strict_dates(self):
        for start in ("2021/01/01", "20210101", "2021.01.01", "NaT", "2021-1-1"):
            with self.subTest(start=start):
                with self.assertRaises(ValueError):
                    parse_window(start, "2026-01-01")
                with self.assertRaises(ValueError):
                    resolve_window(start=start, end="2026-01-01")
        self.assertEqual(parse_window("2021-01-01", "2026-01-01"), resolve_window(start="2021-01-01", end="2026-01-01"))
        self.assertEqual(parse_window(None, None), (None, None))


if __name__ == "__main__":
    unittest.main()
