"""DuckDB 迁移、事务与路由资源生命周期的离线回归。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
import pandas as pd
from src import config as C, data as D, storage
from src.jobs import JobStore, JobRunner
from webapp import app as web


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=C.ROOT / "output")
        self.patch = patch.object(C, "DATA_DIR", Path(self.tmp.name))
        self.patch.start()
        self.frame = pd.DataFrame({"a": [1., 2.], "b": [3., float("nan")]}, index=pd.date_range("2020-01-01", periods=2, name="date"))

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_legacy_migration_refresh_and_version(self):
        self.frame.to_csv(C.DATA_DIR / "legacy.csv")
        (C.DATA_DIR / "legacy.meta.json").write_text(json.dumps({"v": 1}), encoding="utf-8")
        with patch.object(D, "_atomic_csv_write", side_effect=AssertionError("CSV write")):
            actual = D._cached("legacy", lambda: self.fail("should migrate"), {"v": 1})
            pd.testing.assert_frame_equal(actual, storage.read("legacy"), check_freq=False)
            initial = D.data_stamp()
            D._cached("legacy", lambda: self.fail("cache miss"), {"v": 1})
            D._cache_info_write("legacy", {"n_missing": 0})
            self.assertEqual(initial, D.data_stamp())
            updated = self.frame * 2
            D._cached("legacy", lambda: updated, {"v": 2})
            self.assertNotEqual(initial, D.data_stamp())
            with self.assertRaisesRegex(RuntimeError, "fetch"):
                D._cached("legacy", lambda: (_ for _ in ()).throw(RuntimeError("fetch")), {"v": 2}, refresh=True)
            pd.testing.assert_frame_equal(storage.read("legacy"), updated, check_freq=False)
            self.assertTrue((C.DATA_DIR / "legacy.csv").exists())

    def test_cross_process_writers(self):
        import subprocess
        code = "from pathlib import Path; import sys,pandas as pd; from src import storage,config as C; C.DATA_DIR=Path(sys.argv[1]); storage.write(sys.argv[2],pd.DataFrame({'x':[1.]}, index=pd.date_range('2020-01-01',periods=1)),{'v':1})"
        children = [subprocess.Popen([sys.executable, "-c", code, self.tmp.name, key], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for key in ("one", "two")]
        for child in children:
            _, error = child.communicate(timeout=30)
            self.assertEqual(child.returncode, 0, error.decode(errors="replace"))
        self.assertIsNotNone(storage.read("one"))
        self.assertIsNotNone(storage.read("two"))

    def test_run_summary_uses_database_instead_of_csv_backups(self):
        from src.runinfo import data_summary
        storage.write("fund_nav_example", self.frame, {})
        storage.write("index_universe_example", self.frame, {})
        self.frame.to_csv(C.DATA_DIR / "fund_nav_old_backup.csv")
        summary = data_summary()
        self.assertEqual(summary["storage"], "duckdb")
        self.assertEqual(summary["n_datasets"], 2)
        self.assertEqual(summary["n_fund_nav_cached"], 1)
        self.assertEqual(summary["n_index_cache"], 1)
        self.assertEqual(summary["n_price_rows"], 8)

    def test_transaction_rollback(self):
        storage.write("x", self.frame, {"v": 1})
        invalid = self.frame.astype(object)
        invalid.iloc[0, 0] = "not a price"
        with self.assertRaises(Exception):
            storage.write("x", invalid, {"v": 2})
        pd.testing.assert_frame_equal(storage.read("x"), self.frame, check_freq=False)
        self.assertEqual(storage.metadata("x"), {"v": 1})


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.client = web.app.test_client()
        self.runner = JobRunner(JobStore(), max_jobs=1)
        self.patch = patch.object(web, "RUNNER", self.runner)
        self.patch.start()
        self.jobs = patch.object(web, "JOBS", self.runner.store)
        self.jobs.start()

    def tearDown(self):
        self.jobs.stop()
        self.patch.stop()

    def test_cache_identity(self):
        args = ("110022", ["ridge"], {})
        self.assertNotEqual(web._result_cache_path(*args, None), web._result_cache_path(*args, 0))

    def test_cached_requests_do_not_consume_slots(self):
        from unittest.mock import Mock
        import io, pickle
        path = Mock()
        path.exists.return_value = True
        path.open.side_effect = lambda *a: io.BytesIO(pickle.dumps({"meta": {}}))
        with patch.object(web, "_result_cache_path", return_value=path), patch.object(D, "data_stamp", return_value="test"):
            for _ in range(3):
                self.assertEqual(self.client.post("/api/analyze", json={"code": "110022"}).status_code, 200)
        self.assertTrue(self.runner._slots.acquire(False))
        self.runner._slots.release()

    def test_jobs_merge_only_identical_parameters(self):
        with patch("src.jobs.Thread.start"), patch.object(D, "data_stamp", return_value="test"):
            body = {"code": "110022", "models": ["ridge"], "params": {"ridge": {"alpha_scale": 1}}}
            first = self.client.post("/api/analyze", json=body)
            same = self.client.post("/api/analyze", json=body)
            self.assertEqual(first.json["job_id"], same.json["job_id"])
            body["params"]["ridge"]["alpha_scale"] = 2
            self.assertEqual(self.client.post("/api/analyze", json=body).status_code, 429)
            self.assertEqual(self.client.post("/api/refit", json={"code": "110022", "model": "ridge"}).status_code, 429)
            self.assertEqual(self.client.post("/api/backtest", json={"code": "110022"}).status_code, 429)

    def test_all_calculation_routes_execute_background_work(self):
        with tempfile.TemporaryDirectory(dir=C.ROOT / "output") as directory:
            with patch.object(web, "_result_cache_path", return_value=Path(directory) / "result.pkl"), patch.object(D, "data_stamp", return_value="test"):
                for route, function, body in (
                    ("analyze", "analyze", {"code": "110022", "models": ["ridge"]}),
                    ("refit", "refit_single", {"code": "110022", "model": "ridge"}),
                    ("backtest", "backtest_fund", {"code": "110022"}),
                ):
                    with self.subTest(route=route), patch.object(web, function, return_value={"meta": {}, "case": route}) as work:
                        response = self.client.post("/api/" + route, json=body)
                        self.assertEqual(response.status_code, 200)
                        jid = response.json["job_id"]
                        for thread in threading.enumerate():
                            if thread.name == "job-" + jid:
                                thread.join(timeout=5)
                        result = self.client.get("/api/result/" + jid)
                        self.assertEqual(result.status_code, 200)
                        self.assertEqual(result.json["result"]["case"], route)
                        work.assert_called_once()
                        self.assertTrue(self.runner._slots.acquire(False))
                        self.runner._slots.release()

    def test_non_object_body(self):
        for route in ("analyze", "refit", "backtest"):
            self.assertEqual(self.client.post("/api/" + route, json=[1]).status_code, 400)


if __name__ == "__main__":
    unittest.main()
