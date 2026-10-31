"""DuckDB 迁移、事务与路由资源生命周期的离线回归。"""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import market_research.config as C
import market_research.data.marketdata as D
from market_research.tasks.jobs import JobStore, JobRunner
import market_research.services.analysis as web
from market_research.api.app import app
from fastapi.testclient import TestClient


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        class TestRunner(JobRunner):
            def submit(self, code, models, operation, arguments, key=None):
                return super().submit(code, models, lambda jid:web.execute(operation,arguments,jid), key=key)
        self.runner = TestRunner(JobStore(), max_jobs=1)
        self.patch = patch.object(web, "RUNNER", self.runner)
        self.patch.start()
        self.jobs = patch.object(web, "JOBS", self.runner.store)
        self.jobs.start()

    def tearDown(self):
        self.jobs.stop()
        self.patch.stop()

    def test_cache_identity(self):
        args = ("110022", ["ridge"], {})
        self.assertNotEqual(web.result_cache_path(*args, None), web.result_cache_path(*args, 0))

    def test_changed_data_does_not_cache_old_result_under_new_stamp(self):
        with patch.object(web, "fund_cache_path", side_effect=[Path("old.pkl"), Path("new.pkl")]), patch.object(web, "analyze", return_value={"meta": {}}), patch.object(web, "_atomic_pickle_dump") as save:
            self.assertEqual(web.execute("analyze", {}, "test"), {"meta": {}})
            save.assert_not_called()

    def test_cached_requests_do_not_consume_slots(self):
        from unittest.mock import Mock
        import io, pickle
        path = Mock()
        path.exists.return_value = True
        path.open.side_effect = lambda *a: io.BytesIO(pickle.dumps({"meta": {}}))
        with patch.object(web, "result_cache_path", return_value=path), patch.object(D, "data_stamp", return_value="test"):
            for _ in range(3):
                self.assertEqual(self.client.post("/api/analyze", json={"code": "110022"}).status_code, 200)
        self.assertTrue(self.runner._slots.acquire(False))
        self.runner._slots.release()

    def test_jobs_merge_only_identical_parameters(self):
        with patch("market_research.tasks.jobs.Thread"), patch.object(D, "data_stamp", return_value="test"):
            body = {"code": "110022", "models": ["ridge"], "params": {"ridge": {"alpha_scale": 1}}}
            first = self.client.post("/api/analyze", json=body)
            same = self.client.post("/api/analyze", json=body)
            self.assertEqual(first.json()["job_id"], same.json()["job_id"])
            body["params"]["ridge"]["alpha_scale"] = 2
            self.assertEqual(self.client.post("/api/analyze", json=body).status_code, 429)
            self.assertEqual(self.client.post("/api/refit", json={"code": "110022", "model": "ridge"}).status_code, 429)
            self.assertEqual(self.client.post("/api/backtest", json={"code": "110022"}).status_code, 429)

    def test_all_calculation_routes_execute_background_work(self):
        with tempfile.TemporaryDirectory(dir=C.ROOT / "output") as directory:
            with patch.object(web, "result_cache_path", return_value=Path(directory) / "result.pkl"), patch.object(D, "data_stamp", return_value="test"):
                for route, function, body in (
                    ("analyze", "analyze", {"code": "110022", "models": ["ridge"]}),
                    ("refit", "refit_single", {"code": "110022", "model": "ridge"}),
                    ("backtest", "backtest_fund", {"code": "110022"}),
                ):
                    with self.subTest(route=route), patch.object(web, function, return_value={"meta": {}, "case": route}) as work:
                        response = self.client.post("/api/" + route, json=body)
                        self.assertEqual(response.status_code, 200)
                        jid = response.json()["job_id"]
                        for thread in threading.enumerate():
                            if thread.name == "job-" + jid:
                                thread.join(timeout=5)
                        result = self.client.get("/api/result/" + jid)
                        self.assertEqual(result.status_code, 200)
                        self.assertEqual(result.json()["result"]["case"], route)
                        work.assert_called_once()
                        self.assertTrue(self.runner._slots.acquire(False))
                        self.runner._slots.release()

    def test_non_object_body(self):
        for route in ("analyze", "refit", "backtest"):
            self.assertEqual(self.client.post("/api/" + route, json=[1]).status_code, 400)


if __name__ == "__main__":
    unittest.main()
