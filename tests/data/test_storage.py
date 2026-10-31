"""DuckDB 迁移、事务与路由资源生命周期的离线回归。"""
import sys
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import market_research.config as C
import market_research.data.marketdata as D
import market_research.data.storage as storage


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

    def test_fund_stamp_ignores_stocks_but_tracks_indices(self):
        storage.write("fund_nav_example", self.frame, {})
        initial = D.data_stamp(exclude_stocks=True)
        global_initial = D.data_stamp()
        for name in ("market_v1_stock_sh600519_qfq_2020", "sync_history_stock_sh600519_qfq"):
            storage.write(name, self.frame, {})
        self.assertEqual(initial, D.data_stamp(exclude_stocks=True))
        self.assertNotEqual(global_initial, D.data_stamp())
        storage.write("index_universe_example", self.frame, {})
        self.assertNotEqual(initial, D.data_stamp(exclude_stocks=True))

    def test_run_summary_uses_database_instead_of_csv_backups(self):
        from market_research.research.runinfo import data_summary
        storage.write("fund_nav_example", self.frame, {})
        storage.write("index_universe_example", self.frame, {})
        self.frame.to_csv(C.DATA_DIR / "fund_nav_old_backup.csv")
        summary = data_summary()
        self.assertEqual(summary["storage"], "duckdb")
        self.assertEqual(summary["n_datasets"], 2)
        self.assertEqual(summary["n_fund_nav_cached"], 1)
        self.assertEqual(summary["n_index_cache"], 1)
        self.assertEqual(summary["n_price_rows"], 8)

    def test_fund_stamp_isolates_other_funds_and_keeps_shared_dependencies(self):
        names = ["fund_nav_110022_2016-01-01_total", "sync_history_fund_110022_total"]
        for name in names:
            storage.write(name, self.frame, {})
        stamp = lambda: D.data_stamp(exclude_stocks=True, fund_code="110022")
        initial = stamp()
        for name in ["fund_nav_000001_2016-01-01_total", "sync_history_fund_000001_total"]:
            storage.write(name, self.frame, {})
        self.assertEqual(initial, stamp())
        for name in names + ["index_universe_2016-01-01", "pe_panel", "unknown_dependency"]:
            previous = stamp()
            storage.write(name, self.frame * 2, {})
            self.assertNotEqual(previous, stamp(), name)

    def test_transaction_rollback(self):
        storage.write("x", self.frame, {"v": 1})
        invalid = self.frame.astype(object)
        invalid.iloc[0, 0] = "not a price"
        with self.assertRaises(Exception):
            storage.write("x", invalid, {"v": 2})
        pd.testing.assert_frame_equal(storage.read("x"), self.frame, check_freq=False)
        self.assertEqual(storage.metadata("x"), {"v": 1})

    def test_metadata_basis_changes_invalidate_results_but_info_does_not(self):
        storage.write("x", self.frame, {"v": 1})
        initial = storage.stamp()
        storage.write_metadata("x", {"v": 2})
        changed = storage.stamp()
        self.assertNotEqual(initial, changed)
        D._cache_info_write("x", {"fetched_at": "later"})
        self.assertEqual(changed, storage.stamp())
        self.assertEqual(storage.metadata("x"), {"v": 2, "info": {"fetched_at": "later"}})

    def test_descriptive_updates_do_not_replace_current_metadata(self):
        storage.write("x", self.frame, {"v": 2, "info": {"source": "original"}})
        with patch.object(storage, "metadata", side_effect=AssertionError("unlocked read")):
            D._cache_info_write("x", {"n_missing": 0})
        self.assertEqual(storage.metadata("x"), {"v": 2, "info": {"source": "original", "n_missing": 0}})


if __name__ == "__main__":
    unittest.main()
