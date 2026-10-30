"""股票/指数入口、行情口径及分析指标的离线已知答案测试。"""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import market_research.config as C
from market_research.market.analysis import clean_bars, normalize_code, market_payload, load_bars, analyze_market
import market_research.services.analysis as web
from market_research.api.app import app
from fastapi.testclient import TestClient


def bars(n=140):
    close = np.linspace(100, 140, n)
    return pd.DataFrame({"open": close, "close": close, "high": close + 1, "low": close - 1, "volume": np.arange(n) + 1000.}, index=pd.bdate_range("2024-01-01", periods=n, name="date"))


class MarketTests(unittest.TestCase):
    def test_code_namespaces(self):
        self.assertEqual(normalize_code("stock", "000001"), "sz000001")
        self.assertEqual(normalize_code("stock", "920001"), "bj920001")
        self.assertEqual(normalize_code("index", "sh000001"), "sh000001")
        self.assertEqual(normalize_code("stock", "600519"), "sh600519")
        for kind, code in [("stock", "sh000001"), ("index", "sh600519"), ("stock", "sz399006"), ("index", "bj920001"), ("stock", "garbage"), ("fund", "110022")]:
            with self.assertRaises(ValueError): normalize_code(kind, code)

    def test_ohlc_validation_and_volume_units(self):
        raw = bars(30).reset_index()
        clean = clean_bars(raw, "2024-01-01", "2024-12-31", "lots")
        self.assertEqual(clean.volume.iloc[0], 100000)
        raw.loc[0, "high"] = 1
        with self.assertRaisesRegex(ValueError, "OHLC"): clean_bars(raw, "2024-01-01", "2024-12-31")

    def test_known_return_and_benchmark_alignment(self):
        frame = bars()
        frame.loc[frame.index[20], ["close", "low", "open"]] = [50., 49., 50.]
        reference = pd.Series(np.linspace(100, 110, 120), index=frame.index[20:])
        result = market_payload("stock", "sh600519", frame, reference)
        self.assertAlmostEqual(result["stats"]["total_return"], .4)
        self.assertAlmostEqual(result["stats"]["max_drawdown"], 50 / frame.close.iloc[19] - 1, places=6)
        self.assertEqual(result["comparison"]["dates"][0], frame.index[20].strftime("%Y-%m-%d"))
        self.assertEqual(result["comparison"]["asset"][0], 1.)
        self.assertIsNone(result["series"]["ma"]["60"][58])
        self.assertIsNotNone(result["series"]["ma"]["60"][59])
        self.assertAlmostEqual(result["comparison"]["excess_return"], 140 / 50 - 1.1, places=6)
        self.assertEqual(result["series"]["candles"][20], [50., 50., 49., round(float(frame.high.iloc[20]), 6)])

    def test_partial_benchmark_and_short_sample(self):
        with patch("market_research.market.analysis.load_bars", side_effect=[bars(), RuntimeError("benchmark unavailable")]):
            result = analyze_market("stock", "600519", "2024-01-01", "2025-01-01")
        self.assertFalse(result["comparison"]["ok"])
        self.assertTrue(any("基准指数暂不可用" in text for text in result["warnings"]))
        with patch("market_research.market.analysis.load_bars", return_value=bars(10)):
            with self.assertRaisesRegex(ValueError, "21"): analyze_market("stock", "600519", "2024-01-01", "2025-01-01")

    def test_duckdb_cache_and_adjustment_separation(self):
        import akshare as ak
        with tempfile.TemporaryDirectory(dir=C.ROOT / "output") as directory, patch.object(C, "DATA_DIR", Path(directory)):
            raw = bars().reset_index()
            with patch.object(ak, "stock_zh_a_hist", return_value=raw) as fetch:
                first = load_bars("stock", "sh600519", "2024-01-01", "2025-01-01")
                second = load_bars("stock", "sh600519", "2024-01-01", "2025-01-01")
                pd.testing.assert_frame_equal(first, second, check_freq=False)
                self.assertEqual(fetch.call_count, 1)
                load_bars("stock", "sh600519", "2024-01-01", "2025-01-01", adjust="")
                self.assertEqual(fetch.call_count, 2)
                self.assertTrue((Path(directory) / "market.duckdb").exists())
            with patch.object(ak, "stock_zh_a_hist", side_effect=RuntimeError("offline")), patch.object(ak, "stock_zh_a_hist_tx", side_effect=RuntimeError("offline")):
                with self.assertRaises(ValueError): load_bars("stock", "sh600519", "2024-01-01", "2025-01-01", refresh=True)
            pd.testing.assert_frame_equal(load_bars("stock", "sh600519", "2024-01-01", "2025-01-01"), first, check_freq=False)

    def test_stock_fallback_preserves_normalized_volume(self):
        import akshare as ak
        with tempfile.TemporaryDirectory(dir=C.ROOT / "output") as directory, patch.object(C, "DATA_DIR", Path(directory)):
            raw = bars().reset_index()
            with patch.object(ak, "stock_zh_a_hist", side_effect=RuntimeError("primary unavailable")), patch.object(ak, "stock_zh_a_hist_tx", return_value=raw):
                frame = load_bars("stock", "sh600519", "2024-01-01", "2025-01-01")
            self.assertEqual(frame.volume.iloc[0], raw.volume.iloc[0])
            self.assertEqual(frame.attrs["source"], "AKShare / 腾讯证券")
            self.assertEqual(frame.attrs["name"], "贵州茅台")

    def test_short_market_window_message(self):
        from market_research.validation.window import parse_window
        with self.assertRaisesRegex(ValueError, "30 个自然日") as error:
            parse_window("2024-01-01", "2024-01-10", min_days=30)
        self.assertNotIn("2 年", str(error.exception))

    def test_revised_prices_never_mix_with_old_training_history(self):
        current = bars(140)
        stale = current.copy()
        stale["high"] += 3
        with patch("market_research.market.analysis.load_bars", side_effect=[current, RuntimeError("no benchmark")]), patch("market_research.tasks.sync.cached", return_value=stale), patch("market_research.market.technical.analyze_technical", return_value={"ready":False}) as technical:
            result = analyze_market("stock","600519","2024-01-01","2025-01-01")
        pd.testing.assert_frame_equal(technical.call_args.args[0],current)
        self.assertFalse(result["technical"]["ready"])

    def test_market_api_validation_and_shared_runner(self):
        client = TestClient(app)
        for body in ([1], {"kind": "stock", "code": "sh000001"}, {"kind": "stock", "code": "600519", "adjust": "bad"}, {"kind": "index", "code": "sh000300", "start": "2024/01/01"}):
            self.assertEqual(client.post("/api/market/analyze", json=body).status_code, 400)
        with patch.object(web.RUNNER, "submit", return_value=("market-test", False)) as submit:
            response = client.post("/api/market/analyze", json={"kind": "stock", "code": "600519", "start": "2024-01-01", "end": "2025-01-01"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(submit.call_args.args[:2], ("sh600519", ["stock"]))
        self.assertIn("assets", client.get("/api/catalog").json())


if __name__ == "__main__": unittest.main()
