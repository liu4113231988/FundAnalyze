import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from tests.market.test_technical import sample
from market_research.market.context import context_features
from market_research.market.technical import indicators, features_for, direction_forecast, analyze_technical, compare_forecasts
from market_research.market.analysis import analyze_market
from market_research.serialization import _jsafe


class ContextTests(unittest.TestCase):
    def test_known_relative_strength_volume_and_missing_reference(self):
        bars = sample(150)
        bars['amount'] = 100.; bars['turnover'] = 2.
        out = context_features(bars, indicators(bars), bars.close)
        np.testing.assert_allclose(out.relative20.dropna(), 0, atol=1e-12)
        self.assertEqual(out.volume_ratio.iloc[-1], 1)
        self.assertEqual(out.amount_ratio.iloc[-1], 1)
        reference = bars.close.copy(); reference.iloc[-1] = np.nan
        missing = context_features(bars, indicators(bars), reference)
        self.assertTrue(pd.isna(missing.relative20.iloc[-1]))
        self.assertTrue(pd.isna(missing.benchmark_gap60.iloc[-1]))

    def test_expanded_features_and_predictions_are_prefix_stable(self):
        bars = sample(450); prefix = bars.iloc[:420]
        full = context_features(bars, indicators(bars), bars.close * .8)
        prior = context_features(prefix, indicators(prefix), prefix.close * .8)
        pd.testing.assert_frame_equal(full.loc[prefix.index], prior)
        a = direction_forecast(bars, features_for(bars, indicators(bars)).join(full), 5, bars.index[0])
        b = direction_forecast(prefix, features_for(prefix, indicators(prefix)).join(prior), 5, bars.index[0])
        np.testing.assert_allclose(a['series'][:420], b['series'], equal_nan=True)

    def test_comparison_uses_common_mature_dates(self):
        bars = sample(80); bars['close'] = np.arange(1., 81.)
        base = {'series': [.6] * 80}; candidate = {'series': [.9] * 80}
        candidate['series'][:10] = [np.nan]*10
        out = compare_forecasts(bars, base, candidate, 5, bars.index[0])
        self.assertEqual(out['n'], 13)
        self.assertEqual(out['start'], bars.index[10].date().isoformat())
        self.assertEqual(out['end'], bars.index[70].date().isoformat())
        self.assertAlmostEqual(out['base']['brier'], .16)
        self.assertAlmostEqual(out['candidate']['brier'], .01)
        self.assertEqual(sum(b['n'] for b in out['candidate']['calibration']), 13)
        self.assertFalse(out['ok'])

    def test_zero_volume_flat_prices_and_short_data_remain_json_safe(self):
        bars = sample(80); bars[['open','high','low','close']] = 100.; bars['volume'] = 0.
        out = analyze_technical(bars, reference=None)
        self.assertEqual(out['context']['environment'], '数据不足')
        self.assertTrue(all(not f['comparison']['ok'] for f in out['forecasts']))
        json.dumps(_jsafe(out), allow_nan=False)

    def test_index_production_path_has_technical_and_context(self):
        bars = sample(100)
        with patch('market_research.market.analysis.load_bars', return_value=bars), \
             patch('market_research.tasks.sync.cached', return_value=None):
            out = analyze_market('index', 'sh000300', '2023-01-01', '2024-01-01')
        self.assertIn('context', out['technical'])
        self.assertEqual(out['technical']['context']['benchmark'], 'sh000300')
        self.assertTrue(any('ETF' in n for n in out['technical']['notices']))
        json.dumps(out, allow_nan=False)


if __name__ == '__main__': unittest.main()
