import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from market_research.market.technical import indicators, wilder, features_for, direction_forecast, analyze_technical
from market_research.funds.payloads import _jsafe


def sample(n=430):
    rng = np.random.default_rng(117)
    close = 100 * np.exp(np.cumsum(rng.normal(.0002, .012, n)))
    return pd.DataFrame({"open":close, "high":close+1, "low":close-1, "close":close, "volume":1000.}, index=pd.bdate_range('2023-01-01',periods=n))


class TechnicalTests(unittest.TestCase):
    def test_wilder_arithmetic_seed(self):
        result = wilder(pd.Series(np.arange(1.,17.)))
        self.assertTrue(result.iloc[:13].isna().all())
        self.assertAlmostEqual(result.iloc[13],7.5)
        self.assertAlmostEqual(result.iloc[14],(7.5*13+15)/14)

    def test_rsi_flat_and_one_way_prices(self):
        for close, expected in [(np.full(80,100.),50), (np.arange(100.,180.),100), (np.arange(180.,100.,-1),0)]:
            frame=sample(80)
            frame['close']=close; frame['high']=close+1; frame['low']=close-1
            result=indicators(frame)
            self.assertTrue(result.rsi.iloc[:14].isna().all())
            self.assertAlmostEqual(result.rsi.iloc[-1],expected)
            self.assertTrue(result.score.iloc[:64].isna().all())
            if expected==50:
                self.assertEqual(result.boll_upper.iloc[-1],100.)
                self.assertEqual(result.boll_lower.iloc[-1],100.)
                self.assertEqual(result.score.iloc[-1],0.)
                self.assertFalse(result.exit.iloc[-1])

    def test_atr_includes_price_gaps_and_prior_extrema(self):
        frame=sample(40)
        frame[['open','close']]=100.;frame['high']=101.;frame['low']=99.
        frame.iloc[14,frame.columns.get_indexer(['open','close','high','low'])]=[103.,103.,104.,102.]
        result=indicators(frame)
        self.assertAlmostEqual(result.atr.iloc[14],(13*2+4)/14)
        self.assertEqual(result.resistance.iloc[20],104.)
        self.assertTrue(pd.isna(result.resistance.iloc[19]))
        self.assertEqual(result.volume_ratio.iloc[-1],1.)

    def test_prefix_indicators_and_forecasts_do_not_see_future(self):
        frame=sample(430)
        prefix=frame.iloc[:385]
        full_ind=indicators(frame); prior_ind=indicators(prefix)
        pd.testing.assert_frame_equal(full_ind.loc[prefix.index],prior_ind)
        full=direction_forecast(frame,features_for(frame,full_ind),5,frame.index[0])
        prior=direction_forecast(prefix,features_for(prefix,prior_ind),5,frame.index[0])
        np.testing.assert_allclose(full['series'][:385],prior['series'],equal_nan=True,rtol=1e-10)

    def test_training_target_matures_before_prediction_and_evaluation_spacing(self):
        frame=sample()
        positions={d:i for i,d in enumerate(frame.index)}
        horizon=20
        class Spy:
            def fit(self,x,y):
                self.last=positions[x.index[-1]]
                self.base=float(y.mean())
                return self
            def predict_proba(self,x):
                assert self.last+horizon<=positions[x.index[0]],'future target leaked'
                return np.array([[.3,.7]])
        with patch('market_research.market.technical.make_pipeline',side_effect=lambda *args:Spy()):
            out=direction_forecast(frame,features_for(frame,indicators(frame)),horizon,frame.index[0])
        self.assertTrue(out['ok'])
        self.assertLessEqual(out['evaluation']['n'],len(frame)//horizon)
        self.assertFalse(out['evaluation']['ok'])

    def test_short_history_no_forecast_and_json_finite(self):
        frame=sample(40)
        result=analyze_technical(frame,adjust='')
        self.assertFalse(result['ready'])
        self.assertTrue(all(not f['ok'] for f in result['forecasts']))
        self.assertTrue(any('不复权' in x for x in result['notices']))
        json.dumps(_jsafe(result),allow_nan=False)

    def test_signals_are_first_condition_and_use_past_only(self):
        frame=sample()
        ind=indicators(frame)
        np.testing.assert_array_equal(ind.buy_event,ind.entry & ~ind.entry.shift(fill_value=False))
        np.testing.assert_array_equal(ind.sell_event,ind.exit & ~ind.exit.shift(fill_value=False))
        result=analyze_technical(frame,display_start=frame.index[-100],adjust='hfq')
        self.assertEqual(len(result['series']['dates']),100)
        self.assertLessEqual(len(result['signals']),12)
        self.assertTrue(any('不能直接' in x for x in result['notices']))
        json.dumps(_jsafe(result),allow_nan=False)


if __name__=='__main__':unittest.main()
