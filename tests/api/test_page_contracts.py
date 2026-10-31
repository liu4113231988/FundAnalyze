"""页面参数与计算口径的离线边界回归。"""
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
from pydantic import ValidationError

from market_research.api.schemas import AnalysisRequest, BacktestRequest, RefitRequest
from market_research.modeling.catalog import _default_params
from market_research.modeling.estimators import fit_window
from market_research.funds.analysis import backtest_fund


class PageContractTests(unittest.TestCase):
    def test_every_exposed_default_is_accepted_including_mlp_layers(self):
        defaults = _default_params()
        result = AnalysisRequest(code='110022', params=defaults)
        self.assertEqual(result.params['mlp']['hidden'], (32, 16))
        for model, params in defaults.items():
            RefitRequest(code='110022', model=model, params=params)

    def test_invalid_model_parameters_are_rejected_before_job_submission(self):
        cases = [('ridge', {'max_assets': 0}), ('rf', {'n_estimators': 1.5}),
                 ('gbr', {'learning_rate': -1}), ('enet_slsqp', {'l1_ratio': 1.1}),
                 ('pcr', {'min_weight': 1}), ('ridge', {'typo': 1}),
                 ('mlp', {'hidden': []}), ('mlp', {'hidden': [32, 0]}),
                 ('mlp', {'hidden': [True]}), ('mlp', {'hidden': [2.5]}),
                 ('ridge', {'alpha_scale': float('inf')})]
        for model, params in cases:
            with self.subTest(model=model, params=params):
                with self.assertRaises(ValidationError):
                    AnalysisRequest(code='110022', params={model: params})

    def test_backtest_crop_can_be_shorter_than_fitting_window(self):
        request = BacktestRequest(code='110022', start='2025-06-01', end='2025-12-31',
                                  window_start='2021-01-01', window_end='2025-12-31')
        self.assertEqual(request.start, '2025-06-01')
        for changes in ({'start': '2020-01-01'}, {'end': '2026-01-01'},
                        {'window_start': '2025-01-01'}, {'start': '2025-12-31'}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                BacktestRequest(**(request.model_dump() | changes))

    def test_backtest_uses_current_fitting_window_parameters_and_precision(self):
        panel = pd.DataFrame({'__FUND__': [1., 1.1]}, index=pd.date_range('2021-01-01', periods=2))
        zoo = Mock()
        zoo.run.side_effect = RuntimeError('stop before expensive calculation')
        with patch('market_research.funds.analysis.D.load_panel', return_value=panel) as load, \
             patch('market_research.funds.analysis.ModelZoo', return_value=zoo):
            with self.assertRaisesRegex(RuntimeError, 'stop before'):
                backtest_fund('110022', model_key='ridge', start_date='2025-06-01', end_date='2025-12-31',
                              window_start='2021-01-01', window_end='2025-12-31',
                              params={'alpha_scale': 2}, max_rebal=0)
        load.assert_called_once_with('110022', start='2021-01-01', end='2025-12-31', verbose=False)
        self.assertEqual(zoo.run.call_args.kwargs['params'], {'ridge': {'alpha_scale': 2}})
        self.assertEqual(zoo.run.call_args.kwargs['max_rebal'], 0)

    def test_krr_alpha_changes_predictions(self):
        rng = np.random.default_rng(41)
        x = rng.normal(0, .01, (160, 3))
        y = x[:, 0] * .6 + rng.normal(0, .002, 160)
        a = fit_window('krr', x, y, ['a', 'b', 'c'], params={'alpha': .001})
        b = fit_window('krr', x, y, ['a', 'b', 'c'], params={'alpha': 100})
        self.assertGreater(np.max(np.abs(a.pred - b.pred)), 1e-5)


if __name__ == '__main__':
    unittest.main()
