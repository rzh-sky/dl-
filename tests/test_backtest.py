import math
import unittest

import pandas as pd

from src.backtest import compute_backtest_metrics


class BacktestMetricTests(unittest.TestCase):
    def test_compute_backtest_metrics(self):
        preds = pd.DataFrame(
            {
                "trade_date": ["20240101", "20240101", "20240102", "20240102"],
                "ts_code": ["AAA", "BBB", "AAA", "BBB"],
                "score": [0.9, 0.1, 0.8, 0.2],
                "label": [0.3, 0.1, 0.2, 0.0],
            }
        )
        backtest_result = pd.DataFrame(
            {
                "trade_date": ["20240101", "20240102"],
                "positions": [2, 2],
                "daily_return": [0.10, 0.00],
                "equity": [1.10, 1.10],
            }
        )

        metrics = compute_backtest_metrics(preds, backtest_result)

        self.assertTrue(math.isclose(metrics["ic_mean"], 1.0, rel_tol=1e-6))
        self.assertTrue(math.isclose(metrics["ic_std"], 0.0, abs_tol=1e-12))
        self.assertTrue(math.isnan(metrics["icir"]))
        self.assertTrue(math.isclose(metrics["total_return"], 0.10, rel_tol=1e-6))
        self.assertTrue(metrics["sharpe"] > 0)
        self.assertTrue(math.isclose(metrics["max_drawdown"], 0.0, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
