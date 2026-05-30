import unittest
from unittest.mock import patch

import pandas as pd

import src.dataset as dataset_module


class SequenceDatasetTests(unittest.TestCase):
    def setUp(self):
        rows = []
        dates = ["20240101", "20240102", "20240103", "20240104"]
        for code, base in [("AAA", 10.0), ("BBB", 20.0)]:
            for i, trade_date in enumerate(dates):
                rows.append(
                    {
                        "trade_date": trade_date,
                        "ts_code": code,
                        "label": float(i + (0 if code == "AAA" else 100)),
                        "f1": base + i,
                        "f2": base + i + 0.5,
                    }
                )
        self.df = pd.DataFrame(rows).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    def test_index_alignment_and_meta(self):
        with patch.object(dataset_module, "FEATURE_COLS", ["f1", "f2"]):
            ds = dataset_module.SequenceDataset(self.df, lookback=2, horizon=1)

            self.assertEqual(len(ds), 4)

            x0, y0 = ds[0]
            self.assertEqual(tuple(x0.shape), (2, 2))
            self.assertAlmostEqual(float(y0.item()), 1.0)
            self.assertAlmostEqual(float(x0[0, 0].item()), 10.0)
            self.assertAlmostEqual(float(x0[1, 1].item()), 11.5)

            meta0 = ds.get_meta(0)
            self.assertEqual(meta0["date"], "20240102")
            self.assertEqual(meta0["code"], "AAA")
            self.assertAlmostEqual(meta0["label"], 1.0)

    def test_collate_fn(self):
        with patch.object(dataset_module, "FEATURE_COLS", ["f1", "f2"]):
            ds = dataset_module.SequenceDataset(self.df, lookback=2, horizon=1)
            batch = [ds[0], ds[1]]
            x_batch, y_batch = dataset_module.SequenceDataset.collate_fn(batch)

            self.assertEqual(tuple(x_batch.shape), (2, 2, 2))
            self.assertEqual(tuple(y_batch.shape), (2,))
            self.assertAlmostEqual(float(y_batch[0].item()), 1.0)
            self.assertAlmostEqual(float(y_batch[1].item()), 2.0)


if __name__ == "__main__":
    unittest.main()
