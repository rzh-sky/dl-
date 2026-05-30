import argparse
from pathlib import Path

import pandas as pd

from src.config import BACKTEST_DIR, K_TRADE, N_HOLD, PRED_DIR


def run_backtest(pred_path: Path, n_hold: int, k_trade: int) -> pd.DataFrame:
    preds = pd.read_csv(pred_path, dtype={"trade_date": str, "ts_code": str})
    preds = preds.sort_values("trade_date")

    equity = 1.0
    positions: set[str] = set()
    records = []

    for date, group in preds.groupby("trade_date"):
        group = group.sort_values("score", ascending=False)

        if not positions:
            positions = set(group.head(n_hold)["ts_code"].tolist())
        else:
            current = group[group["ts_code"].isin(positions)]
            sell = set(current.sort_values("score").head(k_trade)["ts_code"].tolist())
            candidates = group[~group["ts_code"].isin(positions)]
            buy = set(candidates.head(k_trade)["ts_code"].tolist())

            positions = (positions - sell) | buy

        pnl_group = group[group["ts_code"].isin(positions)]
        if pnl_group.empty:
            daily_ret = 0.0
        else:
            daily_ret = pnl_group["label"].mean()

        equity *= 1.0 + daily_ret

        records.append(
            {
                "trade_date": date,
                "positions": len(positions),
                "daily_return": daily_ret,
                "equity": equity,
            }
        )

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Run simple backtest with predictions.")
    default_pred = PRED_DIR / "val_predictions.csv"
    parser.add_argument(
        "--pred",
        type=str,
        default=str(default_pred),
        help="Path to predictions csv",
    )
    parser.add_argument("--out", type=str, default=str(BACKTEST_DIR / "backtest.csv"))
    parser.add_argument("--n-hold", type=int, default=N_HOLD)
    parser.add_argument("--k-trade", type=int, default=K_TRADE)
    args = parser.parse_args()

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {pred_path}. Run training first to generate it."
        )

    result = run_backtest(pred_path, args.n_hold, args.k_trade)
    result.to_csv(out_path, index=False)
    print(f"Saved backtest to {out_path}")


if __name__ == "__main__":
    main()
