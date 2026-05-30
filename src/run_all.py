import argparse
import subprocess
from pathlib import Path

from src.config import DEFAULT_END, DEFAULT_START, PROCESSED_DIR, PRED_DIR, MODEL_NAME


def run(cmd: list[str]):
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline: preprocess -> train -> backtest")
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=DEFAULT_END)
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--model", type=str, default=MODEL_NAME, choices=["lstm", "gru", "tcn", "transformer"])
    parser.add_argument("--compare", action="store_true", help="Run all models for comparison")
    args = parser.parse_args()

    feature_path = PROCESSED_DIR / "features.parquet"

    run(
        [
            "python",
            "-m",
            "src.data_preprocess",
            "--start",
            args.start,
            "--end",
            args.end,
            "--out",
            str(feature_path),
        ]
        + (["--max-stocks", str(args.max_stocks)] if args.max_stocks else [])
    )

    models = ["lstm", "gru", "tcn", "transformer"] if args.compare else [args.model]
    for model_name in models:
        pred_path = PRED_DIR / f"val_predictions_{model_name}.csv"
        backtest_path = PRED_DIR.parent / "backtest" / f"backtest_{model_name}.csv"

        run(
            [
                "python",
                "-m",
                "src.train",
                "--data",
                str(feature_path),
                "--model",
                model_name,
                "--pred-out",
                str(pred_path),
            ]
        )

        run(["python", "-m", "src.backtest", "--pred", str(pred_path), "--out", str(backtest_path)])


if __name__ == "__main__":
    main()
