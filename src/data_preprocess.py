import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import (
    DAILY_DIR,
    BASE_FEATURE_COLS,
    FEATURE_COLS,
    OUTPUT_DIR,
    PROCESSED_DIR,
    STOCK_ST_DIR,
    METRIC_DIR,
    MONEYFLOW_DIR,
    DEFAULT_START,
    DEFAULT_END,
)


def load_stock_pool() -> set[str]:
    basic_path = DAILY_DIR.parent / "basic.csv"
    basic = pd.read_csv(basic_path, dtype={"ts_code": str})
    pool = basic[basic["market"] != "北交所"]["ts_code"].astype(str)
    return set(pool.tolist())


def list_daily_files(start_date: str, end_date: str) -> list[Path]:
    files = []
    for p in DAILY_DIR.glob("*.csv"):
        date = p.stem
        if start_date <= date <= end_date:
            files.append(p)
    return sorted(files, key=lambda x: x.stem)


class STCache:
    def __init__(self, stock_st_dir: Path):
        self.stock_st_dir = stock_st_dir
        self.cache: dict[str, set[str]] = {}

    def get_st_set(self, trade_date: str) -> set[str]:
        if trade_date in self.cache:
            return self.cache[trade_date]
        st_path = self.stock_st_dir / f"{trade_date}.csv"
        if not st_path.exists():
            self.cache[trade_date] = set()
            return self.cache[trade_date]
        st_df = pd.read_csv(st_path, dtype={"ts_code": str})
        st_set = set(st_df["ts_code"].astype(str).tolist())
        self.cache[trade_date] = st_set
        return st_set


def read_daily_range(start_date: str, end_date: str, max_stocks: int | None = None) -> pd.DataFrame:
    pool = load_stock_pool()
    if max_stocks is not None:
        pool = set(sorted(pool)[:max_stocks])

    st_cache = STCache(STOCK_ST_DIR)
    files = list_daily_files(start_date, end_date)

    frames = []
    usecols = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
    ]

    metric_cols = [
        "ts_code",
        "trade_date",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ratio",
        "dv_ttm",
        "total_share",
        "float_share",
        "free_share",
        "total_mv",
        "circ_mv",
    ]

    money_cols = [
        "ts_code",
        "trade_date",
        "buy_sm_amount",
        "sell_sm_amount",
        "buy_md_amount",
        "sell_md_amount",
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
        "net_mf_vol",
        "net_mf_amount",
    ]

    for f in tqdm(files, desc="Reading daily files"):
        trade_date = f.stem
        df = pd.read_csv(f, usecols=usecols, dtype={"ts_code": str, "trade_date": str})
        df = df[df["ts_code"].isin(pool)]
        st_set = st_cache.get_st_set(trade_date)
        if st_set:
            df = df[~df["ts_code"].isin(st_set)]

        metric_path = METRIC_DIR / f"{trade_date}.csv"
        if metric_path.exists():
            metric_df = pd.read_csv(metric_path, usecols=metric_cols, dtype={"ts_code": str, "trade_date": str})
            df = df.merge(metric_df, on=["ts_code", "trade_date"], how="left")

        money_path = MONEYFLOW_DIR / f"{trade_date}.csv"
        if money_path.exists():
            money_df = pd.read_csv(money_path, usecols=money_cols, dtype={"ts_code": str, "trade_date": str})
            df = df.merge(money_df, on=["ts_code", "trade_date"], how="left")
        frames.append(df)

    if not frames:
        raise RuntimeError("No daily files found for the given date range.")

    data = pd.concat(frames, ignore_index=True)
    data["trade_date"] = data["trade_date"].astype(str)
    
    # 【新增内存优化】将所有 float64 强制降级为 float32
    float_cols = data.select_dtypes(include=['float64']).columns
    data[float_cols] = data[float_cols].astype('float32')
    
    return data


import gc
import numpy as np

def add_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    print(">>> [低内存模式] 开始时序特征提取...")
    
    # 提前降维原始数据
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')

    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    gb = df.groupby("ts_code")

    # 基础引用
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    vol = df["vol"]
    amount = df["amount"]

    # --- 时序特征 (全部强制 float32 省内存) ---
    df["ret1"] = gb["close"].pct_change().astype('float32')
    df["ret3"] = gb["close"].pct_change(3).astype('float32')
    df["ret5"] = gb["close"].pct_change(5).astype('float32')
    df["ret10"] = gb["close"].pct_change(10).astype('float32')
    df["ret20"] = gb["close"].pct_change(20).astype('float32')

    df["vol_chg"] = gb["vol"].pct_change().astype('float32')
    df["amount_chg"] = gb["amount"].pct_change().astype('float32')
    df["vol_chg5"] = gb["vol"].pct_change(5).astype('float32')
    df["amount_chg5"] = gb["amount"].pct_change(5).astype('float32')

    df["hl_range"] = ((high - low) / close).astype('float32')
    df["oc_ret"] = ((close - open_) / open_).astype('float32')
    df["mom5"] = (close / gb["close"].shift(5) - 1).astype('float32')

    def roll_mean(col, w):
        return gb[col].rolling(w).mean().reset_index(level=0, drop=True)
    
    def roll_std(col, w):
        return gb[col].rolling(w).std().reset_index(level=0, drop=True)

    df["ma5"] = (roll_mean("close", 5) / close - 1).astype('float32')
    df["ma10"] = (roll_mean("close", 10) / close - 1).astype('float32')
    df["ma20"] = (roll_mean("close", 20) / close - 1).astype('float32')

    df["std5"] = (roll_std("close", 5) / close).astype('float32')
    df["std10"] = (roll_std("close", 10) / close).astype('float32')

    # RSI (简化计算减小开销)
    delta = gb["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    gain_gb = gain.groupby(df['ts_code'])
    loss_gb = loss.groupby(df['ts_code'])
    
    def calc_rsi(w):
        avg_gain = gain_gb.rolling(w).mean().reset_index(level=0, drop=True)
        avg_loss = loss_gb.rolling(w).mean().reset_index(level=0, drop=True)
        rs = avg_gain / (avg_loss + 1e-9)
        return (100 - (100 / (1 + rs))).astype('float32')

    df["rsi6"] = calc_rsi(6)
    df["rsi12"] = calc_rsi(12)

    # 构造标签 
    df["label"] = (gb["close"].shift(-horizon) / close - 1).astype('float32')

    # 【核心省内存动作 1】：算完特征立刻删掉包含 NaN 的行，数据量瞬间减少
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    
    # 释放无用变量
    del close, open_, high, low, vol, amount, gb, delta, gain, loss, gain_gb, loss_gb
    gc.collect()

    print(">>> [低内存模式] 开始逐列截面处理 (监控内存不会暴涨)...")
    
    date_gb = df.groupby("trade_date")

    # 【核心省内存动作 2】：每次只拿一列出来做标准化，做完释放
    for col in BASE_FEATURE_COLS:
        # 如果原始数据里没有这个列（比如没读到基本面），先补齐
        if col not in df.columns:
            df[col] = 0.0
            
        df[col] = df[col].astype('float32')
        
        # 1. 生成 miss 列 (int8 极小)
        df[f"{col}_miss"] = df[col].isna().astype('int8')
        
        # 2. 填充 0
        df[col] = df[col].fillna(0)
        
        # 3. Z-score
        col_mean = date_gb[col].transform('mean')
        col_std = date_gb[col].transform('std').replace(0, 1e-9)
        df[f"{col}_z"] = ((df[col] - col_mean) / col_std).astype('float32')
        
        # 4. Rank
        df[f"{col}_rank"] = date_gb[col].rank(pct=True).astype('float32')
        
        # 算完一个列立刻删掉它的中间变量
        del col_mean, col_std
        gc.collect() # 强行收回内存

    print(">>> 特征工程全部跑通，准备保存！")
    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess A-share daily data.")
    parser.add_argument("--start", type=str, default=DEFAULT_START, help="Start date YYYYMMDD")
    parser.add_argument("--end", type=str, default=DEFAULT_END, help="End date YYYYMMDD")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon (days)")
    parser.add_argument("--max-stocks", type=int, default=None, help="Limit number of stocks for quick run")
    parser.add_argument("--out", type=str, default=str(PROCESSED_DIR / "features.parquet"))

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = read_daily_range(args.start, args.end, max_stocks=args.max_stocks)
    df = add_features(df, horizon=args.horizon)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved features to {out_path}")


if __name__ == "__main__":
    main()
