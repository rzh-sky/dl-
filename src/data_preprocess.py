import argparse
import gc
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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


def read_daily_range(start_date: str, end_date: str, max_stocks: Optional[int] = None) -> pd.DataFrame:
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
        "vwap",
        "pct_chg",
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

    # --- 时序特征 (使用对数收益以提高数值稳定性) ---
    # log returns 替代 pct_change
    df["log_ret1"] = gb["close"].transform(lambda x: np.log(x / x.shift(1))).astype('float32')
    df["log_ret3"] = gb["close"].transform(lambda x: np.log(x / x.shift(3))).astype('float32')
    df["log_ret5"] = gb["close"].transform(lambda x: np.log(x / x.shift(5))).astype('float32')
    df["log_ret10"] = gb["close"].transform(lambda x: np.log(x / x.shift(10))).astype('float32')
    df["log_ret20"] = gb["close"].transform(lambda x: np.log(x / x.shift(20))).astype('float32')

    df["vol_log1"] = gb["vol"].transform(lambda x: np.log(x / x.shift(1))).astype('float32')
    df["vol_log5"] = gb["vol"].transform(lambda x: np.log(x / x.shift(5))).astype('float32')

    df["hl_range"] = ((high - low) / close).astype('float32')
    df["oc_ret"] = ((close - open_) / open_).astype('float32')
    df["co_ret"] = ((open_ - close) / close).astype('float32')
    if "vwap" in df.columns:
        df["vwap_ret"] = ((df["vwap"] - close) / close).astype('float32')
    else:
        # 保证列长度与 df 一致并使用 float32 NaN，避免后续 dtype 不一致
        df["vwap_ret"] = np.full(len(df), np.nan, dtype=np.float32)
    # momentum 用过去 5 日对数收益之和，等价于多日对数收益
    df["mom5"] = df["log_ret5"].astype('float32')
    df["mom10"] = gb["close"].transform(lambda x: np.log(x / x.shift(10))).astype('float32')
    df["mom20"] = gb["close"].transform(lambda x: np.log(x / x.shift(20))).astype('float32')

    def roll_mean(col, w):
        return gb[col].rolling(w).mean().reset_index(level=0, drop=True)
    
    def roll_std(col, w):
        return gb[col].rolling(w).std().reset_index(level=0, drop=True)

    # 均线用相对距离表示
    df["ma5"] = (roll_mean("close", 5) / close - 1).astype('float32')
    df["ma10"] = (roll_mean("close", 10) / close - 1).astype('float32')
    df["ma20"] = (roll_mean("close", 20) / close - 1).astype('float32')

    # 实现波动率（使用对数收益的滚动标准差）
    df["std5"] = (roll_std("log_ret1", 5)).astype('float32')
    df["std10"] = (roll_std("log_ret1", 10)).astype('float32')
    df["std20"] = (roll_std("log_ret1", 20)).astype('float32')

    # RSI (基于差分的简化计算)
    # 先一次性计算 group 内差分，避免重复 groupby/rolling 带来的额外开销
    delta = df.groupby('ts_code', sort=False)['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    def calc_rsi(w):
        # 使用 transform + rolling，在多数 pandas 版本上比多次 groupby 更节省开销
        avg_gain = gain.groupby(df['ts_code']).transform(lambda s: s.rolling(w).mean())
        avg_loss = loss.groupby(df['ts_code']).transform(lambda s: s.rolling(w).mean())
        rs = avg_gain / (avg_loss + 1e-9)
        return (100 - (100 / (1 + rs))).astype('float32')

    df["rsi6"] = calc_rsi(6)
    df["rsi12"] = calc_rsi(12)
    df["rsi24"] = calc_rsi(24)

    # 量价相关性：过去 10 日价格对数收益 与 成交量对数收益 的滚动相关系数
    # 按股票分别计算，避免 rolling.corr 的 pairwise 输入限制
    gb = df.groupby("ts_code", sort=False)
    df["pv_corr10"] = (
        gb.apply(lambda g: g["log_ret1"].rolling(10).corr(g["vol_log1"]))
        .reset_index(level=0, drop=True)
        .astype('float32')
    )

    # 资金流强度：净流入相对成交额 / 成交量
    amount_safe = df["amount"].replace(0, np.nan)
    vol_safe = df["vol"].replace(0, np.nan)
    if "net_mf_amount" in df.columns:
        df["net_mf_amount_ratio"] = (df["net_mf_amount"] / amount_safe).astype('float32')
    if "net_mf_vol" in df.columns:
        df["net_mf_vol_ratio"] = (df["net_mf_vol"] / vol_safe).astype('float32')

    # ── 补全缺失的常用技术指标 ──────────────────────────────────
    # vol_chg / amount_chg（对数变化，等同对数收益）
    df["vol_chg"] = df["vol_log1"].astype('float32')
    df["vol_chg5"] = df["vol_log5"].astype('float32')
    df["amount_chg"] = gb["amount"].transform(lambda x: np.log(x / x.shift(1))).astype('float32')
    df["amount_chg5"] = gb["amount"].transform(lambda x: np.log(x / x.shift(5))).astype('float32')

    # EMA（指数移动平均，相对距离）
    ema12 = gb["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = gb["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["ema12"] = ((ema12 / close) - 1).astype('float32')
    df["ema26"] = ((ema26 / close) - 1).astype('float32')

    # MACD 系列（原始差值，cross‑section 标准化会统一尺度）
    macd_line = (ema12 - ema26).astype('float32')
    df["macd"] = macd_line
    macd_signal = macd_line.groupby(df["ts_code"]).transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df["macd_signal"] = macd_signal.astype('float32')
    df["macd_hist"] = (macd_line - macd_signal).astype('float32')
    del ema12, ema26, macd_line, macd_signal

    # 布林带（相对距离）
    boll_mid = roll_mean("close", 20)
    boll_sd = roll_std("close", 20)
    df["boll_up"] = ((boll_mid + 2 * boll_sd) / close - 1).astype('float32')
    df["boll_dn"] = ((boll_mid - 2 * boll_sd) / close - 1).astype('float32')
    df["boll_width"] = ((boll_mid + 2 * boll_sd) - (boll_mid - 2 * boll_sd)).div(close).astype('float32')

    # ATR（Average True Range，相对价格）
    prev_close = gb["close"].transform(lambda x: x.shift(1))
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.groupby(df["ts_code"]).transform(lambda x: x.rolling(14).mean()).div(close).astype('float32')
    del prev_close, tr1, tr2, tr3, tr

    # 量价 Z‑score（股票内滚动 Z‑score of vol_log1 / amount_chg）
    roll_mean5 = lambda x: x.rolling(5).mean()
    roll_std5 = lambda x: x.rolling(5).std().replace(0, np.nan)
    df["vol_zscore5"] = (
        (df["vol_log1"] - gb["vol_log1"].transform(roll_mean5))
        / gb["vol_log1"].transform(roll_std5)
    ).astype('float32')
    df["amount_zscore5"] = (
        (df["amount_chg"] - gb["amount_chg"].transform(roll_mean5))
        / gb["amount_chg"].transform(roll_std5)
    ).astype('float32')

    # 换手率滚动均线
    if "turnover_rate" in df.columns:
        df["turnover_5"] = gb["turnover_rate"].transform(lambda x: x.rolling(5).mean()).astype('float32')
        df["turnover_10"] = gb["turnover_rate"].transform(lambda x: x.rolling(10).mean()).astype('float32')

    # 资金流向比例：各档买入 / (买入+卖出)
    for prefix, buy_col, sell_col in [
        ("lg", "buy_lg_amount", "sell_lg_amount"),
        ("sm", "buy_sm_amount", "sell_sm_amount"),
        ("md", "buy_md_amount", "sell_md_amount"),
        ("elg", "buy_elg_amount", "sell_elg_amount"),
    ]:
        if buy_col in df.columns and sell_col in df.columns:
            total = df[buy_col] + df[sell_col]
            df[f"{prefix}_buy_ratio"] = (df[buy_col] / total.replace(0, np.nan)).astype('float32')

    # ── 别名 ────────────────────────────────────────────────────
    df["ret1"] = df["log_ret1"].astype('float32')
    df["ret3"] = df["log_ret3"].astype('float32')
    df["ret5"] = df["log_ret5"].astype('float32')
    df["ret10"] = df["log_ret10"].astype('float32')
    df["ret20"] = df["log_ret20"].astype('float32')

    # 构造标签 & 次日收益（用于回测 PnL）
    df["label"] = (gb["close"].shift(-horizon) / close - 1).astype('float32')
    df["label"] = df["label"].clip(-0.15, 0.15)
    df["ret_next"] = (gb["close"].shift(-1) / close - 1).astype('float32')
    df["ret_next"] = df["ret_next"].clip(-0.10, 0.10)

    # 统一对所有时序特征做一次股票内前向填充，避免在后面的截面循环里反复 groupby
    time_series_fill_cols = [
        "log_ret1", "log_ret3", "log_ret5", "log_ret10", "log_ret20",
        "vol_log1", "vol_log5", "ma5", "ma10", "ma20", "std5", "std10",
        "std20", "rsi6", "rsi12", "rsi24", "mom5", "mom10", "mom20",
        "hl_range", "oc_ret", "co_ret", "vwap_ret", "pv_corr10",
        "ret1", "ret3", "ret5", "ret10", "ret20",
        "vol_chg", "amount_chg", "vol_chg5", "amount_chg5",
        "ema12", "ema26", "macd", "macd_signal", "macd_hist",
        "boll_up", "boll_dn", "boll_width", "atr14",
        "vol_zscore5", "amount_zscore5",
        "turnover_5", "turnover_10",
        "lg_buy_ratio", "sm_buy_ratio", "md_buy_ratio", "elg_buy_ratio",
    ]
    time_series_fill_cols = [c for c in time_series_fill_cols if c in df.columns]
    if time_series_fill_cols:
        df[time_series_fill_cols] = df.groupby("ts_code", sort=False)[time_series_fill_cols].ffill()

    # 【序列完整性】为避免滚动窗口产生的初始 NaN 干扰截面统计，
    # 在进行截面 transform 之前先删除在关键时序特征上为 NaN 的行。
    seq_cols = [
        "log_ret1", "log_ret3", "log_ret5", "log_ret10", "log_ret20",
        "vol_log1", "vol_log5", "ma5", "ma10", "ma20", "std5", "std10",
        "std20",
        "rsi6", "rsi12", "rsi24", "pv_corr10", "mom5", "mom10", "mom20",
    ]
    present_seq_cols = [c for c in seq_cols if c in df.columns]
    drop_subset = present_seq_cols + ["label"]
    df = df.dropna(subset=drop_subset).reset_index(drop=True)
    
    # 释放无用变量（注意：避免删除未定义的局部名）
    try:
        del close, open_, high, low, vol, amount, gb, delta, gain, loss
    except NameError:
        # 如果某些变量未被创建则忽略
        pass
    gc.collect()

    print(">>> [低内存模式] 开始逐列截面处理 (监控内存不会暴涨)...")
    
    date_gb = df.groupby("trade_date")

    # 【核心省内存动作 2】：每次只拿一列出来做标准化与更优填充策略，做完释放
    # 定义时序型特征集合（使用前面创建的列名）
    time_series_cols = {
        "log_ret1", "log_ret3", "log_ret5", "log_ret10", "log_ret20",
        "vol_log1", "vol_log5", "ma5", "ma10", "ma20", "std5", "std10",
        "std20",
        "rsi6", "rsi12", "rsi24", "mom5", "mom10", "mom20", "hl_range", "oc_ret", "co_ret",
        "vwap_ret", "pv_corr10", "ret1", "ret3", "ret5", "ret10", "ret20",
        "vol_chg", "amount_chg", "vol_chg5", "amount_chg5",
        "ema12", "ema26", "macd", "macd_signal", "macd_hist",
        "boll_up", "boll_dn", "boll_width", "atr14",
        "vol_zscore5", "amount_zscore5",
        "turnover_5", "turnover_10",
        "lg_buy_ratio", "sm_buy_ratio", "md_buy_ratio", "elg_buy_ratio",
    }
    # 定义资金流/收益类保持为 0 的集合
    money_like_cols = {"net_mf_amount", "net_mf_vol", "buy_sm_amount", "sell_sm_amount", "buy_md_amount", "sell_md_amount", "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount"}

    # 汇总所有需要进入截面处理的特征，避免只处理 BASE_FEATURE_COLS
    all_features = list(dict.fromkeys(list(BASE_FEATURE_COLS) + [
        "log_ret1", "log_ret3", "log_ret5", "log_ret10", "log_ret20",
        "vol_log1", "vol_log5", "std20", "pv_corr10",
        "net_mf_amount_ratio", "net_mf_vol_ratio", "rsi24", "mom10", "mom20",
        "vol_chg", "amount_chg", "vol_chg5", "amount_chg5",
        "ema12", "ema26", "macd", "macd_signal", "macd_hist",
        "boll_up", "boll_dn", "boll_width", "atr14",
        "vol_zscore5", "amount_zscore5",
        "turnover_5", "turnover_10",
        "lg_buy_ratio", "sm_buy_ratio", "md_buy_ratio", "elg_buy_ratio",
    ]))

    for col in all_features:
        if col not in df.columns:
            df[col] = np.nan

        # 如果原始数据里没有这个列（比如没读到基本面），先补齐为 NaN（保留 miss 信息）
        df[col] = df[col].astype('float32')

        # 1. 生成 miss 列 (int8 极小) —— 基于原始缺失
        df[f"{col}_miss"] = df[col].isna().astype('int8')

        # 2. 更优填充策略
        if col in time_series_cols:
            # 时序特征优先使用股票内前向填充（处理停牌），剩余缺失再填 0
            df[col] = df.groupby('ts_code')[col].ffill().fillna(0)
        elif col in money_like_cols:
            # 资金流类缺失视为 0
            df[col] = df[col].fillna(0)
        else:
            # 截面特征使用当日全市场中位数填充
            # 为避免 groupby.transform 在大表上分配巨型临时数组，先计算每日中位数再 map 回来（更节省内存）
            date_median = df.groupby('trade_date', sort=False)[col].median()
            df[col] = df[col].fillna(df['trade_date'].map(date_median)).fillna(0)
            del date_median

        # 3. 去极值（winsorization）：使用当日 1%/99% 分位数截断
        lower = date_gb[col].transform(lambda x: x.quantile(0.01))
        upper = date_gb[col].transform(lambda x: x.quantile(0.99))
        df[col] = df[col].clip(lower=lower.fillna(-np.inf), upper=upper.fillna(np.inf))
        del lower, upper

        # 4. Z-score（截面标准化）
        col_mean = date_gb[col].transform('mean')
        col_std = date_gb[col].transform('std').replace(0, 1e-9)
        df[f"{col}_z"] = ((df[col] - col_mean) / col_std).astype('float32')

        # 5. Rank（截面百分位）
        df[f"{col}_rank"] = date_gb[col].rank(pct=True).astype('float32')

        # 算完一个列立刻删掉它的中间变量
        del col_mean, col_std
        gc.collect() # 强行收回内存

    print(">>> 特征工程全部跑通，准备保存！")
    return df


def save_schema_artifacts(df: pd.DataFrame, out_dir: Path) -> None:
    feature_cols = [col for col in FEATURE_COLS if col in df.columns]
    dtype_map = {col: str(df[col].dtype) for col in feature_cols}

    (out_dir / "feature_columns.json").write_text(
        json.dumps(feature_cols, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "feature_dtypes.json").write_text(
        json.dumps(dtype_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_debug_sample(df: pd.DataFrame, out_path: Path, n_rows: int = 1000) -> None:
    sample_df = df.tail(n_rows).copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_parquet(out_path, index=False)


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

    BACKTEST_AUX = ["ret_next", "pct_chg", "vol", "close"]
    required_cols = ["trade_date", "ts_code", "label"] + BACKTEST_AUX + [col for col in FEATURE_COLS if col in df.columns]
    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required feature columns: {missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}")

    # 用 pyarrow columnar writer 逐列写入 parquet，完全避免碎片化 DataFrame 的 consolidate
    out_cols = [c for c in required_cols if c in df.columns]
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan
            df[c] = df[c].astype('float32')
        elif df[c].dtype == 'float64':
            df[c] = df[c].astype('float32')

    save_schema_artifacts(df, PROCESSED_DIR)
    save_debug_sample(df, PROCESSED_DIR / "features_debug_sample.parquet", n_rows=1000)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 逐列构建 pyarrow Table，避免 pandas consolidate
    pa_arrays = []
    pa_fields = []
    for c in out_cols:
        col = df[c]
        if col.dtype == 'object':
            pa_arrays.append(pa.array(col, type=pa.string()))
            pa_fields.append(pa.field(c, pa.string()))
        else:
            pa_arrays.append(pa.array(col.to_numpy(), type=pa.float32()))
            pa_fields.append(pa.field(c, pa.float32()))
    table = pa.Table.from_arrays(pa_arrays, schema=pa.schema(pa_fields))
    pq.write_table(table, out_path)
    print(f"Saved features ({table.num_rows} rows x {table.num_columns} cols) to {out_path}")


if __name__ == "__main__":
    main()
