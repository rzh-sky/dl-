from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "A-data"
DAILY_DIR = DATA_DIR / "daily"
STOCK_ST_DIR = DATA_DIR / "stock_st"
METRIC_DIR = DATA_DIR / "metric"
MONEYFLOW_DIR = DATA_DIR / "moneyflow"
NEWS_DIR = DATA_DIR / "news"
MARKET_DIR = DATA_DIR / "market"
OUTPUT_DIR = ROOT / "outputs"
PROCESSED_DIR = OUTPUT_DIR / "processed"
MODEL_DIR = OUTPUT_DIR / "models"
PRED_DIR = OUTPUT_DIR / "preds"
BACKTEST_DIR = OUTPUT_DIR / "backtest"

DEFAULT_START = "20190101"
DEFAULT_END = "20251231"

TRAIN_START = "20190101"
TRAIN_END = "20231231"
VAL_START = "20240101"
VAL_END = "20251231"

LOOKBACK = 20
HORIZON = 1

BATCH_SIZE = 512
EPOCHS = 5
LR = 1e-3
LR_MIN = 1e-5
LR_PATIENCE = 3
LR_FACTOR = 0.5
EARLY_STOP_PATIENCE = 3
GRAD_CLIP = 1.0
HIDDEN = 64
NUM_LAYERS = 2
DROPOUT = 0.2
MODEL_NAME = "lstm"
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2

# Backtest strategy params
TARGET_N_HOLD = 15
REBALANCE_INTERVAL = 5
COMMISSION_BUY = 0.0003    # 买入万分之3
COMMISSION_SELL = 0.0013   # 卖出千分之1.3
SCORE_THRESHOLD_STD = 0.0  # 建仓/买入评分阈值标准差倍数
REBALANCE_N_TRADE = 3      # 每次调仓换几只（固定值）
REBALANCE_THRESHOLD = 0.03 # 调仓最低评分差阈值(3%)
POSITION_LIMIT = 0.15      # 单只股票最大仓位15%
VOLUME_MIN = 100_000       # 最低日均成交量(手)，约折合1000万元
LIMIT_THRESHOLD = 9.5      # 涨跌停阈值(%)

NEWS_FEATURE_COLS = ["news_count", "news_polarity", "industry_news_count", "industry_news_polarity"]

MARKET_FEATURE_COLS = [
    "mkt_sh_pct_chg",   # 上证指数涨跌幅
    "mkt_sz_pct_chg",   # 沪深300涨跌幅
    "mkt_cy_pct_chg",   # 创业板指涨跌幅
    "mkt_sh_pct_chg_ma5",  # 上证5日均涨幅
    "mkt_sz_pct_chg_ma5",
    "mkt_cy_pct_chg_ma5",
]

INDUSTRY_FEATURE_COLS = [
    "ind_ret_avg",       # 个股所属行业当日平均收益
    "ind_ret_avg_ma5",   # 行业5日平均收益
]

EXTRA_FEATURE_COLS = MARKET_FEATURE_COLS + INDUSTRY_FEATURE_COLS

BASE_FEATURE_COLS = [
    "ret1",
    "ret3",
    "ret5",
    "ret10",
    "ret20",
    "vol_chg",
    "amount_chg",
    "vol_chg5",
    "amount_chg5",
    "ma5",
    "ma10",
    "ma20",
    "ema12",
    "ema26",
    "std5",
    "std10",
    "std20",
    "hl_range",
    "oc_ret",
    "co_ret",
    "vwap_ret",
    "mom5",
    "mom10",
    "mom20",
    "rsi6",
    "rsi12",
    "rsi24",
    "macd",
    "macd_signal",
    "macd_hist",
    "boll_up",
    "boll_dn",
    "boll_width",
    "atr14",
    "vol_zscore5",
    "amount_zscore5",
    "turnover_5",
    "turnover_10",
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
    "net_mf_amount",
    "net_mf_vol",
    "net_mf_amount_ratio",
    "net_mf_vol_ratio",
    "lg_buy_ratio",
    "sm_buy_ratio",
    "md_buy_ratio",
    "elg_buy_ratio",

]

DERIVED_SUFFIXES = ["_z", "_rank", "_miss"]
FEATURE_COLS = BASE_FEATURE_COLS + [
    f"{col}{suffix}" for col in BASE_FEATURE_COLS for suffix in DERIVED_SUFFIXES
]
# 额外特征：新闻+行业需要衍生列，市场特征不需要（截面无差异）
EXTRA_FEATURE_COLS = NEWS_FEATURE_COLS + INDUSTRY_FEATURE_COLS
EXTRA_FEATURE_DERIVED = [
    f"{col}{suffix}" for col in EXTRA_FEATURE_COLS for suffix in DERIVED_SUFFIXES
]
FEATURE_COLS = list(dict.fromkeys(
    FEATURE_COLS + EXTRA_FEATURE_COLS + EXTRA_FEATURE_DERIVED + MARKET_FEATURE_COLS
))

# 全局调试开关：在调试模式下会执行额外的完整性检查（可能会扫描大数组）
DEBUG = False

# 随机种子（可通过环境变量覆盖）
SEED = 42
