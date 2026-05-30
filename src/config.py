from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "A-data"
DAILY_DIR = DATA_DIR / "daily"
STOCK_ST_DIR = DATA_DIR / "stock_st"
METRIC_DIR = DATA_DIR / "metric"
MONEYFLOW_DIR = DATA_DIR / "moneyflow"
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
EPOCHS = 50
LR = 1e-3
LR_MIN = 1e-5
LR_PATIENCE = 3
LR_FACTOR = 0.5
EARLY_STOP_PATIENCE = 10
GRAD_CLIP = 1.0
HIDDEN = 64
NUM_LAYERS = 2
DROPOUT = 0.2
MODEL_NAME = "lstm"
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2

# Backtest strategy params
TARGET_N_HOLD = 10
REBALANCE_INTERVAL = 2
COMMISSION_BUY = 0.0003    # 买入万分之3
COMMISSION_SELL = 0.0013   # 卖出千分之1.3
SCORE_THRESHOLD_STD = 0.5  # 建仓/买入评分阈值标准差倍数
REBALANCE_THRESHOLD = 0.03 # 调仓最低评分差阈值(3%)
POSITION_LIMIT = 0.15      # 单只股票最大仓位15%
VOLUME_MIN = 100_000       # 最低日均成交量(手)，约折合1000万元
LIMIT_THRESHOLD = 9.5      # 涨跌停阈值(%)

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

# 全局调试开关：在调试模式下会执行额外的完整性检查（可能会扫描大数组）
DEBUG = False

# 随机种子（可通过环境变量覆盖）
SEED = 42
