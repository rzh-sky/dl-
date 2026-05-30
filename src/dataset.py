import gc
import numpy as np
import pandas as pd
import torch
from typing import Optional
from torch.utils.data import Dataset

from src.config import FEATURE_COLS, DEBUG


class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, lookback: int, horizon: int, check_nan: Optional[bool] = None):
        self.lookback = lookback
        self.horizon = horizon
        self.feature_cols = FEATURE_COLS

        # check_nan: whether to run full-array NaN assertions. Default由全局 DEBUG 控制。
        if check_nan is None:
            check_nan = bool(DEBUG)

        # 该 Dataset 依赖股票行在内存中按 ts_code 物理连续、且组内按 trade_date 升序排列。
        assert df["ts_code"].is_monotonic_increasing, "df must be sorted by ts_code before building SequenceDataset"
        assert df.groupby("ts_code", sort=False)["trade_date"].apply(lambda s: s.is_monotonic_increasing).all(), \
            "each ts_code group must be sorted by trade_date before building SequenceDataset"

        print(">>> [内存优化] 正在逐列提取特征矩阵，避免内存暴涨...")
        
        # 1. 提取元数据和标签，强制脱离原 df (copy=True)
        self.dates = df["trade_date"].to_numpy(copy=True)
        self.codes = df["ts_code"].to_numpy(copy=True)
        self.labels = df["label"].fillna(0).to_numpy(dtype=np.float32, copy=True)

        # 回测辅助列（可选，由 data_preprocess 产出）
        self.ret_next = df["ret_next"].to_numpy(dtype=np.float32, copy=True) if "ret_next" in df else None
        self.pct_chg = df["pct_chg"].to_numpy(dtype=np.float32, copy=True) if "pct_chg" in df else None
        self.vol = df["vol"].to_numpy(dtype=np.float32, copy=True) if "vol" in df else None
        self.close = df["close"].to_numpy(dtype=np.float32, copy=True) if "close" in df else None

        # 2. 预先分配好连续的空内存块，防止 Pandas 产生庞大的中间数据块
        num_samples = len(df)
        num_features = len(self.feature_cols)
        self.values = np.empty((num_samples, num_features), dtype=np.float32)
        
        # 3. 逐列填充数据（非常省内存）
        for i, col in enumerate(self.feature_cols):
            self.values[:, i] = df[col].to_numpy(dtype=np.float32, copy=False)

        if check_nan:
            # 此处会完整扫描 large array，仅在调试模式下启用
            if np.isnan(self.values).any():
                raise ValueError("feature matrix contains NaN after preprocessing")
            if np.isnan(self.labels).any():
                raise ValueError("labels contain NaN after preprocessing")
            
        print(">>> [内存优化] 正在计算滑动窗口索引...")
        self.index_map = self._build_index(df)

    def _build_index(self, df: pd.DataFrame) -> list[tuple[int, int]]:
        index_map: list[tuple[int, int]] = []
        group_offsets: dict[str, int] = {}

        # 直接统计每只股票的长度，避免遍历 groupby 对象产生高昂内存开销
        group_sizes = df.groupby("ts_code", sort=False).size()
        
        offset = 0
        for ts_code, n in group_sizes.items():
            group_offsets[ts_code] = offset
            # 只有当数据长度足够滑动窗口时才加入
            if n >= self.lookback + self.horizon:
                for end_i in range(self.lookback - 1, n - self.horizon):
                    index_map.append((offset, end_i))
            offset += n

        return index_map

    @staticmethod
    def collate_fn(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        offset, end_i = self.index_map[idx]
        start_i = end_i - self.lookback + 1

        global_start = offset + start_i
        global_end = offset + end_i + 1

        x = self.values[global_start:global_end]
        y = self.labels[global_end - 1]

        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        return x_t, y_t

    def get_meta(self, idx: int) -> dict[str, object]:
        offset, end_i = self.index_map[idx]
        # end_i is the index (0-based) within the group pointing to the sample end row
        start_i = end_i - self.lookback + 1
        global_start = offset + start_i
        row_idx = offset + end_i
        meta = {
            "start_date": str(self.dates[global_start]),
            "date": str(self.dates[row_idx]),
            "code": str(self.codes[row_idx]),
            "label": float(self.labels[row_idx]),
        }
        if self.ret_next is not None:
            meta["ret_next"] = float(self.ret_next[row_idx])
        if self.pct_chg is not None:
            meta["pct_chg"] = float(self.pct_chg[row_idx])
        if self.vol is not None:
            meta["vol"] = float(self.vol[row_idx])
        if self.close is not None:
            meta["close"] = float(self.close[row_idx])
        return meta