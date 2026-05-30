import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import FEATURE_COLS


class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, lookback: int, horizon: int):
        self.lookback = lookback
        self.horizon = horizon
        self.feature_cols = FEATURE_COLS

        print(">>> [内存优化] 正在逐列提取特征矩阵，避免内存暴涨...")
        
        # 1. 提取元数据和标签，强制脱离原 df (copy=True)
        self.dates = df["trade_date"].to_numpy(copy=True)
        self.codes = df["ts_code"].to_numpy(copy=True)
        self.labels = df["label"].fillna(0).to_numpy(dtype=np.float32, copy=True)

        # 2. 预先分配好连续的空内存块，防止 Pandas 产生庞大的中间数据块
        num_samples = len(df)
        num_features = len(self.feature_cols)
        self.values = np.empty((num_samples, num_features), dtype=np.float32)
        
        # 3. 逐列填充数据（非常省内存）
        for i, col in enumerate(self.feature_cols):
            self.values[:, i] = df[col].to_numpy(dtype=np.float32, copy=False)
            
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

    def get_meta(self, idx: int) -> tuple[str, str]:
        offset, end_i = self.index_map[idx]
        global_end = offset + end_i
        return str(self.dates[global_end]), str(self.codes[global_end])