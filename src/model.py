import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        last = h_n[-1]
        out = self.head(last)
        return out.squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        last = h_n[-1]
        out = self.head(last)
        return out.squeeze(-1)


class TCNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 【修复点】：转置维度，将 [batch, seq_len, features] 转为 [batch, features, seq_len]
        x = x.transpose(1, 2)
        feat = self.net(x).squeeze(-1)
        out = self.head(feat)
        return out.squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        h = self.encoder(x)
        last = h[:, -1, :]
        out = self.head(last)
        return out.squeeze(-1)


def build_model(
    name: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    num_heads: int = 4,
):
    name = name.lower()
    if name == "lstm":
        return LSTMRegressor(input_dim, hidden_dim, num_layers, dropout)
    if name == "gru":
        return GRURegressor(input_dim, hidden_dim, num_layers, dropout)
    if name == "tcn":
        return TCNRegressor(input_dim, hidden_dim, dropout)
    if name == "transformer":
        return TransformerRegressor(input_dim, hidden_dim, num_layers, num_heads, dropout)
    raise ValueError(f"Unknown model name: {name}")