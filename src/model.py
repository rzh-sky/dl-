import torch
import torch.nn as nn


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        if module.in_proj_weight is not None:
            nn.init.xavier_uniform_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
        nn.init.xavier_uniform_(module.out_proj.weight)
        if module.out_proj.bias is not None:
            nn.init.zeros_(module.out_proj.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                hidden_size = param.data.shape[0] // 4
                param.data[hidden_size: 2 * hidden_size] = 1.0
    elif isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = None if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.residual_act = nn.GELU()

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.act2(out)
        out = self.drop2(out)

        residual = x if self.downsample is None else self.downsample(x)
        return self.residual_act(out + residual)


class TemporalConvNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_levels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        blocks = []
        in_channels = hidden_dim
        for level in range(num_levels):
            dilation = 2 ** level
            blocks.append(TemporalBlock(in_channels, hidden_dim, kernel_size, dilation, dropout))
            in_channels = hidden_dim
        self.network = nn.Sequential(*blocks)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.network(x)
        return x


class PredictionHead(nn.Module):
    """共享的预测头：LayerNorm → Linear → GELU → Dropout → Linear"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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
        self.head = PredictionHead(hidden_dim, dropout)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


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
        self.head = PredictionHead(hidden_dim, dropout)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return self.head(h_n[-1])


class TCNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, kernel_size: int = 3, num_levels: int = 4):
        super().__init__()
        self.tcn = TemporalConvNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_levels=num_levels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.head = PredictionHead(hidden_dim, dropout)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.tcn(x)
        return self.head(feat[:, :, -1])


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_len: int = 512,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.max_len = max_len
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        self.pos_drop = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = PredictionHead(hidden_dim, dropout)

        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}")

        x = self.input_norm(x)
        x = self.proj(x)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.pos_drop(x)
        h = self.encoder(x)
        return self.head(h[:, -1, :])


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