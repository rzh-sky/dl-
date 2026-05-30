import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    DEFAULT_END,
    DEFAULT_START,
    DROPOUT,
    EPOCHS,
    FEATURE_COLS,
    HIDDEN,
    HORIZON,
    LOOKBACK,
    LR,
    MODEL_NAME,
    NUM_LAYERS,
    PROCESSED_DIR,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)
from src.dataset import SequenceDataset
from src.model import build_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        # 【暴力破解点 1】：强制将输入的特征和标签转为 32 位 Float
        x, y = x.float().to(device), y.float().to(device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            # 【暴力破解点 2】：验证集也必须强制转为 32 位 Float
            x, y = x.float().to(device), y.float().to(device)
            
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds.extend(out.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
    return total_loss / len(loader.dataset), preds, labels


def generate_predictions(model, dataset, preds, out_path: Path):
    records = []
    for i in range(len(dataset)):
        trade_date, ts_code = dataset.get_meta(i)
        records.append(
            {
                "trade_date": trade_date,
                "ts_code": ts_code,
                "score": preds[i],
                "label": dataset.labels[dataset.index_map[i][0] + dataset.index_map[i][1]],
            }
        )
    out_df = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train sequence model on A-share data.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROCESSED_DIR / "features.parquet"),
        help="Path to processed features parquet",
    )
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--train-start", type=str, default=TRAIN_START)
    parser.add_argument("--train-end", type=str, default=TRAIN_END)
    parser.add_argument("--val-start", type=str, default=VAL_START)
    parser.add_argument("--val-end", type=str, default=VAL_END)
    parser.add_argument("--pred-out", type=str, default="outputs/preds/val_predictions.csv")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Feature file not found: {data_path}")

    print(">>> 正在从 Parquet 按需加载【训练集】数据...")
    df_train = pd.read_parquet(
        data_path, 
        filters=[('trade_date', '>=', args.train_start), ('trade_date', '<=', args.train_end)]
    )
    train_ds = SequenceDataset(df_train, args.lookback, args.horizon)
    del df_train
    gc.collect()

    print(">>> 正在从 Parquet 按需加载【验证集】数据...")
    df_val = pd.read_parquet(
        data_path, 
        filters=[('trade_date', '>=', args.val_start), ('trade_date', '<=', args.val_end)]
    )
    val_ds = SequenceDataset(df_val, args.lookback, args.horizon)
    del df_val
    gc.collect()

    # 【修复这里】：在 Windows 大内存矩阵下，务必将 num_workers 设置为 0，防止多进程 pickle 内存爆炸
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    input_dim = len(FEATURE_COLS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 【暴力破解点 3】：加上 .float()，确保无论什么模型，所有初始化的权重都是 32 位
    model = build_model(
        name=args.model,
        input_dim=input_dim,
        hidden_dim=HIDDEN,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_preds = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, preds, labels = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch [{epoch}/{EPOCHS}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds = preds

    pred_out_path = Path(args.pred_out)
    if best_preds is not None:
        generate_predictions(model, val_ds, best_preds, pred_out_path)


if __name__ == "__main__":
    main()