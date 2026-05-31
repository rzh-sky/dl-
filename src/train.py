import argparse
import gc
import os
import random
import numpy as np
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    DROPOUT,
    EPOCHS,
    FEATURE_COLS,
    GRAD_CLIP,
    HIDDEN,
    HORIZON,
    LOOKBACK,
    LR,
    LR_FACTOR,
    LR_MIN,
    LR_PATIENCE,
    EARLY_STOP_PATIENCE,
    MODEL_DIR,
    MODEL_NAME,
    NUM_LAYERS,
    PROCESSED_DIR,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)
from typing import Optional
from src.config import SEED
from src.dataset import SequenceDataset
from src.model import build_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.float().to(device), y.float().to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        # 梯度裁剪：防止 RNN 梯度爆炸
        if GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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
            x, y = x.float().to(device), y.float().to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds.extend(out.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())
    return total_loss / len(loader.dataset), preds, labels


def generate_predictions(model, dataset, preds, out_path: Path, data_path: Optional[Path] = None):
    row_indices = [offset + end_i for offset, end_i in dataset.index_map]
    out_df = pd.DataFrame(
        {
            "trade_date": dataset.dates[row_indices],
            "ts_code": dataset.codes[row_indices],
            "score": preds,
            "label": dataset.labels[row_indices],
        }
    )
    # 加入回测辅助列（优先从 dataset 取，没有则从 parquet 补齐）
    aux_cols = ["ret_next", "pct_chg", "vol", "close"]
    missing = []
    for col in aux_cols:
        arr = getattr(dataset, col, None)
        if arr is not None:
            out_df[col] = arr[row_indices]
        else:
            missing.append(col)

    if missing and data_path is not None and data_path.exists():
        print(f"  >> Fallback: loading {missing} from {data_path}")
        # 推断预测集的时间范围
        dates = out_df["trade_date"].unique()
        date_min, date_max = str(dates.min()), str(dates.max())
        aux_df = pd.read_parquet(
            data_path,
            columns=["trade_date", "ts_code"] + missing,
            filters=[("trade_date", ">=", date_min), ("trade_date", "<=", date_max)],
        )
        out_df = out_df.merge(aux_df, on=["trade_date", "ts_code"], how="left")
    elif missing:
        print(f"  >> Warning: missing columns {missing}, no fallback data provided")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions ({len(out_df)} rows) to {out_path}")
    return out_df


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
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Feature file not found: {data_path}")

    # 设置随机种子
    seed = int(os.environ.get("SEED", args.seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── 加载数据 ──
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

    worker_count = 0 if os.name == "nt" else min(4, max(1, (os.cpu_count() or 2) // 2))
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=worker_count,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=use_pin_memory,
    )

    input_dim = len(FEATURE_COLS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | input_dim={input_dim} | model={args.model}")

    model = build_model(
        name=args.model,
        input_dim=input_dim,
        hidden_dim=HIDDEN,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 学习率调度：验证集 loss 不再下降时衰减 LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE,
        min_lr=LR_MIN, verbose=True,
    )
    criterion = nn.MSELoss()

    # ── 训练循环 ──
    best_val_loss = float("inf")
    best_preds = None
    best_epoch = 0
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, preds, labels = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)  # 更新 LR

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch:2d}/{args.epochs}] "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds = preds
            best_epoch = epoch
            early_stop_counter = 0
            # 保存最佳模型权重
            model_path = MODEL_DIR / f"{args.model}_best.pt"
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"  >> New best model saved to {model_path} (val_loss={val_loss:.6f})")
        else:
            early_stop_counter += 1

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f">> Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    print(f"\n>>> Training complete. Best epoch: {best_epoch}, best val_loss: {best_val_loss:.6f}")

    # ── 输出预测 ──
    if best_preds is not None:
        pred_out_path = Path(args.pred_out)
        generate_predictions(model, val_ds, best_preds, pred_out_path, data_path)

    # 输出最终 train / val loss
    print(f"Final — Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")


if __name__ == "__main__":
    main()
