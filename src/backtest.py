import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BACKTEST_DIR,
    PRED_DIR,
    TARGET_N_HOLD,
    REBALANCE_INTERVAL,
    COMMISSION_BUY,
    COMMISSION_SELL,
    SCORE_THRESHOLD_STD,
    REBALANCE_THRESHOLD,
    POSITION_LIMIT,
    VOLUME_MIN,
    LIMIT_THRESHOLD,
)


def normalize_weights(scores: list[float], n_hold: int, limit: float) -> np.ndarray:
    """排名加权（rank=1→n, rank=n→1），单只仓位上限约束后归一化。"""
    sorted_idx = np.argsort(scores)[::-1]  # 最高分排第0位
    ranks = np.argsort(sorted_idx) + 1      # 1 = 最好
    w = np.array([n_hold + 1 - r for r in ranks], dtype=float)
    w = w / w.sum()
    # 迭代截断上限
    for _ in range(20):
        over = w > limit
        if not over.any():
            break
        excess = (w[over] - limit).sum()
        w[over] = limit
        under = ~over
        if under.sum() == 0:
            return w
        w[under] += excess * w[under] / w[under].sum()
    return w


def run_backtest(pred_path: Path) -> pd.DataFrame:
    preds = pd.read_csv(pred_path, dtype={"trade_date": str, "ts_code": str})
    required = {"trade_date", "ts_code", "score", "ret_next", "pct_chg", "vol"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing required columns: {missing}")

    preds = preds.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    # ── 状态变量 ──
    equity = 1.0
    positions: set[str] = set()               # 当前持仓股票代码
    position_weights: dict[str, float] = {}   # 前一交易日收盘后的权重
    prev_ret_map: dict[str, float] = {}       # 前一交易日的 ret_next
    rebalance_counter = 0
    toggle_flag = 0
    daily_returns: list[float] = []           # 用于 20 日波动率计算
    records: list[dict] = []

    # ── 逐日回测 ──
    for idx, (date, group) in enumerate(preds.groupby("trade_date")):
        group = group.copy()
        ret_map = dict(zip(group["ts_code"], group["ret_next"]))
        scores_map = dict(zip(group["ts_code"], group["score"]))

        # ── 可交易池（用于调仓决策） ──
        tradable = group[group["vol"] >= VOLUME_MIN]
        tradable = tradable[tradable["pct_chg"].abs() <= LIMIT_THRESHOLD]
        tradable = tradable.sort_values("score", ascending=False).reset_index(drop=True)

        # ── 当日市场统计（基于可交易池） ──
        if len(tradable) > 0:
            score_mean = float(tradable["score"].mean())
            score_std = float(tradable["score"].std(ddof=0)) if len(tradable) > 1 else 0.0
            score_threshold = score_mean + SCORE_THRESHOLD_STD * score_std
        else:
            score_mean = score_std = score_threshold = 0.0

        # ═══════════════════════════════════════════════════════════
        #  第一步：组合收益（基于前一日权重 × 前一日 ret_next）
        # ═══════════════════════════════════════════════════════════
        if idx == 0:
            port_ret = 0.0
        else:
            port_ret = sum(
                prev_ret_map.get(code, 0.0) * w
                for code, w in position_weights.items()
                if code in prev_ret_map
            )

        equity_after_ret = equity * (1.0 + port_ret)

        # ═══════════════════════════════════════════════════════════
        #  第二步：调仓执行
        # ═══════════════════════════════════════════════════════════
        sell_cost_frac = 0.0
        buy_cost_frac = 0.0
        trades = 0

        if idx == 0:
            # ── 首次建仓 ──
            qualified = tradable[tradable["score"] >= score_threshold]
            if len(qualified) >= TARGET_N_HOLD:
                selected = qualified.head(TARGET_N_HOLD)["ts_code"].tolist()
            else:
                selected = tradable.head(TARGET_N_HOLD)["ts_code"].tolist()
            positions = set(selected)
            trades = len(selected)
            buy_cost_frac = trades / TARGET_N_HOLD
            rebalance_counter = 0

        elif rebalance_counter >= REBALANCE_INTERVAL:
            rebalance_counter = 0

            # ── 自适应调仓数量 ──
            if len(daily_returns) >= 20:
                vol_20 = float(np.std(daily_returns[-20:], ddof=0))
            else:
                vol_20 = 0.01  # 默认假设 1%

            if vol_20 < 0.01:
                n_raw = 1
            elif vol_20 < 0.02:
                n_raw = 0.5
            else:
                n_raw = 0

            if 0 < n_raw < 1:
                toggle_flag = 1 - toggle_flag
                n_trade = 1 if toggle_flag else 0
            else:
                n_trade = int(n_raw)
                toggle_flag = 0

            if n_trade > 0 and positions:
                # 卖出候选：持仓 ∩ 可交易，评分低于均值，取最低 N 只
                active_pos = positions & set(tradable["ts_code"])
                pos_tradable_df = tradable[tradable["ts_code"].isin(active_pos)]
                sell_candidates = pos_tradable_df[
                    pos_tradable_df["score"] < score_mean
                ].sort_values("score")
                to_sell = set(sell_candidates.head(n_trade)["ts_code"].tolist())

                if to_sell:
                    # 买入候选：可交易 \ 持仓，评分≥阈值，取最高 N 只
                    non_pos = tradable[~tradable["ts_code"].isin(positions)]
                    buy_candidates = non_pos[
                        non_pos["score"] >= score_threshold
                    ].sort_values("score", ascending=False)
                    to_buy = buy_candidates.head(n_trade)

                    if len(to_buy) >= n_trade:
                        highest_buy = to_buy["score"].iloc[0]
                        lowest_sell = min(scores_map.get(c, -999.0) for c in to_sell)

                        # 动态调仓阈值：买入最高评分 > 卖出最低评分 * (1+3%)
                        if highest_buy > lowest_sell * (1.0 + REBALANCE_THRESHOLD):
                            for code in to_sell:
                                positions.discard(code)
                                position_weights.pop(code, None)
                            for code in to_buy["ts_code"].tolist():
                                positions.add(code)
                            trades = n_trade
                            sell_cost_frac = sum(
                                position_weights.get(c, 1.0 / TARGET_N_HOLD)
                                for c in to_sell
                            )
                            buy_cost_frac = n_trade / TARGET_N_HOLD

        # ── 交易成本 ──
        cost_frac = (
            sell_cost_frac * COMMISSION_SELL
            + buy_cost_frac * COMMISSION_BUY
        )
        equity = equity_after_ret * (1.0 - cost_frac)

        # ═══════════════════════════════════════════════════════════
        #  第三步：重新计算持仓权重（排名加权 + 仓位上限）
        # ═══════════════════════════════════════════════════════════
        if positions:
            held = group[group["ts_code"].isin(positions)][
                ["ts_code", "score"]
            ].dropna()
            if len(held) >= TARGET_N_HOLD:
                held = held.sort_values("score", ascending=False)
                w = normalize_weights(
                    held["score"].tolist(), TARGET_N_HOLD, POSITION_LIMIT
                )
                position_weights = dict(zip(held["ts_code"], w))
            else:
                # 部分持仓股票当日无数据（停牌等）→ 保留旧权重，再归一化
                new_w = {}
                if not held.empty:
                    held = held.sort_values("score", ascending=False)
                    w = normalize_weights(
                        held["score"].tolist(), len(held), POSITION_LIMIT
                    )
                    new_w.update(zip(held["ts_code"], w))
                for code in positions:
                    if code not in new_w and code in position_weights:
                        new_w[code] = position_weights[code]
                w_sum = sum(new_w.values())
                if w_sum > 0:
                    position_weights = {k: v / w_sum for k, v in new_w.items()}
                else:
                    position_weights = {
                        k: 1.0 / len(positions) for k in positions
                    }

        # ── 保存状态，准备下一日 ──
        daily_ret_adjusted = port_ret - cost_frac
        daily_returns.append(daily_ret_adjusted)
        rebalance_counter += 1
        prev_ret_map = dict(ret_map)

        records.append(
            {
                "trade_date": date,
                "n_positions": len(positions),
                "daily_return": daily_ret_adjusted,
                "equity": equity,
                "trades": trades,
            }
        )

    return pd.DataFrame(records)


def compute_backtest_metrics(
    preds: pd.DataFrame, backtest_result: pd.DataFrame
) -> dict[str, float]:
    """计算 IC/ICIR 及组合绩效指标。"""
    daily = preds.groupby("trade_date", sort=True)
    ic_values = []
    for _, group in daily:
        if len(group) < 2:
            continue
        score = group["score"].to_numpy(dtype=float)
        label = group["label"].to_numpy(dtype=float)
        sm, lm = float(np.mean(score)), float(np.mean(label))
        ss, ls = float(np.std(score, ddof=0)), float(np.std(label, ddof=0))
        if math.isclose(ss, 0.0) or math.isclose(ls, 0.0):
            continue
        cov = float(np.mean((score - sm) * (label - lm)))
        ic = cov / (ss * ls)
        if pd.notna(ic):
            ic_values.append(ic)

    ic_mean = (
        float(pd.Series(ic_values).mean()) if ic_values else float("nan")
    )
    ic_std = (
        float(pd.Series(ic_values).std(ddof=0)) if ic_values else float("nan")
    )
    icir = (
        ic_mean / ic_std
        if ic_values
        and pd.notna(ic_std)
        and not math.isclose(ic_std, 0.0, abs_tol=1e-12)
        else float("nan")
    )

    equity = backtest_result["equity"].astype(float)
    daily_ret = backtest_result["daily_return"].astype(float)
    avg_daily = float(daily_ret.mean()) if not daily_ret.empty else float("nan")
    std_daily = (
        float(daily_ret.std(ddof=0)) if not daily_ret.empty else float("nan")
    )
    total_return = (
        float(equity.iloc[-1] - 1.0) if not equity.empty else float("nan")
    )
    annualized_return = (
        float((equity.iloc[-1] ** (252.0 / len(equity))) - 1.0)
        if not equity.empty and len(equity) > 1
        else float("nan")
    )
    sharpe = (
        float((avg_daily / std_daily) * math.sqrt(252))
        if pd.notna(std_daily) and not math.isclose(std_daily, 0.0)
        else float("nan")
    )
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1.0
    max_drawdown = (
        float(drawdown.min()) if not drawdown.empty else float("nan")
    )

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run strategy backtest with predictions."
    )
    default_pred = PRED_DIR / "val_predictions.csv"
    parser.add_argument(
        "--pred", type=str, default=str(default_pred), help="Path to predictions csv"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(BACKTEST_DIR / "backtest.csv"),
        help="Output csv path",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {pred_path}. Run training first."
        )

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_backtest(pred_path)
    result.to_csv(out_path, index=False)
    print(f"Saved backtest to {out_path}")

    if not result.empty:
        preds = pd.read_csv(pred_path, dtype={"trade_date": str, "ts_code": str})
        summary = compute_backtest_metrics(preds, result)
        print("Backtest summary:")
        print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
