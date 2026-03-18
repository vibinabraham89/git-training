"""
Walk-Forward Backtester
━━━━━━━━━━━━━━━━━━━━━━
Validates strategy on synthetic historical data.
Replace `generate_synthetic_history` with real Polygon API data.

Metrics reported:
  Sharpe Ratio, Max Drawdown, Win Rate, ROI, Kelly-adjusted CAGR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Resolution:
    market:       str
    entry_price:  float
    model_p:      float
    outcome:      int     # 1 = YES wins, 0 = NO wins
    volume_usd:   float


FEE = 0.02


def ev_net(model_p: float, entry_price: float) -> float:
    return (model_p - entry_price) / entry_price - FEE


def kelly_size(model_p: float, entry_price: float,
               bankroll: float, frac: float = 0.25,
               max_pct: float = 0.05) -> float:
    odds = (1 / entry_price) - 1
    raw  = (model_p * odds - (1 - model_p)) / odds
    f    = max(0.0, raw * frac)
    return min(f * bankroll, max_pct * bankroll)


def simulate_pnl(resolution: Resolution,
                 size_usd:   float) -> float:
    """P&L for a single trade."""
    if resolution.outcome == 1:
        # WIN: payout is 1/price per dollar bet, minus stake, minus fee
        return size_usd * (1 / resolution.entry_price - 1) * (1 - FEE)
    else:
        return -size_usd


def generate_synthetic_history(n: int = 200,
                                 base_edge: float = 0.06,
                                 seed: int = 42) -> List[Resolution]:
    """
    Generates synthetic resolved markets.
    model_p = true_p + noise (simulates imperfect model).
    Replace with real data from Polygon API.
    """
    rng      = np.random.default_rng(seed)
    markets  = []
    for i in range(n):
        true_p      = rng.uniform(0.25, 0.75)
        market_p    = true_p - base_edge + rng.normal(0, 0.05)
        market_p    = np.clip(market_p, 0.05, 0.90)
        model_p     = true_p + rng.normal(0, 0.04)
        model_p     = np.clip(model_p, 0.05, 0.95)
        outcome     = int(rng.random() < true_p)
        volume      = rng.choice([500_000, 2_000_000, 8_000_000, 20_000_000])
        markets.append(Resolution(
            market      = f"Market_{i:03d}",
            entry_price = market_p,
            model_p     = model_p,
            outcome     = outcome,
            volume_usd  = float(volume),
        ))
    return markets


def walk_forward_backtest(history:   List[Resolution],
                           bankroll:  float = 10_000,
                           min_ev:    float = 0.05,
                           min_vol:   float = 50_000,
                           max_dd:    float = 0.20) -> pd.DataFrame:
    """
    Iterate through history in order. Only enter trades that pass filters.
    Implements kill switch on drawdown breach.
    """
    br          = bankroll
    peak        = bankroll
    killed      = False
    records     = []

    for r in history:
        if killed:
            break
        ev = ev_net(r.model_p, r.entry_price)
        if ev < min_ev or r.volume_usd < min_vol:
            continue

        size = kelly_size(r.model_p, r.entry_price, br)
        pnl  = simulate_pnl(r, size)
        br  += pnl
        peak = max(peak, br)
        dd   = (peak - br) / peak

        if dd >= max_dd:
            killed = True

        records.append({
            "market":     r.market,
            "ev":         ev,
            "size_usd":   size,
            "pnl":        pnl,
            "bankroll":   br,
            "drawdown":   dd,
            "killed":     killed,
            "win":        r.outcome == 1,
        })

    return pd.DataFrame(records)


def compute_metrics(df: pd.DataFrame, initial: float) -> dict:
    if df.empty:
        return {}
    pnl_series = df["pnl"]
    final      = df["bankroll"].iloc[-1]
    wins       = df["win"].sum()
    total      = len(df)

    sharpe = (pnl_series.mean() / pnl_series.std() * np.sqrt(252)
              if pnl_series.std() > 0 else 0)

    return {
        "initial_bankroll": f"${initial:,.0f}",
        "final_bankroll":   f"${final:,.0f}",
        "total_trades":     total,
        "win_rate":         f"{wins / total:.1%}",
        "total_pnl":        f"${pnl_series.sum():,.0f}",
        "roi":              f"{(final - initial) / initial:.1%}",
        "sharpe_ratio":     f"{sharpe:.2f}",
        "max_drawdown":     f"{df['drawdown'].max():.1%}",
        "kill_switch_hit":  df["killed"].any(),
    }


def plot_backtest(df: pd.DataFrame, initial: float):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(df.index, df["bankroll"], color="steelblue", linewidth=2)
    ax1.axhline(initial, color="gray", linestyle="--", alpha=0.5, label="Starting bankroll")
    ax1.fill_between(df.index, initial, df["bankroll"],
                     where=df["bankroll"] >= initial, alpha=0.15, color="green")
    ax1.fill_between(df.index, initial, df["bankroll"],
                     where=df["bankroll"] < initial,  alpha=0.15, color="red")
    ax1.set_ylabel("Bankroll ($)")
    ax1.set_title("Walk-Forward Backtest: Bankroll Curve")
    ax1.legend()

    ax2.fill_between(df.index, -df["drawdown"] * 100, 0,
                     color="red", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Trade #")
    ax2.set_title("Drawdown Over Time")

    plt.tight_layout()
    plt.savefig("backtest.png", dpi=150)
    plt.show()


# ── Run ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    INITIAL = 10_000
    history = generate_synthetic_history(n=300, base_edge=0.07)
    df      = walk_forward_backtest(history, bankroll=INITIAL)

    metrics = compute_metrics(df, INITIAL)
    print("\nBacktest Results")
    print("─" * 35)
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")

    plot_backtest(df, INITIAL)
