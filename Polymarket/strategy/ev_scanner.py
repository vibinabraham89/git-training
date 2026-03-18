"""
Formula 3: Expected Value Gap Scanner
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best for: Geopolitics (Iran/Russia), earnings, news-reactive markets
Edge:     Markets lag real-time news by 2-10 min. Your model is faster.
Signal:   EV = (p_true − price) × (1/price);  enter only if EV > 0.05
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from dataclasses import dataclass


POLYMARKET_FEE = 0.02   # ~2% round-trip


@dataclass
class MarketSignal:
    name:         str
    market_price: float   # current implied prob
    model_p:      float   # your estimated true prob
    volume_usd:   float   # 24h volume


def ev_gap(model_p: float, market_price: float,
           fee: float = POLYMARKET_FEE) -> float:
    """
    Net EV after fees.
    Payout = 1/price on a $1 bet if correct.
    EV = (p_true − price) / price − fee
    """
    payout = 1 / market_price
    return (model_p - market_price) * payout - fee


def rank_opportunities(signals: list[MarketSignal],
                       min_ev:    float = 0.05,
                       min_vol:   float = 50_000) -> pd.DataFrame:
    """
    Score and rank a list of MarketSignals.
    Returns DataFrame sorted by expected value, filtered for quality.
    """
    rows = []
    for s in signals:
        ev    = ev_gap(s.model_p, s.market_price)
        edge  = s.model_p - s.market_price
        kelly = max(0, (s.model_p * ((1/s.market_price) - 1) - (1 - s.model_p))
                    / ((1/s.market_price) - 1))
        rows.append({
            "market":       s.name,
            "price":        s.market_price,
            "model_p":      s.model_p,
            "edge":         edge,
            "ev_net":       ev,
            "kelly_f":      kelly * 0.25,   # quarter-Kelly
            "vol_usd":      s.volume_usd,
            "enter":        ev > min_ev and s.volume_usd >= min_vol,
        })
    df = pd.DataFrame(rows).sort_values("ev_net", ascending=False)
    return df


def build_model_p_from_polls(poll_values: list[float],
                              weights: Optional[list[float]] = None) -> float:
    """
    Simple weighted average of external signals (polls, news scores, sentiment).
    Replace with your NLP / regression model output.
    """
    arr = np.array(poll_values)
    w   = np.ones(len(arr)) if weights is None else np.array(weights)
    w   = w / w.sum()
    return float(np.dot(arr, w))


def plot_ev_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["green" if e else "salmon" for e in df["enter"]]
    ax.barh(df["market"], df["ev_net"], color=colors)
    ax.axvline(0.05, color="orange", linestyle="--", label="EV threshold (5%)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Net EV (after 2% fees)")
    ax.set_title("Expected Value Gap Scanner")
    ax.legend()
    plt.tight_layout()
    plt.savefig("ev_scanner.png", dpi=150)
    plt.show()


# ── Demo ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulated signals — replace with live Polygon / CLOB data
    signals = [
        MarketSignal("Iran ceasefire by June",    0.47, 0.52, 5_000_000),
        MarketSignal("BTC >100K end of March",    0.38, 0.42, 12_000_000),
        MarketSignal("Vance 2028 nominee",        0.21, 0.28, 3_200_000),
        MarketSignal("Russia ceasefire Q2",       0.33, 0.31,   800_000),
        MarketSignal("Oscars: Sinners Best Pic",  0.15, 0.14,   400_000),
        MarketSignal("Fed rate cut March 2026",   0.60, 0.72, 18_000_000),
        MarketSignal("Musk tweets >50 today",     0.40, 0.55,   150_000),  # thin
    ]

    df = rank_opportunities(signals)
    print(df.to_string(index=False))
    print(f"\nActionable trades: {df['enter'].sum()}")
    plot_ev_distribution(df)
