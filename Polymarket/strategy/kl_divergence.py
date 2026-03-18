"""
Formula 4: KL-Divergence Correlation Scanner
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best for: Correlated political pairs (Vance vs Newsom 2028),
          Party control (Senate D vs Senate R), crypto ecosystem bets
Edge:     Two markets pricing the SAME reality differently = arb via hedging.
Signal:   D_KL > 0.20 → trade the cheaper, hedge the expensive.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MarketPair:
    name_a:   str
    name_b:   str
    price_a:  float   # YES price of market A
    price_b:  float   # YES price of market B
    volume_a: float
    volume_b: float
    # Are these mutually exclusive outcomes? (e.g. one person wins an election)
    mutex: bool = False


def kl_divergence(p: np.ndarray, q: np.ndarray,
                  eps: float = 1e-10) -> float:
    """D_KL(P||Q) = Σ P_i log(P_i / Q_i)  — asymmetric."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return float(entropy(p, q))


def symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon proxy: average of both directions."""
    return 0.5 * (kl_divergence(p, q) + kl_divergence(q, p))


def build_dist(price: float) -> np.ndarray:
    """Binary distribution from a YES price."""
    return np.array([price, 1 - price])


def analyze_pair(pair: MarketPair, kl_threshold: float = 0.20) -> dict:
    """
    Full KL analysis for a correlated pair.
    Hedge signal: buy the underpriced market, sell/no-buy the overpriced.
    """
    dist_a = build_dist(pair.price_a)
    dist_b = build_dist(pair.price_b)

    kl_ab   = kl_divergence(dist_a, dist_b)
    kl_ba   = kl_divergence(dist_b, dist_a)
    sym_kl  = symmetric_kl(dist_a, dist_b)

    # If mutex: prices should sum closer to 1 across all outcomes
    mutex_gap = abs(pair.price_a + pair.price_b - 1.0) if pair.mutex else None

    underpriced = (
        pair.name_a if pair.price_a < pair.price_b else pair.name_b
    )

    return {
        "pair":         f"{pair.name_a} / {pair.name_b}",
        "price_a":      pair.price_a,
        "price_b":      pair.price_b,
        "kl_ab":        kl_ab,
        "kl_ba":        kl_ba,
        "symmetric_kl": sym_kl,
        "mutex_gap":    mutex_gap,
        "signal":       sym_kl > kl_threshold,
        "action":       f"Buy {underpriced}" if sym_kl > kl_threshold else "No trade",
        "avg_volume":   (pair.volume_a + pair.volume_b) / 2,
    }


def scan_pairs(pairs: List[MarketPair], threshold: float = 0.20) -> pd.DataFrame:
    rows = [analyze_pair(p, threshold) for p in pairs]
    df   = pd.DataFrame(rows).sort_values("symmetric_kl", ascending=False)
    return df


def plot_kl_heatmap(pairs: List[MarketPair]):
    """
    Visualise symmetric KL across all pairs as a bar chart.
    """
    results = scan_pairs(pairs)
    colors  = ["green" if s else "steelblue" for s in results["signal"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(results["pair"], results["symmetric_kl"], color=colors)
    ax.axvline(0.20, color="red", linestyle="--", label="Arb threshold (0.20)")
    ax.set_xlabel("Symmetric KL Divergence")
    ax.set_title("KL Divergence: Correlated Market Pair Scanner")
    ax.legend()
    plt.tight_layout()
    plt.savefig("kl_heatmap.png", dpi=150)
    plt.show()


# ── Demo ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pairs = [
        # 2028 Presidential — mutex pair (only one wins)
        MarketPair("Vance 2028",  "Newsom 2028",  0.21, 0.17, 3_200_000, 2_800_000, mutex=True),
        MarketPair("Vance 2028",  "Harris 2028",  0.21, 0.14, 3_200_000, 2_100_000, mutex=True),
        # Crypto ecosystem correlation
        MarketPair("BTC >100K",   "ETH >5K",      0.38, 0.29, 12_000_000, 8_000_000),
        # Senate control
        MarketPair("Senate D maj","Senate R maj",  0.42, 0.55, 5_000_000,  5_200_000, mutex=True),
        # Geopolitics linked
        MarketPair("Iran deal Q2","Russia deal Q2",0.47, 0.33, 5_000_000,  2_000_000),
    ]

    df = scan_pairs(pairs)
    print(df[["pair", "symmetric_kl", "mutex_gap", "signal", "action"]].to_string(index=False))
    plot_kl_heatmap(pairs)
