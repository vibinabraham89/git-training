"""
Formula 1: LMSR Pricing Model
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best for: Crypto 5-min markets, Esports (low b → fat price impact per trade)
Edge:     Pre-calculate price impact before others. Buy before the move, sell after.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List


@dataclass
class LMSRMarket:
    b: float           # liquidity depth (lower = more impact per trade)
    quantities: List[float]  # q vector, one per outcome


def lmsr_price(q: np.ndarray, b: float) -> np.ndarray:
    """
    Core LMSR formula:  p_i = exp(q_i / b) / Σ exp(q_j / b)
    Returns probability vector for all outcomes.
    """
    shifted = q - q.max()          # numerical stability (log-sum-exp trick)
    exps    = np.exp(shifted / b)
    return exps / exps.sum()


def price_impact(market: LMSRMarket, outcome_idx: int, shares: float) -> dict:
    """
    Compute price BEFORE and AFTER buying `shares` of outcome `outcome_idx`.
    Returns the edge window for arb entry.
    """
    q_before = np.array(market.quantities, dtype=float)
    q_after  = q_before.copy()
    q_after[outcome_idx] += shares

    p_before = lmsr_price(q_before, market.b)
    p_after  = lmsr_price(q_after,  market.b)

    cost = market.b * (
        np.log(np.exp(q_after  / market.b).sum()) -
        np.log(np.exp(q_before / market.b).sum())
    )

    return {
        "price_before": p_before[outcome_idx],
        "price_after":  p_after[outcome_idx],
        "price_impact": p_after[outcome_idx] - p_before[outcome_idx],
        "cost_usdc":    cost,
        "avg_fill":     cost / shares,  # effective average price paid
    }


def scan_arb_window(market: LMSRMarket, outcome_idx: int,
                    true_prob: float, max_shares: float = 500) -> dict:
    """
    Find the optimal share quantity that maximises +EV before impact erodes edge.
    Strategy: Buy until avg_fill ≈ true_prob (edge = 0).
    """
    best = {"shares": 0, "ev": 0.0}
    for shares in np.arange(10, max_shares, 10):
        impact = price_impact(market, outcome_idx, shares)
        ev     = (true_prob - impact["avg_fill"]) * shares
        if ev > best["ev"]:
            best = {"shares": shares, "ev": ev, **impact}
        else:
            break  # EV is declining — stop
    return best


def plot_lmsr_curve(b: float = 100, max_q: float = 1000):
    qs     = np.linspace(0, max_q, 300)
    prices = [lmsr_price(np.array([q, 0.0]), b)[0] for q in qs]
    plt.figure(figsize=(8, 4))
    plt.plot(qs, prices, color="steelblue", linewidth=2)
    plt.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Fair coin")
    plt.xlabel("YES Shares Bought")
    plt.ylabel("YES Price (Implied Prob)")
    plt.title(f"LMSR Pricing Curve  (b={b})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lmsr_curve.png", dpi=150)
    plt.show()


# ── Demo ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # BTC 5-min up/down, thin pool
    btc_market = LMSRMarket(b=80, quantities=[0.0, 0.0])  # YES / NO
    result = scan_arb_window(btc_market, outcome_idx=0, true_prob=0.55)
    print("LMSR Arb Window (BTC 5-min):")
    for k, v in result.items():
        print(f"  {k:15s}: {v:.4f}" if isinstance(v, float) else f"  {k:15s}: {v}")

    plot_lmsr_curve(b=80)
