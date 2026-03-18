"""
Formula 5: Bregman Projection — Multi-Outcome Arb Optimizer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best for: Oscars Best Picture, NBA Champion, World Cup, US Election (multi-candidate)
Edge:     All outcomes MUST sum to 1. When they don't, you have risk-free arb.
          Bregman finds the projection that reveals which legs to buy/sell.

Products with most value:
  • Oscars (21M vol, 10+ candidates)
  • NBA/NFL championship (tournament bracket inconsistencies)
  • 2028 Presidential primary (10+ candidates, prices often sum to >1)
  • Crypto: which chain hits $1T first
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class MultiOutcomeMarket:
    name:     str
    outcomes: List[str]
    prices:   List[float]   # current market prices (may not sum to 1)
    volume:   float


def market_inconsistency(prices: List[float]) -> dict:
    """
    Check if prices violate the probability simplex.
    total > 1.0 → vig / over-round (market maker profit built in)
    total < 1.0 → under-round → pure arb exists (buy all outcomes)
    """
    total   = sum(prices)
    overvig = total - 1.0
    return {
        "total":       total,
        "over_round":  overvig,
        "arb_exists":  total < 1.0,
        "vig_pct":     overvig * 100,
    }


def bregman_project(theta: np.ndarray) -> np.ndarray:
    """
    Project theta onto probability simplex via KL-Bregman:
    min Σ μ_i log(μ_i / θ_i)  s.t.  Σ μ_i = 1, μ >= 0
    Returns the closest valid probability vector.
    """
    n  = len(theta)
    mu = cp.Variable(n)
    objective = cp.Minimize(cp.sum(cp.kl_div(mu, theta)))
    constraints = [cp.sum(mu) == 1, mu >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    return mu.value


def find_mispriced_outcomes(market: MultiOutcomeMarket,
                             min_gap: float = 0.03) -> pd.DataFrame:
    """
    Compare market prices to Bregman-projected 'fair' prices.
    Positive gap → market underprices this outcome → BUY.
    Negative gap → overpriced → AVOID or hedge.
    """

    theta     = np.array(market.prices, dtype=float)
    projected = bregman_project(theta)

    rows = []
    for outcome, mkt_p, fair_p in zip(market.outcomes, theta, projected):
        gap = fair_p - mkt_p
        rows.append({
            "outcome":     outcome,
            "market_p":   mkt_p,
            "fair_p":     round(fair_p, 4),
            "gap":        round(gap, 4),
            "action":     "BUY" if gap > min_gap else ("AVOID" if gap < -min_gap else "HOLD"),
        })

    return pd.DataFrame(rows).sort_values("gap", ascending=False)


def plot_projection(market: MultiOutcomeMarket):
    theta     = np.array(market.prices)
    projected = bregman_project(theta)

    x     = np.arange(len(market.outcomes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, theta,     width, label="Market Prices",    color="steelblue", alpha=0.8)
    ax.bar(x + width/2, projected, width, label="Bregman Projected", color="orange",   alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(market.outcomes, rotation=30, ha="right")
    ax.set_ylabel("Implied Probability")
    ax.set_title(f"Bregman Projection: {market.name}")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("bregman_projection.png", dpi=150)
    plt.show()


# ── Demo: Oscars Best Picture 2026 ─────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    oscars = MultiOutcomeMarket(
        name="Oscars Best Picture 2026",
        outcomes=["Sinners", "Conclave", "The Brutalist", "Emilia Perez",
                  "Wicked", "Nickel Boys", "A Complete Unknown", "Other"],
        prices  =[0.15,     0.18,       0.22,            0.12,
                  0.10,     0.08,       0.11,             0.09],
        volume=21_000_000,
    )

    incon = market_inconsistency(oscars.prices)
    print(f"Oscars Market Check:")
    print(f"  Sum of prices : {incon['total']:.4f}")
    print(f"  Over-round    : {incon['vig_pct']:.2f}%")
    print(f"  Pure arb      : {incon['arb_exists']}\n")

    df = find_mispriced_outcomes(oscars)
    print(df.to_string(index=False))
    plot_projection(oscars)
