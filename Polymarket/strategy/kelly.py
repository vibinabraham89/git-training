"""
Formula 2: Kelly Criterion + Growth Simulation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best for: Political markets (2028 race), long-duration macro bets
Edge:     Fractional Kelly (0.25×) maximises geometric growth, avoids ruin.
          Outperforms fixed-sizing on 100+ bet sequences.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def kelly_fraction(p: float, market_price: float) -> float:
    """
    f* = (p × odds − (1−p)) / odds
    odds = decimal odds − 1  = (1/price) − 1
    """
    if market_price <= 0 or market_price >= 1:
        raise ValueError("market_price must be in (0,1)")
    odds = (1 / market_price) - 1
    return (p * odds - (1 - p)) / odds


def fractional_kelly(p: float, market_price: float,
                     fraction: float = 0.25) -> float:
    """Quarter-Kelly — standard HF practice for drawdown control."""
    raw = kelly_fraction(p, market_price)
    return max(0.0, raw * fraction)   # never bet negative (skip negative EV)


def simulate_growth(p: float, market_price: float,
                    fraction: float = 0.25,
                    n_bets: int = 200,
                    n_paths: int = 500,
                    starting_bankroll: float = 1_000) -> np.ndarray:
    """
    Monte-Carlo bankroll paths.
    Returns array shape (n_paths, n_bets+1).
    """
    f    = fractional_kelly(p, market_price, fraction)
    odds = (1 / market_price) - 1
    rolls  = np.random.rand(n_paths, n_bets) < p   # True = win
    gains  = np.where(rolls, 1 + f * odds, 1 - f)
    paths  = np.cumprod(gains, axis=1)
    paths  = np.hstack([np.ones((n_paths, 1)), paths])
    return paths * starting_bankroll


def kelly_sweep(p: float, market_price: float,
                n_bets: int = 100, n_paths: int = 200) -> dict:
    """
    Sweep Kelly fractions 0→1, return (fraction, median_terminal_bankroll).
    Identifies peak (full Kelly) and safe zone (quarter Kelly).
    """
    fractions = np.linspace(0, 1, 50)
    medians   = []
    for frac in fractions:
        paths   = simulate_growth(p, market_price, frac, n_bets, n_paths)
        medians.append(np.median(paths[:, -1]))
    return {"fractions": fractions, "medians": np.array(medians)}


def plot_kelly(p: float, market_price: float, starting_bankroll: float = 1_000):
    paths = simulate_growth(p, market_price, fraction=0.25,
                             n_bets=200, n_paths=300,
                             starting_bankroll=starting_bankroll)
    sweep = kelly_sweep(p, market_price)
    f_opt = fractional_kelly(p, market_price, fraction=1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Growth paths
    for path in paths[:50]:
        ax1.plot(path, alpha=0.08, color="steelblue")
    ax1.plot(np.median(paths, axis=0), color="orange", linewidth=2.5,
             label="Median path")
    ax1.set_yscale("log")
    ax1.set_xlabel("Bet #")
    ax1.set_ylabel("Bankroll (USD, log scale)")
    ax1.set_title(f"Quarter-Kelly Growth  (p={p}, price={market_price})")
    ax1.legend()

    # Kelly sweep
    ax2.plot(sweep["fractions"], sweep["medians"], color="steelblue", linewidth=2)
    ax2.axvline(f_opt, color="red",    linestyle="--", label=f"Full Kelly f*={f_opt:.2f}")
    ax2.axvline(f_opt * 0.25, color="green", linestyle="--",
                label=f"Quarter Kelly={f_opt*0.25:.2f}")
    ax2.set_xlabel("Kelly Fraction")
    ax2.set_ylabel("Median Terminal Bankroll")
    ax2.set_title("Kelly Fraction Sweep (peak = full Kelly)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("kelly_growth.png", dpi=150)
    plt.show()


# ── Demo: Vance 2028 ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p_model      = 0.28   # your poll/X-sentiment model
    market_price = 0.21   # current Polymarket odds

    f_raw  = kelly_fraction(p_model, market_price)
    f_safe = fractional_kelly(p_model, market_price, fraction=0.25)

    print(f"Market: Vance 2028 Winner")
    print(f"  Market price  : {market_price:.0%}")
    print(f"  Model p       : {p_model:.0%}")
    print(f"  Full Kelly f* : {f_raw:.2%}  of bankroll")
    print(f"  Quarter Kelly : {f_safe:.2%} of bankroll  ← USE THIS")
    print(f"  On $10,000    : ${f_safe * 10_000:,.0f} at risk")

    plot_kelly(p_model, market_price, starting_bankroll=10_000)
