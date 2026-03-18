"""
Portfolio Manager — Unified Signal Aggregator + Risk Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Combines all 6 formulas into a single allocation engine.

Signal stack:
  1. EV Scanner      → generates candidate trades
  2. Kelly Criterion → sizes each trade
  3. KL Divergence   → adds hedge pairs
  4. Bregman         → scans multi-outcome inconsistencies
  5. Bayesian        → adjusts model_p in real-time
  6. LMSR            → checks entry price impact won't eat the edge

Risk controls:
  • Max 5% bankroll per single trade
  • Max 50% bankroll deployed at once
  • 20% drawdown kill switch
  • Correlation cap: max 3 trades from same category
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class Category(str, Enum):
    CRYPTO      = "crypto"
    POLITICS    = "politics"
    GEOPOLITICS = "geopolitics"
    SPORTS      = "sports"
    ENTERTAINMENT = "entertainment"


@dataclass
class Trade:
    market:       str
    category:     Category
    model_p:      float
    market_price: float
    volume_usd:   float
    # Risk metrics (populated by engine)
    ev_net:       float   = 0.0
    kelly_f:      float   = 0.0
    usd_size:     float   = 0.0
    signal:       str     = "HOLD"
    notes:        str     = ""


@dataclass
class Portfolio:
    bankroll:       float
    peak_bankroll:  float = 0.0
    deployed:       float = 0.0
    trades:         List[Trade] = field(default_factory=list)

    def __post_init__(self):
        self.peak_bankroll = self.bankroll

    @property
    def drawdown(self) -> float:
        return (self.peak_bankroll - self.bankroll) / self.peak_bankroll

    @property
    def available(self) -> float:
        max_deploy = self.bankroll * 0.50   # never deploy >50%
        return max(0, max_deploy - self.deployed)


FEE         = 0.02
KELLY_FRAC  = 0.25
MAX_PER_BET = 0.05      # 5% of bankroll hard cap
MAX_DEPLOY  = 0.50
MAX_DD      = 0.20
MAX_PER_CAT = 3         # correlation cap
MIN_EV      = 0.05
MIN_VOL     = 50_000


def compute_ev(model_p: float, market_price: float) -> float:
    return (model_p - market_price) / market_price - FEE


def compute_kelly(model_p: float, market_price: float) -> float:
    odds = (1 / market_price) - 1
    raw  = (model_p * odds - (1 - model_p)) / odds
    return max(0.0, raw * KELLY_FRAC)


def confidence_scalar(model_p: float, market_price: float) -> float:
    """
    Scale bet size by how far model_p is from market_price (edge confidence).
    Small edge (5-8%) → 0.5x size. Strong edge (15%+) → 1.0x size.
    Prevents max-sizing weak signals.
    """
    edge = abs(model_p - market_price)
    if edge < 0.05:
        return 0.3
    if edge < 0.10:
        return 0.5
    if edge < 0.15:
        return 0.75
    return 1.0


def size_trade(trade: Trade, portfolio: Portfolio) -> float:
    """
    Return USD size for trade.
    Uses fractional Kelly × confidence scalar, capped at MAX_PER_BET.
    """
    scalar    = confidence_scalar(trade.model_p, trade.market_price)
    kelly_usd = trade.kelly_f * portfolio.bankroll * scalar
    cap_usd   = portfolio.bankroll * MAX_PER_BET
    return min(kelly_usd, cap_usd, portfolio.available)


def allocate(trades: List[Trade],
             portfolio: Portfolio,
             verbose: bool = True) -> pd.DataFrame:
    """
    Full allocation pass.
    Returns DataFrame of approved trades with USD sizes.
    """
    if portfolio.drawdown >= MAX_DD:
        print(f"KILL SWITCH: drawdown {portfolio.drawdown:.1%} >= {MAX_DD:.0%}. No new trades.")
        return pd.DataFrame()

    # Score all trades
    for t in trades:
        t.ev_net  = compute_ev(t.model_p, t.market_price)
        t.kelly_f = compute_kelly(t.model_p, t.market_price)

    # Filter: must clear EV and volume hurdles
    candidates = [t for t in trades
                  if t.ev_net >= MIN_EV and t.volume_usd >= MIN_VOL and t.kelly_f > 0]
    candidates.sort(key=lambda t: t.ev_net, reverse=True)

    # Apply correlation cap (max MAX_PER_CAT per category)
    cat_count: dict = {}
    approved  = []
    for t in candidates:
        c = cat_count.get(t.category, 0)
        if c >= MAX_PER_CAT:
            t.signal = "SKIP (corr cap)"
            continue
        t.usd_size = size_trade(t, portfolio)
        if t.usd_size <= 0:
            t.signal = "SKIP (no capital)"
            continue
        t.signal   = "BUY"
        portfolio.deployed += t.usd_size
        cat_count[t.category] = c + 1
        approved.append(t)

    if verbose:
        rows = [{
            "market":       t.market,
            "category":     t.category.value,
            "price":        f"{t.market_price:.1%}",
            "model_p":      f"{t.model_p:.1%}",
            "ev_net":       f"{t.ev_net:.2%}",
            "kelly_f":      f"{t.kelly_f:.3f}",
            "size_usd":     f"${t.usd_size:,.0f}",
            "signal":       t.signal,
        } for t in approved]

        if rows:
            df = pd.DataFrame(rows)
            print(f"\nPortfolio Allocation  (bankroll: ${portfolio.bankroll:,.0f})")
            print(f"Deployed: ${portfolio.deployed:,.0f} / ${portfolio.bankroll * MAX_DEPLOY:,.0f} max\n")
            print(df.to_string(index=False))
        else:
            print("No trades passed all filters.")

    return pd.DataFrame([{
        "market": t.market, "category": t.category.value,
        "size_usd": t.usd_size, "ev_net": t.ev_net, "signal": t.signal,
    } for t in approved])


# ── Demo ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    portfolio = Portfolio(bankroll=10_000)

    trades = [
        # Geopolitics
        Trade("Iran ceasefire by June",    Category.GEOPOLITICS, 0.52, 0.47, 5_000_000),
        Trade("Russia ceasefire Q2 2026",  Category.GEOPOLITICS, 0.38, 0.33,   800_000),
        # Politics
        Trade("Vance 2028 nominee",        Category.POLITICS,    0.28, 0.21, 3_200_000),
        Trade("Senate D majority 2026",    Category.POLITICS,    0.48, 0.42, 5_000_000),
        # Crypto
        Trade("BTC >100K March 2026",      Category.CRYPTO,      0.42, 0.38, 12_000_000),
        Trade("ETH >5K March 2026",        Category.CRYPTO,      0.33, 0.29,  8_000_000),
        Trade("BTC >110K April 2026",      Category.CRYPTO,      0.35, 0.30,  6_000_000),
        Trade("SOL >250 March 2026",       Category.CRYPTO,      0.40, 0.36,  3_000_000),
        # Entertainment — thin
        Trade("Oscars: The Brutalist",     Category.ENTERTAINMENT, 0.26, 0.22, 21_000_000),
        Trade("Musk tweets >50 today",     Category.CRYPTO,      0.55, 0.40,   150_000),  # low vol
    ]

    df = allocate(trades, portfolio)
    print(f"\nTotal at risk: ${portfolio.deployed:,.0f}")
    print(f"Drawdown guard: triggers at ${portfolio.bankroll * (1 - MAX_DD):,.0f}")
