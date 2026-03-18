"""
Master Runner — executes the full quant pipeline end-to-end.

Usage:
  python run_strategy.py

In production: schedule this via cron every 5-15 minutes.
Replace all placeholder data with live Polygon API calls.
"""

import sys
import numpy as np
from strategy import (
    LMSRMarket, scan_arb_window,
    fractional_kelly,
    MarketSignal, rank_opportunities,
    MarketPair, scan_pairs,
    MultiOutcomeMarket, find_mispriced_outcomes,
    BayesianModel, Evidence, edge_vs_market,
    Portfolio, Trade, Category, allocate,
    walk_forward_backtest, compute_metrics,
)
from strategy.backtest import generate_synthetic_history


BANKROLL = 10_000   # starting capital (USD)


def step1_lmsr_scan():
    print("\n" + "═"*60)
    print("  STEP 1 — LMSR: Impact Arb Scan (Crypto / Esports)")
    print("═"*60)
    # Low-b markets: crypto 5-min, esports
    markets = [
        ("BTC 5-min up/down", LMSRMarket(b=75,  quantities=[0.0, 0.0]), 0.0, 0.55),
        ("ETH 1-hr up/down",  LMSRMarket(b=120, quantities=[0.0, 0.0]), 0.0, 0.58),
        ("Esports CS2 match", LMSRMarket(b=40,  quantities=[0.0, 0.0]), 0.0, 0.62),
    ]
    for name, mkt, outcome_idx, true_p in markets:
        result = scan_arb_window(mkt, outcome_idx=0, true_prob=true_p)
        print(f"\n  {name} (b={mkt.b})")
        print(f"    Optimal shares : {result.get('shares', 0):.0f}")
        print(f"    Max EV ($)     : {result.get('ev', 0):.2f}")
        print(f"    Avg fill price : {result.get('avg_fill', 0):.3f}")


def step2_ev_scan():
    print("\n" + "═"*60)
    print("  STEP 2 — EV Gap Scanner (Geopolitics / Politics / Macro)")
    print("═"*60)
    signals = [
        MarketSignal("Iran ceasefire by June 2026",   0.47, 0.52, 5_000_000),
        MarketSignal("Russia ceasefire Q2 2026",      0.33, 0.38,   800_000),
        MarketSignal("Fed rate cut March 2026",        0.60, 0.72, 18_000_000),
        MarketSignal("Vance 2028 GOP nominee",         0.21, 0.28,  3_200_000),
        MarketSignal("BTC >100K end of March 2026",   0.38, 0.42, 12_000_000),
        MarketSignal("ETH >5K March 2026",            0.29, 0.33,  8_000_000),
    ]
    df = rank_opportunities(signals)
    print(df[["market", "price", "model_p", "ev_net", "enter"]].to_string(index=False))


def step3_kl_scan():
    print("\n" + "═"*60)
    print("  STEP 3 — KL Divergence: Correlated Pair Scanner")
    print("═"*60)
    pairs = [
        MarketPair("Vance 2028",   "Newsom 2028",   0.21, 0.17, 3_200_000, 2_800_000, mutex=True),
        MarketPair("BTC >100K",    "ETH >5K",       0.38, 0.29, 12_000_000, 8_000_000),
        MarketPair("Senate D",     "Senate R",      0.42, 0.55,  5_000_000, 5_200_000, mutex=True),
        MarketPair("Iran deal Q2", "Russia deal Q2",0.47, 0.33,  5_000_000, 2_000_000),
    ]
    df = scan_pairs(pairs)
    print(df[["pair", "symmetric_kl", "mutex_gap", "signal", "action"]].to_string(index=False))


def step4_bregman_scan():
    print("\n" + "═"*60)
    print("  STEP 4 — Bregman Projection: Multi-Outcome Arb (Oscars/Tournament)")
    print("═"*60)
    oscars = MultiOutcomeMarket(
        name="Oscars Best Picture 2026",
        outcomes=["Sinners","Conclave","The Brutalist","Emilia Perez",
                  "Wicked","Nickel Boys","A Complete Unknown","Other"],
        prices  =[0.15,    0.18,      0.22,           0.12,
                  0.10,    0.08,      0.11,            0.09],
        volume=21_000_000,
    )
    total = sum(oscars.prices)
    print(f"\n  {oscars.name}  |  Sum of prices: {total:.4f}")
    df = find_mispriced_outcomes(oscars)
    print(df[["outcome", "market_p", "fair_p", "gap", "action"]].to_string(index=False))


def step5_bayesian_update():
    print("\n" + "═"*60)
    print("  STEP 5 — Bayesian Update: Real-Time Signal Engine")
    print("═"*60)
    market_price = 0.40
    model = BayesianModel(alpha=2, beta=3)
    evidences = [
        Evidence("Morning tweet burst (15 by 9am)",    0.72, 1.2),
        Evidence("Historical avg on earnings days",    0.68, 1.0),
        Evidence("Current count at noon: 28 tweets",  0.78, 1.5),
    ]
    for ev in evidences:
        p = model.update(ev)
        print(f"  After '{ev.source}': posterior = {p:.1%}")
    result = edge_vs_market(model, market_price)
    print(f"\n  Final edge: {result['edge']:+.1%}  →  {result['signal']}")


def step6_allocate():
    print("\n" + "═"*60)
    print("  STEP 6 — Portfolio Allocation (Kelly + Risk Engine)")
    print("═"*60)
    portfolio = Portfolio(bankroll=BANKROLL)
    trades = [
        Trade("Iran ceasefire by June",   Category.GEOPOLITICS, 0.52, 0.47, 5_000_000),
        Trade("Fed rate cut March 2026",  Category.GEOPOLITICS, 0.72, 0.60, 18_000_000),
        Trade("Vance 2028 nominee",       Category.POLITICS,    0.28, 0.21,  3_200_000),
        Trade("Senate D majority",        Category.POLITICS,    0.48, 0.42,  5_000_000),
        Trade("BTC >100K March",          Category.CRYPTO,      0.42, 0.38, 12_000_000),
        Trade("ETH >5K March",            Category.CRYPTO,      0.33, 0.29,  8_000_000),
        Trade("Oscars: The Brutalist",    Category.ENTERTAINMENT,0.26, 0.22,21_000_000),
    ]
    allocate(trades, portfolio)


def step7_backtest():
    print("\n" + "═"*60)
    print("  STEP 7 — Walk-Forward Backtest (Validation)")
    print("═"*60)
    history = generate_synthetic_history(n=300, base_edge=0.07)
    df      = walk_forward_backtest(history, bankroll=BANKROLL, min_ev=0.05)
    metrics = compute_metrics(df, BANKROLL)
    for k, v in metrics.items():
        print(f"  {k:<25}: {v}")


if __name__ == "__main__":
    print("\n  POLYMARKET QUANT STRATEGY — Full Pipeline Run")
    print(f"  Bankroll: ${BANKROLL:,}  |  Date: 2026-03-18\n")

    step1_lmsr_scan()
    step2_ev_scan()
    step3_kl_scan()
    step4_bregman_scan()
    step5_bayesian_update()
    step6_allocate()
    step7_backtest()

    print("\n  Done. Review signals above and cross-check with live Polymarket data.")
