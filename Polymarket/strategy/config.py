"""
Configuration: API keys, thresholds, risk limits.
"""

POLYGON_API_KEY = "YOUR_POLYGON_API_KEY"   # https://polygon.io
CLOB_API_URL   = "https://clob.polymarket.com"

# ── Risk controls ──────────────────────────────────────────────────────────────
MAX_BANKROLL_FRACTION = 0.50   # never deploy more than 50% at once
KELLY_FRACTION        = 0.25   # quarter-Kelly for safety
MAX_DRAWDOWN_PCT      = 0.20   # kill switch at 20% drawdown
MIN_EV_THRESHOLD      = 0.05   # only enter trades with EV > 5%
KL_ARB_THRESHOLD      = 0.20   # KL divergence trigger
MIN_LIQUIDITY_B       = 30     # skip markets with b < 30 (whale risk)
MIN_DAILY_VOLUME_USD  = 50_000 # skip thin markets

# ── Target market categories ───────────────────────────────────────────────────
TARGET_CATEGORIES = [
    "crypto",        # BTC/ETH 5-min up/down — LMSR arb
    "politics",      # 2028 race, Senate — Kelly + KL
    "geopolitics",   # Iran, Russia/Ukraine — EV Gap + Bayesian
    "entertainment", # Oscars, Grammys — Bregman Projection
    "sports",        # Esports, NFL, NBA — LMSR + Kelly
]
