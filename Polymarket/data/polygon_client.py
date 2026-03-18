"""
Polygon.io Client — fetches real-time crypto prices for model_p estimation.
Used by the EV scanner to compute model_p for crypto markets (BTC, ETH).

Free tier: 5 calls/min, delayed 15 min
Starter+:  real-time, unlimited

Set your key in .env:
  POLYGON_API_KEY=your_key_here
"""

import os
import requests
import time
from datetime import datetime, timedelta
from typing import Optional
import json

POLYGON_BASE = "https://api.polygon.io"


def get_api_key() -> str:
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "POLYGON_API_KEY not set.\n"
            "  1. Get a free key at polygon.io\n"
            "  2. Run: export POLYGON_API_KEY=your_key\n"
            "  OR create a .env file with: POLYGON_API_KEY=your_key"
        )
    return key


def _get(endpoint: str, params: dict = None) -> dict | None:
    key    = get_api_key()
    params = params or {}
    params["apiKey"] = key
    try:
        r = requests.get(f"{POLYGON_BASE}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"[Polygon] Error: {e}")
        return None


def get_crypto_price(ticker: str) -> Optional[float]:
    """
    Get latest crypto price.
    ticker examples: X:BTCUSD, X:ETHUSD
    Returns price in USD.
    """
    data = _get(f"/v2/last/trade/{ticker}")
    if not data or data.get("status") != "OK":
        return None
    return data["result"]["p"]   # price field


def get_daily_bars(ticker: str, days: int = 30) -> list[dict]:
    """
    Fetch daily OHLCV bars for the past N days.
    Useful for volatility estimation → improves model_p.
    """
    end   = datetime.now()
    start = end - timedelta(days=days)
    data  = _get(
        f"/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}",
        {"adjusted": "true", "sort": "asc", "limit": 50}
    )
    if not data or data.get("status") != "OK":
        return []
    return data.get("results", [])


def estimate_btc_model_p(target_price: float,
                          days_forward: int = 14) -> float:
    """
    Naive model_p for "BTC above TARGET by DATE".
    Uses: current price + historical daily vol → log-normal probability.

    Replace with a proper regression model for production.
    """
    import math
    bars = get_daily_bars("X:BTCUSD", days=60)
    if not bars or len(bars) < 5:
        print("[Polygon] Not enough bar data. Using fallback.")
        return 0.50

    closes = [b["c"] for b in bars]
    current_price = closes[-1]

    # Log-normal model: P(S_T > K) = N(d2)
    log_returns = [math.log(closes[i+1] / closes[i]) for i in range(len(closes)-1)]
    mu_daily  = sum(log_returns) / len(log_returns)
    var_daily = sum((r - mu_daily)**2 for r in log_returns) / len(log_returns)
    sigma_daily = var_daily ** 0.5

    # Drift-adjusted
    sigma_T = sigma_daily * (days_forward ** 0.5)
    mu_T    = (mu_daily - 0.5 * var_daily) * days_forward
    log_ratio = math.log(target_price / current_price)

    from scipy.stats import norm
    d2 = (math.log(current_price / target_price) + mu_T) / sigma_T
    prob = norm.cdf(d2)

    print(f"[Polygon] BTC current: ${current_price:,.0f}  |  Target: ${target_price:,.0f}")
    print(f"[Polygon] σ_daily: {sigma_daily:.2%}  |  Horizon: {days_forward}d")
    print(f"[Polygon] Model P(BTC > ${target_price:,.0f}): {prob:.1%}")
    return float(prob)


def load_env_file(path: str = ".env"):
    """Load a .env file into os.environ (no dotenv library needed)."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass


# ── Quick test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_env_file()

    if not os.environ.get("POLYGON_API_KEY"):
        print("No POLYGON_API_KEY found.")
        print("Create a .env file with:  POLYGON_API_KEY=your_key_here")
    else:
        print("Testing Polygon.io connection...\n")
        price = get_crypto_price("X:BTCUSD")
        if price:
            print(f"BTC/USD live price: ${price:,.2f}")
            p = estimate_btc_model_p(target_price=100_000, days_forward=14)
            print(f"\nModel p(BTC >$100K in 14 days): {p:.1%}")
        else:
            print("Could not fetch price. Check your API key.")
