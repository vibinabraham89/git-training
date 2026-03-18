"""
Crypto Price Client — Binance Public API (no API key needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Replaces Polygon.io. No signup, no key, 1200 calls/min free.

Tickers: BTCUSDT, ETHUSDT, SOLUSDT (Binance symbol format)
"""

import math
import requests
import time
from typing import Optional
from scipy.stats import norm


BINANCE_BASE = "https://api.binance.com/api/v3"

SYMBOL_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
}


def _get(endpoint: str, params: dict = None) -> dict | list | None:
    try:
        r = requests.get(f"{BINANCE_BASE}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"[Binance] Error: {e}")
        return None


def get_price(coin: str = "BTC") -> Optional[float]:
    """
    Get current price for a coin.
    coin: "BTC", "ETH", "SOL"
    """
    symbol = SYMBOL_MAP.get(coin.upper(), f"{coin.upper()}USDT")
    data   = _get("/ticker/price", {"symbol": symbol})
    if not data:
        return None
    return float(data["price"])


def get_daily_closes(coin: str = "BTC", days: int = 60) -> list[float]:
    """
    Fetch last N daily closing prices.
    Uses Binance klines (candlestick) endpoint.
    """
    symbol = SYMBOL_MAP.get(coin.upper(), f"{coin.upper()}USDT")
    data   = _get("/klines", {
        "symbol":   symbol,
        "interval": "1d",
        "limit":    days + 1,
    })
    if not data:
        return []
    # kline format: [open_time, open, high, low, close, ...]
    return [float(k[4]) for k in data]


def log_normal_model_p(current_price: float,
                        target_price:  float,
                        closes:        list[float],
                        days_forward:  int = 14) -> float:
    """
    P(price > target in N days) via log-normal model.

    Uses historical daily log-returns to estimate:
      μ  = mean daily log-return
      σ  = daily volatility
    Then: P(S_T > K) = N(d2)
    """
    if len(closes) < 5:
        return 0.5

    log_returns  = [math.log(closes[i+1] / closes[i])
                    for i in range(len(closes) - 1)]
    mu_daily     = sum(log_returns) / len(log_returns)
    var_daily    = sum((r - mu_daily)**2 for r in log_returns) / len(log_returns)
    sigma_daily  = var_daily ** 0.5

    sigma_T = sigma_daily * math.sqrt(days_forward)
    mu_T    = (mu_daily - 0.5 * var_daily) * days_forward

    if sigma_T == 0:
        return 1.0 if current_price > target_price else 0.0

    d2   = (math.log(current_price / target_price) + mu_T) / sigma_T
    prob = norm.cdf(d2)
    return float(prob)


def estimate_model_p(coin: str,
                      target_price: float,
                      days_forward: int = 14,
                      verbose:      bool = True) -> float:
    """
    Full pipeline: fetch price + history → compute model_p.
    """
    current = get_price(coin)
    if not current:
        print(f"[Binance] Could not fetch {coin} price.")
        return 0.5

    closes = get_daily_closes(coin, days=60)
    if not closes:
        print(f"[Binance] Could not fetch {coin} history.")
        return 0.5

    p = log_normal_model_p(current, target_price, closes, days_forward)

    if verbose:
        print(f"[Binance] {coin} current: ${current:,.0f}  |  Target: ${target_price:,.0f}")
        print(f"[Binance] Horizon: {days_forward}d  |  Model P({coin} > ${target_price:,.0f}): {p:.1%}")

    return p


def load_env_file(path: str = ".env"):
    """Load .env file into os.environ (no dotenv needed)."""
    import os
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
    print("Testing Binance public API (no key needed)...\n")

    for coin in ["BTC", "ETH", "SOL"]:
        price = get_price(coin)
        print(f"  {coin}/USDT: ${price:,.2f}" if price else f"  {coin}: fetch failed")

    print()
    estimate_model_p("BTC", target_price=150_000, days_forward=13)
    print()
    estimate_model_p("ETH", target_price=5_000,   days_forward=13)
