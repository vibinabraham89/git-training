"""
Polymarket CLOB Client — fetches live market data.
No API key needed for read-only access.

Endpoints used:
  GET /markets          → list all active markets
  GET /markets/{id}     → single market details (price, volume, b parameter)
  GET /book?token_id=X  → live orderbook (best bid/ask)
"""

import requests
import time
from dataclasses import dataclass
from typing import Optional

CLOB_BASE = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"   # richer metadata


@dataclass
class LiveMarket:
    condition_id:  str
    question:      str
    yes_price:     float   # current YES implied prob
    no_price:      float
    volume_usd:    float
    active:        bool
    category:      str
    end_date:      str


def _get(url: str, params: dict = None, retries: int = 3) -> dict | list | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                print(f"[CLOB] Failed after {retries} attempts: {e}")
                return None
            time.sleep(1.5 ** attempt)


def fetch_markets(limit: int = 100,
                  active_only: bool = True) -> list[LiveMarket]:
    """
    Fetch markets from Gamma API (richer data than raw CLOB).
    Returns list of LiveMarket objects.
    """
    params = {
        "limit":    limit,
        "active":   str(active_only).lower(),
        "closed":   "false",
        "order":    "volume24hr",
        "ascending": "false",
    }
    data = _get(f"{GAMMA_BASE}/markets", params=params)
    if not data:
        return []

    markets = []
    for m in data:
        try:
            outcomes = m.get("outcomePrices", "[]")
            # outcomePrices is a JSON string like '["0.65", "0.35"]'
            import json
            prices = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
            yes_p  = float(prices[0]) if prices else 0.5
            no_p   = float(prices[1]) if len(prices) > 1 else 1 - yes_p

            markets.append(LiveMarket(
                condition_id = m.get("conditionId", ""),
                question     = m.get("question", ""),
                yes_price    = yes_p,
                no_price     = no_p,
                volume_usd   = float(m.get("volume", 0)),
                active       = m.get("active", True),
                category     = m.get("groupItemTitle", ""),
                end_date     = m.get("endDate", ""),
            ))
        except (ValueError, IndexError, KeyError):
            continue

    return markets


def fetch_orderbook(token_id: str) -> dict | None:
    """
    Fetch live orderbook for a specific outcome token.
    Returns best bid, best ask, and spread.
    """
    data = _get(f"{CLOB_BASE}/book", params={"token_id": token_id})
    if not data:
        return None

    bids = data.get("bids", [])
    asks = data.get("asks", [])

    best_bid = float(bids[0]["price"]) if bids else None
    best_ask = float(asks[0]["price"]) if asks else None
    spread   = (best_ask - best_bid) if (best_bid and best_ask) else None

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread":   spread,
        "mid":      (best_bid + best_ask) / 2 if spread else None,
    }


def fetch_single_market(condition_id: str) -> dict | None:
    """Fetch a single market by condition ID."""
    return _get(f"{GAMMA_BASE}/markets/{condition_id}")


def filter_by_category(markets: list[LiveMarket],
                        keywords: list[str]) -> list[LiveMarket]:
    """Filter markets by keyword in question text."""
    kw = [k.lower() for k in keywords]
    return [m for m in markets
            if any(k in m.question.lower() for k in kw)]


def filter_by_volume(markets: list[LiveMarket],
                     min_vol: float = 50_000) -> list[LiveMarket]:
    return [m for m in markets if m.volume_usd >= min_vol]


# ── Quick test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching top 20 active Polymarket markets by volume...\n")
    markets = fetch_markets(limit=20)

    if not markets:
        print("No data returned. Check your internet connection.")
    else:
        print(f"{'Question':<55} {'YES':>6}  {'Volume':>12}")
        print("─" * 76)
        for m in markets:
            q = m.question[:52] + "..." if len(m.question) > 52 else m.question
            print(f"{q:<55} {m.yes_price:>6.1%}  ${m.volume_usd:>11,.0f}")
