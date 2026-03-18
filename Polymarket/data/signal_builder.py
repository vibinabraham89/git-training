"""
Signal Builder — combines free news sources + Binance into model_p.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Market type → signal sources:

  CRYPTO       → Binance log-normal (hard data) + Google News + Reddit crypto
  POLITICS     → Google News + Reddit politics
  GEOPOLITICS  → Google News + Reddit worldnews
  ENTERTAINMENT→ Google News
  SPORTS       → Google News + Reddit sports

Twitter/X removed: free tier no longer supports search (402 error).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.polymarket_client import LiveMarket
from data.news_client import (
    build_evidence_from_google_news,
    build_evidence_from_reddit,
    build_evidence_from_cryptopanic,
)
from strategy.bayesian import BayesianModel, Evidence, edge_vs_market


# Query templates per market keyword
QUERY_TEMPLATES = {
    "btc":       ("bitcoin BTC price prediction",       "crypto"),
    "bitcoin":   ("bitcoin BTC price prediction",       "crypto"),
    "eth":       ("ethereum ETH price prediction",      "crypto"),
    "ethereum":  ("ethereum ETH price prediction",      "crypto"),
    "sol":       ("solana SOL price prediction",        "crypto"),
    "iran":      ("Iran ceasefire nuclear deal 2026",   "geopolitics"),
    "russia":    ("Russia Ukraine ceasefire peace",     "geopolitics"),
    "ceasefire": ("ceasefire peace deal negotiations",  "geopolitics"),
    "ukraine":   ("Ukraine Russia war ceasefire",       "geopolitics"),
    "fed":       ("Federal Reserve interest rate cut",  "geopolitics"),
    "rate":      ("Federal Reserve interest rate",      "geopolitics"),
    "vance":     ("JD Vance 2028 president election",   "politics"),
    "newsom":    ("Gavin Newsom 2028 president",        "politics"),
    "trump":     ("Trump 2028 president election",      "politics"),
    "harris":    ("Kamala Harris 2028 president",       "politics"),
    "senate":    ("US Senate election 2026",            "politics"),
    "oscar":     ("Oscars best picture 2026 winner",    "entertainment"),
    "nba":       ("NBA champion 2026 playoffs",         "sports"),
    "nfl":       ("NFL Super Bowl 2027",                "sports"),
    "barcelona": ("FC Barcelona match result",          "sports"),
    "bitcoin reach": ("bitcoin price target 2026",      "crypto"),
}


def build_news_query(question: str) -> tuple[str, str] | tuple[None, None]:
    """Match market question → (search query, category)."""
    q = question.lower()
    for keyword, (template, category) in QUERY_TEMPLATES.items():
        if keyword in q:
            return template, category
    return None, None


def compute_bayesian_model_p(market:   LiveMarket,
                              hours_back: int  = 6,
                              verbose:    bool = True) -> float:
    """
    Compute model_p using:
      1. Market price as prior
      2. Google News RSS sentiment (always free, no key)
      3. Reddit sentiment (no key)
      4. Binance log-normal (crypto only)
      5. CryptoPanic (crypto, if CRYPTOPANIC_KEY set)
    """
    # Prior: market price with moderate confidence
    strength = 4.0
    alpha    = market.yes_price * strength
    beta     = (1 - market.yes_price) * strength
    model    = BayesianModel(alpha=alpha, beta=beta)

    query, category = build_news_query(market.question)

    # ── Google News (always runs — zero keys needed) ───────────────────────────
    if query:
        try:
            e = build_evidence_from_google_news(query, verbose=verbose)
            model.update(e)
        except Exception as ex:
            if verbose:
                print(f"[Signal] Google News failed: {ex}")

    # ── Reddit (no key, rate limit 1 req/sec) ─────────────────────────────────
    if query and category:
        try:
            e = build_evidence_from_reddit(query, category=category,
                                            verbose=verbose)
            model.update(e)
        except Exception as ex:
            if verbose:
                print(f"[Signal] Reddit failed: {ex}")

    # ── Binance log-normal (crypto price markets) ──────────────────────────────
    q_lower = market.question.lower()
    if any(k in q_lower for k in ["btc", "bitcoin", "eth", "ethereum", "sol"]):
        try:
            import re, contextlib, io
            from data.crypto_client import estimate_model_p

            q_clean = market.question.replace(",", "")
            match   = re.search(r'\$(\d+(?:\.\d+)?)[Kk]?', q_clean)
            if match:
                val = float(match.group(1))
                if 'k' in q_clean[match.start():match.end()].lower():
                    val *= 1000
                if 10_000 < val < 1_000_000:
                    coin = ("ETH" if "eth" in q_lower or "ethereum" in q_lower
                            else "SOL" if "sol" in q_lower else "BTC")
                    with contextlib.redirect_stdout(io.StringIO()):
                        p = estimate_model_p(coin, val, days_forward=14,
                                             verbose=False)
                    e = Evidence(
                        source = f"Binance log-normal ({coin}>${val:,.0f})",
                        value  = p,
                        weight = 2.0,
                    )
                    model.update(e)
                    if verbose:
                        print(f"[Binance] {coin} log-normal p={p:.1%}  (weight=2.0)")
        except Exception as ex:
            if verbose:
                print(f"[Signal] Binance failed: {ex}")

    # ── CryptoPanic (optional, if key set) ────────────────────────────────────
    if os.environ.get("CRYPTOPANIC_KEY") and any(
            k in q_lower for k in ["btc", "bitcoin", "eth", "ethereum"]):
        try:
            coin = "ETH" if "eth" in q_lower else "BTC"
            e    = build_evidence_from_cryptopanic(coin, verbose=verbose)
            if e:
                model.update(e)
        except Exception as ex:
            if verbose:
                print(f"[Signal] CryptoPanic failed: {ex}")

    posterior = model.posterior_mean
    if verbose:
        result = edge_vs_market(model, market.yes_price)
        print(f"[Bayesian] Prior={market.yes_price:.1%} → "
              f"Posterior={posterior:.1%}  Edge={result['edge']:+.1%}  "
              f"{result['signal']}")

    return posterior


def build_signals_for_markets(markets:    list[LiveMarket],
                               verbose:    bool = False) -> dict[str, float]:
    """Compute model_p for a list of markets. Returns {question: model_p}."""
    import time
    results = {}
    for i, m in enumerate(markets):
        if verbose:
            print(f"\n[{i+1}/{len(markets)}] {m.question[:70]}")
        results[m.question] = compute_bayesian_model_p(
            m, verbose=verbose
        )
        time.sleep(1)   # Reddit rate limit
    return results


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data.news_client import load_env_file
    load_env_file()

    from data.polymarket_client import LiveMarket
    test_markets = [
        LiveMarket("id1", "Will Iran reach a ceasefire by June 2026?",
                   0.03, 0.97, 38_000_000, True, "geopolitics", "2026-06-30"),
        LiveMarket("id2", "Will the Fed decrease interest rates by 25 bps?",
                   0.001, 0.999, 173_000_000, True, "macro", "2026-03-20"),
        LiveMarket("id3", "Will Bitcoin reach $150000 in March?",
                   0.002, 0.998, 17_000_000, True, "crypto", "2026-03-31"),
    ]

    for m in test_markets:
        print(f"\n{'═'*60}")
        print(f"Market : {m.question}")
        print(f"Price  : {m.yes_price:.1%}  |  Vol: ${m.volume_usd:,.0f}")
        p = compute_bayesian_model_p(m, verbose=True)
        print(f"model_p: {p:.1%}")
