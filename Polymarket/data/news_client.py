"""
News Sentiment Client — free alternatives to Twitter/X API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sources (all free):
  1. Reddit      — r/worldnews, r/politics, r/CryptoCurrency  (free API, no key)
  2. NewsAPI     — breaking headlines  (free tier, 100 req/day, needs key)
  3. CryptoPanic — crypto news + built-in sentiment  (free tier, needs key)
  4. Google News RSS — completely free, no key ever

Each source returns Evidence objects for BayesianModel.update()
"""

import os
import re
import time
import requests
from dataclasses import dataclass

# ── load .env ──────────────────────────────────────────────────────────────────
def load_env_file(path: str = ".env"):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — Reddit (no key needed)
# ══════════════════════════════════════════════════════════════════════════════

# Reddit requires a descriptive User-Agent or it throttles cloud IPs aggressively
REDDIT_HEADERS = {
    "User-Agent": "python:polymarket-quant-bot:v1.0 (research only)"
}

SUBREDDIT_MAP = {
    "crypto":       ["CryptoCurrency", "Bitcoin", "ethereum"],
    "politics":     ["politics", "PoliticalDiscussion"],
    "geopolitics":  ["worldnews", "geopolitics"],
    "entertainment":["movies", "oscarrace"],
    "sports":       ["nba", "nfl", "soccer"],
}


def fetch_reddit_posts(subreddit: str,
                        limit: int = 25,
                        sort: str = "hot") -> list[dict]:
    """Fetch top posts from a subreddit (JSON API, no key needed)."""
    url  = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
    try:
        r = requests.get(url, headers=REDDIT_HEADERS,
                         params={"limit": limit}, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("data", {}).get("children", [])
    except Exception as e:
        print(f"[Reddit] {subreddit}: {e}")
        return []


def search_reddit(query: str, subreddit: str = "all",
                   limit: int = 25, retries: int = 2) -> list[dict]:
    """Search Reddit with retry + backoff. Returns [] if rate limited."""
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=REDDIT_HEADERS,
                             params={"q": query, "limit": limit,
                                     "sort": "new", "t": "week"},
                             timeout=10)
            if r.status_code == 429:
                wait = 2 ** (attempt + 1)   # 2s, 4s
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json().get("data", {}).get("children", [])
        except Exception as e:
            print(f"[Reddit] Search '{query}': {e}")
            return []
    print(f"[Reddit] Skipping '{query}' — still rate limited after {retries} retries")
    return []


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — NewsAPI (free, 100 req/day, needs key)
# ══════════════════════════════════════════════════════════════════════════════

NEWSAPI_BASE = "https://newsapi.org/v2"


def fetch_newsapi_headlines(query: str, max_results: int = 20) -> list[dict]:
    """
    Fetch top headlines from NewsAPI.
    Get free key at newsapi.org (100 req/day free).
    Set NEWSAPI_KEY in .env
    """
    key = os.environ.get("NEWSAPI_KEY", "")
    if not key:
        return []
    try:
        r = requests.get(
            f"{NEWSAPI_BASE}/everything",
            params={"q": query, "pageSize": max_results,
                    "sortBy": "publishedAt", "apiKey": key},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("articles", [])
    except Exception as e:
        print(f"[NewsAPI] {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — CryptoPanic (free tier, needs key)
# ══════════════════════════════════════════════════════════════════════════════

CRYPTOPANIC_BASE = "https://cryptopanic.com/api/v1"


def fetch_cryptopanic(currencies: str = "BTC,ETH",
                       filter_: str = "hot") -> list[dict]:
    """
    Fetch crypto news with sentiment from CryptoPanic.
    Get free key at cryptopanic.com/developers/api
    Set CRYPTOPANIC_KEY in .env
    filter_: "hot" | "bullish" | "bearish" | "important"
    """
    key = os.environ.get("CRYPTOPANIC_KEY", "")
    if not key:
        return []
    try:
        r = requests.get(
            f"{CRYPTOPANIC_BASE}/posts/",
            params={"auth_token": key, "currencies": currencies,
                    "filter": filter_, "public": "true"},
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        print(f"[CryptoPanic] {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 4 — Google News RSS (completely free, zero keys)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_google_news_rss(query: str, max_items: int = 20) -> list[str]:
    """
    Fetch Google News RSS headlines for a query.
    No key, no rate limit, completely free.
    Returns list of headline strings.
    """
    import urllib.parse
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    try:
        r = requests.get(url, timeout=10,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        # Parse titles from RSS XML (no lxml needed)
        # Google News uses plain <title> tags (no CDATA)
        titles = re.findall(r"<title>(.*?)</title>", r.text)
        # First 2 are feed-level titles, skip them
        return [t for t in titles[2:max_items+2] if t]
    except Exception as e:
        print(f"[GoogleNews] {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED SCORER — works across all sources
# ══════════════════════════════════════════════════════════════════════════════

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.sentiment import score_tweet_text, BULLISH_WORDS, BEARISH_WORDS


def score_texts(texts: list[str]) -> float:
    """Average sentiment score across a list of text strings."""
    if not texts:
        return 0.5
    scores = [score_tweet_text(t) for t in texts]
    return sum(scores) / len(scores)


def build_evidence_from_reddit(query: str,
                                 category: str = "geopolitics",
                                 verbose: bool = True):
    """
    Search Reddit → score posts → return Evidence.
    Falls back to Google News if Reddit is rate limited.
    """
    from strategy.bayesian import Evidence

    subreddits = SUBREDDIT_MAP.get(category, ["worldnews"])
    all_titles = []

    for sub in subreddits[:1]:   # max 1 subreddit on Railway to avoid 429s
        posts = search_reddit(query, subreddit=sub, limit=15)
        titles = [p["data"].get("title", "") for p in posts]
        all_titles.extend(titles)
        time.sleep(2)   # be polite — 2s gap between requests

    # Fall back to Google News if Reddit returned nothing
    if not all_titles:
        if verbose:
            print(f"[Reddit] No data for '{query}' — falling back to Google News")
        return build_evidence_from_google_news(query, verbose=verbose)

    score = score_texts(all_titles)

    if verbose:
        direction = "BULLISH" if score > 0.55 else "BEARISH" if score < 0.45 else "NEUTRAL"
        print(f"[Reddit] '{query}' — {len(all_titles)} posts — "
              f"score: {score:.2f} ({direction})")

    import math
    weight = min(1.5, 0.5 + math.log1p(len(all_titles)) / 6)
    return Evidence(source=f"Reddit:{query[:30]}", value=score, weight=weight)


def build_evidence_from_google_news(query: str, verbose: bool = True):
    """Google News RSS → Evidence. Zero keys, always available."""
    from strategy.bayesian import Evidence

    headlines = fetch_google_news_rss(query, max_items=20)
    score     = score_texts(headlines)

    if verbose:
        direction = "BULLISH" if score > 0.55 else "BEARISH" if score < 0.45 else "NEUTRAL"
        print(f"[GoogleNews] '{query}' — {len(headlines)} headlines — "
              f"score: {score:.2f} ({direction})")
        if headlines:
            print(f"  Top: {headlines[0][:80]}")

    import math
    weight = min(1.2, 0.4 + math.log1p(len(headlines)) / 6)
    return Evidence(source=f"GNews:{query[:30]}", value=score, weight=weight)


def build_evidence_from_cryptopanic(coin: str = "BTC", verbose: bool = True):
    """CryptoPanic sentiment → Evidence. Needs free API key."""
    from strategy.bayesian import Evidence

    posts = fetch_cryptopanic(currencies=coin)
    if not posts:
        return None

    # CryptoPanic has explicit votes: bullish/bearish counts
    bull, bear = 0, 0
    for p in posts:
        votes = p.get("votes", {})
        bull += votes.get("positive", 0)
        bear += votes.get("negative", 0)

    total = bull + bear
    score = (bull / total) if total > 0 else 0.5

    if verbose:
        print(f"[CryptoPanic] {coin} — {len(posts)} posts — "
              f"bull:{bull} bear:{bear} — score:{score:.2f}")

    return Evidence(source=f"CryptoPanic:{coin}", value=score, weight=1.5)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_env_file()

    print("=" * 55)
    print("Testing free news sources (no keys needed)\n")

    print("1. Google News RSS:")
    e1 = build_evidence_from_google_news("Iran ceasefire 2026")
    e2 = build_evidence_from_google_news("Bitcoin price 2026")
    e3 = build_evidence_from_google_news("Federal Reserve interest rate cut")

    print("\n2. Reddit:")
    e4 = build_evidence_from_reddit("ceasefire negotiations", category="geopolitics")
    e5 = build_evidence_from_reddit("bitcoin price prediction", category="crypto")

    print("\n3. CryptoPanic (needs CRYPTOPANIC_KEY in .env):")
    e6 = build_evidence_from_cryptopanic("BTC")

    print("\nEvidence objects ready for BayesianModel.update():")
    for e in [e1, e2, e3, e4, e5]:
        if e:
            print(f"  {e.source:<35} value={e.value:.2f}  weight={e.weight:.2f}")
