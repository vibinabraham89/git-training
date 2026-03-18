"""
Twitter/X Client — fetches tweets for signal extraction.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Uses Twitter API v2 (Bearer Token, read-only).
Free tier: 500K tweets/month, 1 search per 15 seconds.

Set in .env:
  TWITTER_BEARER_TOKEN=your_bearer_token_here
"""

import os
import time
import requests
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional


TWITTER_BASE = "https://api.twitter.com/2"

# ── How many hours back to look ─────────────────────────────────────────────────
DEFAULT_LOOKBACK_HOURS = 6


@dataclass
class Tweet:
    id:           str
    text:         str
    author_id:    str
    created_at:   str
    retweet_count: int
    like_count:   int
    reply_count:  int

    @property
    def engagement(self) -> int:
        return self.retweet_count + self.like_count + self.reply_count


def _bearer_header() -> dict:
    token = os.environ.get("TWITTER_BEARER_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "TWITTER_BEARER_TOKEN not set.\n"
            "  1. Go to developer.twitter.com → create app\n"
            "  2. Copy Bearer Token\n"
            "  3. Add to .env:  TWITTER_BEARER_TOKEN=your_token"
        )
    return {"Authorization": f"Bearer {token}"}


def _get(endpoint: str, params: dict = None,
         retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            r = requests.get(
                f"{TWITTER_BASE}{endpoint}",
                headers=_bearer_header(),
                params=params,
                timeout=15,
            )
            if r.status_code == 429:
                # Rate limited — wait and retry
                wait = int(r.headers.get("x-rate-limit-reset", time.time() + 60))
                sleep_sec = max(wait - int(time.time()), 15)
                print(f"[X] Rate limited. Waiting {sleep_sec}s...")
                time.sleep(sleep_sec)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                print(f"[X] Failed after {retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def search_recent_tweets(query:        str,
                          max_results:  int = 100,
                          hours_back:   int = DEFAULT_LOOKBACK_HOURS,
                          exclude_rts:  bool = True) -> list[Tweet]:
    """
    Search recent tweets (last 7 days on free tier).
    query: Twitter search query string (supports AND, OR, from:user, etc.)
    """
    start_time = (
        datetime.now(timezone.utc) - timedelta(hours=hours_back)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    full_query = query
    if exclude_rts:
        full_query += " -is:retweet"

    params = {
        "query":       full_query,
        "max_results": min(max_results, 100),   # API cap per request
        "start_time":  start_time,
        "tweet.fields": "created_at,public_metrics,author_id",
    }

    data = _get("/tweets/search/recent", params=params)
    if not data or "data" not in data:
        return []

    tweets = []
    for t in data["data"]:
        m = t.get("public_metrics", {})
        tweets.append(Tweet(
            id            = t["id"],
            text          = t["text"],
            author_id     = t.get("author_id", ""),
            created_at    = t.get("created_at", ""),
            retweet_count = m.get("retweet_count", 0),
            like_count    = m.get("like_count", 0),
            reply_count   = m.get("reply_count", 0),
        ))

    return tweets


def get_user_tweet_count(username: str,
                          hours_back: int = 24) -> int:
    """
    Count how many tweets a user posted in the last N hours.
    Used for markets like 'Will Elon tweet >50 times today?'
    """
    query = f"from:{username}"
    tweets = search_recent_tweets(query, max_results=100, hours_back=hours_back)
    return len(tweets)


def get_user_id(username: str) -> Optional[str]:
    """Resolve a Twitter username to user ID."""
    data = _get(f"/users/by/username/{username}",
                params={"user.fields": "public_metrics"})
    if not data or "data" not in data:
        return None
    return data["data"]["id"]


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


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_env_file()
    if not os.environ.get("TWITTER_BEARER_TOKEN"):
        print("TWITTER_BEARER_TOKEN not set in .env")
    else:
        print("Testing X API...\n")
        tweets = search_recent_tweets("bitcoin OR BTC", max_results=10, hours_back=2)
        print(f"Found {len(tweets)} tweets about BTC in last 2 hours\n")
        for t in tweets[:3]:
            print(f"  [{t.like_count} likes] {t.text[:90]}...")
