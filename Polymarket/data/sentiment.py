"""
Sentiment Scorer — converts tweets into Evidence objects for the Bayesian engine.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
No ML model needed — uses keyword scoring + engagement weighting.
Good enough to beat static market prices. Swap in an LLM later for +5% accuracy.

Output: list[Evidence] → fed directly into BayesianModel.update_batch()
"""

import re
from dataclasses import dataclass
from typing import Optional
from data.twitter_client import Tweet


# ── Keyword banks ───────────────────────────────────────────────────────────────
# Each word has a direction score: +1 = bullish/yes, -1 = bearish/no

BULLISH_WORDS = {
    # General positive
    "yes": 0.7, "true": 0.7, "confirmed": 0.9, "official": 0.8,
    "announced": 0.8, "happening": 0.7, "done": 0.8, "signed": 0.9,
    "agreed": 0.8, "passed": 0.8, "approved": 0.8, "deal": 0.7,
    # Crypto
    "bullish": 0.8, "moon": 0.6, "buy": 0.6, "pump": 0.6,
    "all-time high": 0.9, "ath": 0.9, "breakout": 0.8,
    # Politics/geo
    "ceasefire": 0.85, "peace": 0.75, "talks": 0.6, "meeting": 0.5,
    "negotiation": 0.65, "progress": 0.65, "breakthrough": 0.9,
    "victory": 0.85, "win": 0.75, "lead": 0.6,
}

BEARISH_WORDS = {
    # General negative
    "no": 0.3, "false": 0.3, "denied": 0.2, "failed": 0.2,
    "cancelled": 0.1, "rejected": 0.2, "blocked": 0.25, "suspended": 0.2,
    "delayed": 0.3, "postponed": 0.3, "unlikely": 0.25,
    # Crypto
    "bearish": 0.2, "dump": 0.2, "sell": 0.3, "crash": 0.15,
    "correction": 0.35, "down": 0.3,
    # Politics/geo
    "collapse": 0.1, "failure": 0.15, "war": 0.2, "attack": 0.2,
    "escalation": 0.15, "sanctions": 0.3, "breakdown": 0.15,
    "losing": 0.2, "lost": 0.2, "behind": 0.35,
}


def score_tweet_text(text: str) -> float:
    """
    Score a single tweet → float in [0, 1].
    0 = strongly bearish/NO, 1 = strongly bullish/YES, 0.5 = neutral.
    """
    text_lower = text.lower()
    bull_score = 0.0
    bear_score = 0.0

    for word, strength in BULLISH_WORDS.items():
        if word in text_lower:
            bull_score += strength

    for word, strength in BEARISH_WORDS.items():
        if word in text_lower:
            bear_score += strength

    total = bull_score + bear_score
    if total == 0:
        return 0.5   # neutral

    # Normalise to [0, 1]
    raw = bull_score / total
    # Shrink towards 0.5 — don't overreact to single tweets
    return 0.5 + (raw - 0.5) * 0.6


def engagement_weight(tweet: Tweet,
                       max_engagement: int = 10_000) -> float:
    """
    Weight a tweet by its engagement (likes + retweets + replies).
    High-engagement tweets get weight up to 2.0×.
    Low-engagement tweets get weight 0.5×.
    """
    eng = min(tweet.engagement, max_engagement)
    # Log scale: avoids one viral tweet dominating
    import math
    if eng == 0:
        return 0.5
    return 0.5 + 1.5 * (math.log1p(eng) / math.log1p(max_engagement))


@dataclass
class SentimentSummary:
    query:         str
    n_tweets:      int
    avg_score:     float    # 0-1 scale
    avg_weight:    float
    bullish_count: int
    bearish_count: int
    neutral_count: int
    top_tweets:    list     # top 3 by engagement


def analyse_tweets(tweets: list[Tweet],
                    query:  str = "") -> SentimentSummary:
    """
    Aggregate tweet scores into a summary.
    """
    if not tweets:
        return SentimentSummary(query, 0, 0.5, 1.0, 0, 0, 0, [])

    scores      = [score_tweet_text(t.text) for t in tweets]
    weights     = [engagement_weight(t) for t in tweets]
    total_w     = sum(weights)
    avg_score   = sum(s * w for s, w in zip(scores, weights)) / total_w

    bullish = sum(1 for s in scores if s > 0.6)
    bearish = sum(1 for s in scores if s < 0.4)
    neutral = len(scores) - bullish - bearish

    top = sorted(tweets, key=lambda t: t.engagement, reverse=True)[:3]

    return SentimentSummary(
        query         = query,
        n_tweets      = len(tweets),
        avg_score     = avg_score,
        avg_weight    = sum(weights) / len(weights),
        bullish_count = bullish,
        bearish_count = bearish,
        neutral_count = neutral,
        top_tweets    = top,
    )


def summary_to_evidence(summary: SentimentSummary,
                          source_label: str = "twitter_sentiment"):
    """
    Convert a SentimentSummary into an Evidence object
    ready to feed into BayesianModel.update().
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from strategy.bayesian import Evidence

    # Weight scales with how many tweets we have (more data = more confidence)
    import math
    confidence = min(1.5, 0.5 + math.log1p(summary.n_tweets) / 5)

    return Evidence(
        source    = f"{source_label} ({summary.n_tweets} tweets)",
        value     = summary.avg_score,
        weight    = confidence,
        timestamp = None,
    )


# ── Convenience: one-shot fetch + score ─────────────────────────────────────────
def fetch_and_score(query:       str,
                     hours_back:  int = 6,
                     max_results: int = 100,
                     verbose:     bool = True):
    """
    Full pipeline: search tweets → score → return Evidence.
    Falls back gracefully if API key is missing.
    """
    from data.twitter_client import search_recent_tweets

    tweets  = search_recent_tweets(query, max_results=max_results,
                                    hours_back=hours_back)
    summary = analyse_tweets(tweets, query=query)

    if verbose:
        direction = ("BULLISH" if summary.avg_score > 0.55
                     else "BEARISH" if summary.avg_score < 0.45
                     else "NEUTRAL")
        print(f"\n[X Sentiment] '{query}'")
        print(f"  Tweets: {summary.n_tweets}  |  "
              f"Score: {summary.avg_score:.2f}  |  {direction}")
        print(f"  Bullish: {summary.bullish_count}  "
              f"Bearish: {summary.bearish_count}  "
              f"Neutral: {summary.neutral_count}")
        if summary.top_tweets:
            print("  Top tweet:")
            print(f"    [{summary.top_tweets[0].engagement} eng] "
                  f"{summary.top_tweets[0].text[:100]}...")

    return summary_to_evidence(summary, source_label=f"X:{query[:30]}")
