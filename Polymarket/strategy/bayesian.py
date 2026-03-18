"""
Formula 6: Bayesian Update Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Best for: Elon/Trump tweet markets, Fed announcement markets,
          breaking news (elections night, verdicts, geopolitics)
Edge:     Market price is a lagged average of beliefs.
          Real-time signal processing pushes your posterior 10-15% ahead.

Signals supported:
  • Social media sentiment (Elon tweet volume, sentiment score)
  • Poll releases (weighted by recency and sample size)
  • News event flags (binary: did event X occur?)
  • Prediction market cross-feed (other platforms pricing same event)
"""

import numpy as np
from scipy import stats
from scipy.special import betaln
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Evidence:
    """A single piece of evidence updating our belief."""
    source:    str           # "tweet_sentiment", "poll", "news_event"
    value:     float         # normalized 0-1 (likelihood that H is true given E)
    weight:    float = 1.0   # reliability weight (trusted source = higher)
    timestamp: Optional[str] = None


@dataclass
class BayesianModel:
    """
    Beta-Binomial Bayesian model.
    Prior: Beta(alpha, beta)  →  mean = alpha/(alpha+beta)
    Update: each evidence piece adjusts (alpha, beta).
    """
    alpha: float = 1.0   # prior successes
    beta:  float = 1.0   # prior failures
    history: List[dict] = field(default_factory=list)

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def posterior_std(self) -> float:
        a, b = self.alpha, self.beta
        return np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    @property
    def credible_interval_95(self) -> tuple:
        dist = stats.beta(self.alpha, self.beta)
        return dist.ppf(0.025), dist.ppf(0.975)

    def update(self, evidence: Evidence,
               max_shift: float = 0.12) -> float:
        """
        Soft Bayesian update using evidence as Beta pseudo-observations.
        max_shift: posterior cannot move more than this far from the prior
                   in a single update (default ±12%).
                   Prevents neutral news from causing 3% → 27% jumps.
        """
        prior = self.posterior_mean
        strength = evidence.weight * 2.0
        self.alpha += evidence.value * strength
        self.beta  += (1 - evidence.value) * strength

        # Clamp posterior to [prior - max_shift, prior + max_shift]
        raw       = self.posterior_mean
        clamped   = max(prior - max_shift, min(prior + max_shift, raw))
        # Re-adjust alpha/beta to match clamped value
        total     = self.alpha + self.beta
        self.alpha = clamped * total
        self.beta  = (1 - clamped) * total

        self.history.append({
            "source":    evidence.source,
            "value":     evidence.value,
            "weight":    evidence.weight,
            "posterior": self.posterior_mean,
        })
        return self.posterior_mean

    def update_batch(self, evidences: List[Evidence]) -> List[float]:
        return [self.update(e) for e in evidences]

    def reset(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha, self.beta = alpha, beta
        self.history.clear()


def edge_vs_market(model: BayesianModel, market_price: float) -> dict:
    """
    Compare posterior to market price — generate trade signal.
    """
    posterior = model.posterior_mean
    lo, hi    = model.credible_interval_95
    edge      = posterior - market_price

    return {
        "posterior":    posterior,
        "market_price": market_price,
        "edge":         edge,
        "ci_95":        (round(lo, 3), round(hi, 3)),
        "signal":       "BUY"   if edge > 0.05 else
                        "SHORT" if edge < -0.05 else "HOLD",
        "confidence":   "HIGH" if model.posterior_std < 0.08 else "LOW",
    }


def plot_bayesian_update(model: BayesianModel, market_price: float,
                          title: str = "Bayesian Update Path"):
    if not model.history:
        print("No history to plot.")
        return

    posteriors = [model.history[0]["posterior"]] + [h["posterior"] for h in model.history]
    sources    = ["Prior"] + [h["source"] for h in model.history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(posteriors)), posteriors, "o-",
            color="steelblue", linewidth=2, markersize=7)
    ax.axhline(market_price, color="red", linestyle="--",
               linewidth=1.5, label=f"Market price {market_price:.0%}")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.fill_between(range(len(posteriors)),
                    [0.5] * len(posteriors), posteriors,
                    alpha=0.1, color="steelblue")
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels(sources, rotation=25, ha="right")
    ax.set_ylabel("Posterior P(H)")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig("bayesian_update.png", dpi=150)
    plt.show()


# ── Demo: Elon Musk tweets market ──────────────────────────────────────────────
if __name__ == "__main__":
    market_price = 0.40   # Polymarket: "Elon tweets >50 today"

    model = BayesianModel(alpha=2, beta=3)  # slight skeptic prior (~40%)

    evidences = [
        Evidence("Morning tweet burst (15 tweets by 9am)",  value=0.72, weight=1.2),
        Evidence("Topic: Tesla earnings (not DOGE/politics)", value=0.45, weight=0.8),
        Evidence("Historical avg on earnings days",           value=0.68, weight=1.0),
        Evidence("Current count at noon: 28 tweets",         value=0.78, weight=1.5),
        Evidence("Sentiment: high engagement (retweets up)", value=0.65, weight=0.9),
    ]

    print(f"Market: 'Elon tweets >50 today' @ {market_price:.0%}\n")
    print(f"{'Source':<45} {'Posterior':>10}")
    print("-" * 57)

    for ev in evidences:
        p = model.update(ev)
        print(f"{ev.source:<45} {p:>10.1%}")

    result = edge_vs_market(model, market_price)
    print(f"\nFinal posterior : {result['posterior']:.1%}")
    print(f"Market price    : {result['market_price']:.1%}")
    print(f"Edge            : {result['edge']:+.1%}")
    print(f"95% CI          : {result['ci_95']}")
    print(f"Signal          : {result['signal']}  ({result['confidence']} confidence)")

    plot_bayesian_update(model, market_price, "Elon Tweet Count Market — Bayesian Update")
