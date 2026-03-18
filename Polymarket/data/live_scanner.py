"""
Live Scanner — fetches Polymarket prices, builds model_p via
Binance (crypto) + Twitter/X sentiment (all categories), then
runs EV scanner and portfolio allocator.

Flow:
  1. Fetch top N markets from Polymarket CLOB
  2. For each market: compute model_p via signal_builder
       - Twitter/X sentiment  (if TWITTER_BEARER_TOKEN set)
       - Binance log-normal   (for crypto price markets)
  3. EV scan → filter EV > threshold
  4. Portfolio allocator → size trades via fractional Kelly
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.polymarket_client  import fetch_markets, filter_by_volume, LiveMarket
from data.crypto_client      import get_price, load_env_file
from data.signal_builder     import compute_bayesian_model_p
from data.live_scanner_utils import classify_market
from strategy.ev_scanner     import MarketSignal, rank_opportunities
from strategy.portfolio_manager import Portfolio, Trade, allocate


def run_live_scan(bankroll:  float = 10_000,
                  limit:     int   = 50,
                  min_vol:   float = 100_000,
                  min_ev:    float = 0.05,
                  verbose:   bool  = True) -> None:
    load_env_file()

    has_twitter = bool(os.environ.get("TWITTER_BEARER_TOKEN"))

    print("─" * 65)
    print(" LIVE POLYMARKET SCAN  (Bayesian: Binance + X sentiment)")
    print("─" * 65)
    print(f"  Twitter/X signals : {'ENABLED' if has_twitter else 'DISABLED (no TWITTER_BEARER_TOKEN)'}")
    print(f"  Binance prices    : ENABLED (no key needed)")

    # 1. Fetch live markets
    print(f"\n[1/4] Fetching top {limit} Polymarket markets by volume...")
    from datetime import datetime, timezone
    markets = fetch_markets(limit=limit)
    markets = filter_by_volume(markets, min_vol=min_vol)
    # Drop already-expired markets
    now = datetime.now(timezone.utc)
    markets = [m for m in markets
               if not m.end_date or
               datetime.fromisoformat(m.end_date.replace("Z", "+00:00")) > now]
    print(f"      {len(markets)} markets pass volume filter (>${min_vol:,.0f})")

    if not markets:
        print("No markets returned. Check internet connection.")
        return

    # 2. Connectivity checks
    print("\n[2/4] Checking data feeds...")
    btc_now = get_price("BTC")
    print(f"      Binance BTC/USDT : {'$'+f'{btc_now:,.0f}' if btc_now else 'OFFLINE'}")
    if has_twitter:
        print("      Twitter/X        : Bearer token found")

    # 3. Build model_p via Bayesian signal builder
    print(f"\n[3/4] Building model_p for {len(markets)} markets...")
    print(      "      (Twitter rate limit: 1 call per 15s — may take a moment)\n")

    signals = []
    for i, m in enumerate(markets):
        model_p = compute_bayesian_model_p(m, hours_back=6, verbose=False)
        signals.append(MarketSignal(
            name         = m.question[:60],
            market_price = m.yes_price,
            model_p      = model_p,
            volume_usd   = m.volume_usd,
        ))
        # Progress dot every 5 markets
        if (i + 1) % 5 == 0:
            print(f"      ... {i+1}/{len(markets)} done")

    # 4. EV scan
    print(f"\n[4/4] Scoring EV gaps (threshold: {min_ev:.0%})...\n")
    df         = rank_opportunities(signals, min_ev=min_ev, min_vol=min_vol)
    actionable = df[df["enter"]]

    if actionable.empty:
        print(f"No opportunities cleared EV > {min_ev:.0%}.")
        if not has_twitter:
            print("Tip: add TWITTER_BEARER_TOKEN to .env for sentiment-driven edges.")
        return

    if verbose:
        print(f"{'Market':<55} {'Price':>6}  {'Model':>6}  {'EV':>7}")
        print("─" * 80)
        for _, row in actionable.iterrows():
            q = str(row["market"])
            q = q[:52] + "..." if len(q) > 52 else q
            print(f"{q:<55} {row['price']:>6.1%}  "
                  f"{row['model_p']:>6.1%}  {row['ev_net']:>7.2%}")

    # 5. Allocate
    print(f"\nAllocating (bankroll ${bankroll:,.0f})...")
    portfolio = Portfolio(bankroll=bankroll)
    trades = [
        Trade(
            market       = row["market"],
            category     = classify_market(row["market"]),
            model_p      = row["model_p"],
            market_price = row["price"],
            volume_usd   = row["vol_usd"],
        )
        for _, row in actionable.iterrows()
    ]
    allocate(trades, portfolio)


if __name__ == "__main__":
    run_live_scan(bankroll=10_000, limit=50, min_vol=100_000)
