"""
Cron Runner — entry point for Railway scheduled job.
Runs every 15 minutes. Fetches markets, scores signals,
logs results, and (Step 4) sends Telegram alerts.

Railway executes: python cron_runner.py
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone

# ── logs dir must exist before FileHandler is created ─────────────────────────
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),       # Railway captures stdout
        logging.FileHandler("logs/scanner.log"), # local backup
    ],
)
log = logging.getLogger("cron")


def load_env():
    """Load .env locally. Railway injects env vars automatically."""
    try:
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass


def run_scan() -> list[dict]:
    """Run the full pipeline. Returns list of actionable trade dicts."""
    from data.polymarket_client     import fetch_markets, filter_by_volume
    from data.crypto_client         import get_price
    from data.signal_builder        import compute_bayesian_model_p
    from data.live_scanner_utils    import classify_market
    from strategy.ev_scanner        import MarketSignal, rank_opportunities
    from strategy.portfolio_manager import Portfolio, Trade, allocate

    MIN_VOL  = float(os.environ.get("MIN_VOL_USD",    "100000"))
    MIN_EV   = float(os.environ.get("MIN_EV",         "0.05"))
    LIMIT    = int(os.environ.get("MARKET_LIMIT",     "50"))
    BANKROLL = float(os.environ.get("BANKROLL",       "10000"))

    log.info("─" * 55)
    log.info("SCAN START")

    # 1. Fetch and filter markets
    now     = datetime.now(timezone.utc)
    markets = fetch_markets(limit=LIMIT)
    markets = filter_by_volume(markets, min_vol=MIN_VOL)
    markets = [m for m in markets
               if not m.end_date or
               datetime.fromisoformat(
                   m.end_date.replace("Z", "+00:00")) > now]
    log.info(f"Markets: {len(markets)} active above ${MIN_VOL:,.0f}")

    if not markets:
        log.warning("No markets found.")
        return []

    # 2. Binance connectivity check
    btc = get_price("BTC")
    log.info(f"BTC/USDT: ${btc:,.0f}" if btc else "Binance unavailable")

    # 3. Build model_p via Bayesian signal builder
    signals = []
    for m in markets:
        p = compute_bayesian_model_p(m, verbose=False)
        signals.append(MarketSignal(
            name         = m.question[:60],
            market_price = m.yes_price,
            model_p      = p,
            volume_usd   = m.volume_usd,
        ))

    # 4. EV scan
    df         = rank_opportunities(signals, min_ev=MIN_EV, min_vol=MIN_VOL)
    actionable = df[df["enter"]]

    if actionable.empty:
        log.info("No actionable signals this run.")
        return []

    # 5. Allocate
    portfolio = Portfolio(bankroll=BANKROLL)
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
    allocation_df = allocate(trades, portfolio, verbose=False)

    # 6. Collect results
    results = []
    for _, row in allocation_df.iterrows():
        r = {
            "market":    row["market"],
            "category":  row["category"],
            "size_usd":  round(row["size_usd"], 2),
            "ev_net":    round(row["ev_net"], 4),
            "signal":    row["signal"],
            "timestamp": now.isoformat(),
        }
        results.append(r)
        log.info(f"SIGNAL  {r['signal']:4s}  "
                 f"EV={r['ev_net']:.2%}  "
                 f"${r['size_usd']:>7,.0f}  "
                 f"{r['market'][:55]}")

    log.info(f"SCAN END — {len(results)} signal(s)")
    return results


def save_results(results: list[dict]):
    """Append results to a rolling JSON log (last 500 records)."""
    path     = "logs/signals.json"
    existing = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.extend(results)
    existing = existing[-500:]
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def send_telegram_alerts(results: list[dict]):
    """Send a Telegram summary for this scan run."""
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        log.info("Telegram not configured (add tokens to Railway env vars).")
        return
    try:
        from data.telegram_client import send_scan_summary
        send_scan_summary(results)
        log.info(f"Telegram summary sent ({len(results)} signal(s))")
    except Exception as e:
        log.warning(f"Telegram error: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_env()
    try:
        results = run_scan()
        if results:
            save_results(results)
            send_telegram_alerts(results)
        sys.exit(0)
    except Exception as e:
        log.exception(f"Cron job crashed: {e}")
        sys.exit(1)
