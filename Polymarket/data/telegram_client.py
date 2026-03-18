"""
Telegram Alert Client
━━━━━━━━━━━━━━━━━━━━
Sends trade signals to your Telegram phone in real time.

Setup:
  1. Open Telegram → search @BotFather → /newbot → copy token
  2. Open your new bot → click Start
  3. Run: python data/telegram_client.py --setup
     This auto-fetches your chat_id and writes it to .env
  4. Run: python data/telegram_client.py --test
     Sends a test message to confirm it works
"""

import os
import sys
import json
import argparse
import requests

TELEGRAM_API = "https://api.telegram.org"


def load_env(path: str = ".env"):
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass


def _get_token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "TELEGRAM_BOT_TOKEN not set.\n"
            "  1. Message @BotFather on Telegram → /newbot\n"
            "  2. Add TELEGRAM_BOT_TOKEN=your_token to .env"
        )
    return token


def get_chat_id() -> str | None:
    """
    Fetch the chat_id of whoever last messaged your bot.
    User must have sent /start to the bot first.
    """
    token = _get_token()
    try:
        r = requests.get(
            f"{TELEGRAM_API}/bot{token}/getUpdates",
            timeout=10,
        )
        r.raise_for_status()
        data    = r.json()
        results = data.get("result", [])
        if not results:
            print("No messages found. Make sure you sent /start to your bot first.")
            return None
        # Get the most recent message's chat id
        chat_id = str(results[-1]["message"]["chat"]["id"])
        print(f"Your chat_id: {chat_id}")
        return chat_id
    except Exception as e:
        print(f"Error fetching chat_id: {e}")
        return None


def write_chat_id_to_env(chat_id: str, path: str = ".env"):
    """Write the fetched chat_id back into .env."""
    try:
        with open(path) as f:
            lines = f.readlines()
        with open(path, "w") as f:
            for line in lines:
                if line.startswith("TELEGRAM_CHAT_ID="):
                    f.write(f"TELEGRAM_CHAT_ID={chat_id}\n")
                else:
                    f.write(line)
        print(f"Saved TELEGRAM_CHAT_ID={chat_id} to {path}")
    except Exception as e:
        print(f"Could not write to .env: {e}")
        print(f"Add this manually: TELEGRAM_CHAT_ID={chat_id}")


def send_message(text: str,
                  chat_id: str | None = None,
                  parse_mode: str = "Markdown") -> bool:
    """
    Send a message to your Telegram chat.
    Returns True on success.
    """
    token   = _get_token()
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not chat_id:
        print("TELEGRAM_CHAT_ID not set. Run: python data/telegram_client.py --setup")
        return False
    try:
        r = requests.post(
            f"{TELEGRAM_API}/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram send error: {e}")
        return False


def format_signal_message(signal: dict) -> str:
    """Format a trade signal dict into a clean Telegram message."""
    ev_pct = signal.get("ev_net", 0)
    # Cap display at 999% to avoid ugly huge numbers from near-zero prices
    ev_display = f"{min(ev_pct * 100, 999):.1f}%" if ev_pct < 10 else f"{ev_pct * 100:.0f}%"

    return (
        f"*Polymarket Signal* \U0001F4C8\n"
        f"{'─' * 30}\n"
        f"*{signal.get('market', '')}*\n\n"
        f"\U0001F4B0 Size:     `${signal.get('size_usd', 0):,.0f}`\n"
        f"\U0001F3AF EV:       `{ev_display}`\n"
        f"\U0001F4C2 Category: `{signal.get('category', '')}`\n"
        f"\u2705 Signal:   *{signal.get('signal', '')}*\n"
        f"\U0001F552 Time:     `{signal.get('timestamp', '')[:16]}`"
    )


def send_signals(signals: list[dict]) -> int:
    """Send all signals. Returns count of successful sends."""
    if not signals:
        return 0
    sent = 0
    for s in signals:
        msg = format_signal_message(s)
        if send_message(msg):
            sent += 1
    return sent


def send_scan_summary(signals: list[dict], btc_price: float | None = None):
    """
    Send a single summary message instead of one per signal.
    Better for high-frequency runs to avoid notification spam.
    """
    if not signals:
        send_message("*Polymarket Scan* — No signals this run \U0001F50D")
        return

    lines = ["*Polymarket Scan* \U0001F4CA\n"]
    if btc_price:
        lines.append(f"BTC: `${btc_price:,.0f}`\n")
    lines.append(f"{len(signals)} signal(s) found:\n")

    for s in signals:
        ev = s.get("ev_net", 0)
        ev_str = f"{min(ev * 100, 999):.0f}%" if ev < 10 else f"{ev * 100:.0f}%"
        lines.append(
            f"\u2022 *{s['market'][:45]}*\n"
            f"  EV `{ev_str}` | Size `${s['size_usd']:,.0f}` | {s['category']}\n"
        )

    send_message("\n".join(lines))


# ── CLI helpers ───────────────────────────────────────────────────────────────
def setup():
    """Auto-fetch chat_id and save to .env."""
    print("Fetching your Telegram chat_id...\n")
    chat_id = get_chat_id()
    if chat_id:
        write_chat_id_to_env(chat_id)
        print("\nSetup complete. Run --test to verify.")
    else:
        print("\nMake sure you:")
        print("  1. Have TELEGRAM_BOT_TOKEN set in .env")
        print("  2. Opened your bot in Telegram and sent /start")


def test_alert():
    """Send a test signal to verify everything works."""
    test_signal = {
        "market":    "Test: Will this alert work?",
        "category":  "test",
        "size_usd":  250.00,
        "ev_net":    0.18,
        "signal":    "BUY",
        "timestamp": "2026-03-19T00:00:00",
    }
    print("Sending test alert...")
    ok = send_message(format_signal_message(test_signal))
    if ok:
        print("Success! Check your Telegram.")
    else:
        print("Failed. Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")


if __name__ == "__main__":
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true",
                        help="Fetch chat_id and save to .env")
    parser.add_argument("--test",  action="store_true",
                        help="Send a test alert")
    args = parser.parse_args()

    if args.setup:
        setup()
    elif args.test:
        test_alert()
    else:
        parser.print_help()
