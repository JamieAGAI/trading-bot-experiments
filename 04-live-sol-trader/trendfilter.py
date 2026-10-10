"""
trendfilter.py — SOL/USDT Weekly EMA50 Trend Filter (alert-only, single file)

Every Monday 01:00 UTC (after Sunday's weekly close):
  - Fetch SOLUSDT weekly klines from Binance
  - Compute EMA50 on the last 60 weekly closes
  - Compare to last known signal stored in state file
  - Send an email: loud flip alert if signal changed, quiet heartbeat otherwise

You execute the swap manually on Jupiter when a flip email arrives. No wallet
keys, no swaps, no balance checks — just price data in, email out.

Usage:
    python trendfilter.py               # LIVE — sends email (default)
    python trendfilter.py --shadow      # compute + print only, no email, no state write
    python trendfilter.py --force       # bypass once-per-week idempotency guard
    python trendfilter.py --test-email  # send a test email and exit (skips Binance)

Cron (Linux/Mac):
    0 1 * * 1 cd /path/to/solpy-agent && python trendfilter.py >> trendfilter.log 2>&1

Windows Task Scheduler:
    Trigger weekly Monday 01:00 UTC, action: python.exe C:\\path\\to\\trendfilter.py

Dependencies: python-binance, pandas (already in your env). SMTP is stdlib.
"""

import argparse
import json
import os
import smtplib
import sys
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
from binance.client import Client

# Load .env from current working dir (or any parent) — matches the working bot's setup.
# override=True so .env wins over any stale OS-level env vars on the machine.
# Silently no-op if python-dotenv isn't installed or no .env file is found.
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass


# =============================================================================
# CONFIG — adjust env var names below to match what your other bots already use
# =============================================================================

EMA_PERIOD = 50
KLINE_LIMIT = 60  # enough warmup for EMA50

STATE_FILE = Path("C:/Users/jamie/SOLPY/trendfilter_state.json")
LOG_FILE = Path("C:/Users/jamie/SOLPY/trendfilter_log.jsonl")

# Binance — public endpoints work without auth, keys optional
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Email — matches your other bots' convention (EMAIL_FROM / EMAIL_TO / EMAIL_PASS)
EMAIL_FROM = os.getenv("EMAIL_FROM", "")
EMAIL_PASS = os.getenv("EMAIL_PASS", "")
EMAIL_TO = os.getenv("EMAIL_TO", "") or os.getenv("EMAIL_FROM", "")  # default: send to self
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))  # Gmail SSL — matches the bot that works


# =============================================================================
# STATE
# =============================================================================

_EMPTY_STATE = {
    "last_signal": None,        # "bullish" | "bearish" | None (first run)
    "last_action_week": None,   # "YYYY-Www", e.g. "2026-W21"
    "last_close": None,
    "last_ema50": None,
}


def load_state() -> dict:
    if not STATE_FILE.exists():
        return dict(_EMPTY_STATE)
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        for k, v in _EMPTY_STATE.items():
            data.setdefault(k, v)
        return data
    except Exception as e:
        raise RuntimeError(
            f"State file corrupt ({STATE_FILE}): {e}. "
            "Fix or delete the file to reset."
        )


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def current_iso_week() -> str:
    return datetime.now(timezone.utc).strftime("%G-W%V")


def log_decision(record: dict) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# =============================================================================
# SIGNAL
# =============================================================================

def fetch_weekly_signal() -> tuple[bool, float, float]:
    """
    Returns (signal_on, latest_close, ema50).
    signal_on=True means close > EMA50 (be in SOL).
    Uses the most recently CLOSED weekly bar — works any day of the week.
    Raises RuntimeError on Binance failure or insufficient data.
    """
    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
    try:
        klines = client.get_klines(
            symbol="SOLUSDT",
            interval=Client.KLINE_INTERVAL_1WEEK,
            limit=KLINE_LIMIT,
        )
    except Exception as e:
        raise RuntimeError(f"Binance kline fetch failed: {e}")

    if not klines:
        raise RuntimeError("No klines returned from Binance")

    # Binance returns the in-progress current-week bar at index -1. Filter it out
    # so we only ever act on fully-closed weekly data.
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    closed = [k for k in klines if int(k[6]) < now_ms]
    if len(closed) < 2:
        raise RuntimeError(f"Insufficient closed klines: got {len(closed)}")

    closes = pd.Series([float(k[4]) for k in closed])
    ema = closes.ewm(span=EMA_PERIOD, adjust=False).mean()

    latest = closed[-1]
    latest_close = float(latest[4])
    latest_ema = float(ema.iloc[-1])

    return latest_close > latest_ema, latest_close, latest_ema


# =============================================================================
# EMAIL
# =============================================================================

def build_email(signal_on: bool, close: float, ema50: float,
                last_signal: str | None, flipped: bool) -> tuple[str, str]:
    new_signal = "bullish" if signal_on else "bearish"
    distance_pct = (close - ema50) / ema50 * 100

    if flipped and signal_on:
        subject = f"SOL EMA50 FLIP -> BUY  (close ${close:.2f} > EMA50 ${ema50:.2f})"
        action = "BUY SOL  --  swap USDT -> SOL on Jupiter"
    elif flipped and not signal_on:
        subject = f"SOL EMA50 FLIP -> SELL  (close ${close:.2f} <= EMA50 ${ema50:.2f})"
        action = "SELL SOL  --  swap SOL -> USDT on Jupiter"
    elif signal_on:
        subject = f"SOL EMA50: still bullish  (close ${close:.2f} > EMA50 ${ema50:.2f})"
        action = "Hold SOL"
    else:
        subject = f"SOL EMA50: still bearish  (close ${close:.2f} <= EMA50 ${ema50:.2f})"
        action = "Stay in USDT"

    body = (
        f"Week:            {current_iso_week()}\n"
        f"Run time (UTC):  {datetime.now(timezone.utc).isoformat()}\n"
        f"\n"
        f"Weekly close:    ${close:.4f}\n"
        f"EMA50:           ${ema50:.4f}\n"
        f"Distance:        {distance_pct:+.2f}%\n"
        f"\n"
        f"Previous signal: {last_signal or 'none (first run)'}\n"
        f"Current signal:  {new_signal}\n"
        f"Flipped:         {flipped}\n"
        f"\n"
        f"Action:          {action}\n"
    )
    return subject, body


def send_email(subject: str, body: str) -> None:
    if not (EMAIL_FROM and EMAIL_PASS and EMAIL_TO):
        raise RuntimeError(
            "Email env vars missing. Need EMAIL_FROM, EMAIL_PASS, EMAIL_TO."
        )
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
        smtp.login(EMAIL_FROM, EMAIL_PASS)
        smtp.send_message(msg)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SOL/USDT Weekly EMA50 Trend Alert (single file)"
    )
    parser.add_argument("--shadow", action="store_true",
                        help="Compute signal + print only. No email, no state write.")
    parser.add_argument("--force", action="store_true",
                        help="Bypass the once-per-ISO-week idempotency guard.")
    parser.add_argument("--test-email", action="store_true",
                        help="Send a test email and exit. Skips Binance and state entirely.")
    args = parser.parse_args()

    # --- Test-email path: skip everything, just fire one email and exit
    if args.test_email:
        run_ts = datetime.now(timezone.utc).isoformat()
        print(f"[trendfilter] {run_ts}  mode=TEST_EMAIL")
        print(f"  From: {EMAIL_FROM!r}  To: {EMAIL_TO!r}  Host: {SMTP_HOST}:{SMTP_PORT}")
        if not (EMAIL_FROM and EMAIL_PASS and EMAIL_TO):
            print("[ABORT] Missing EMAIL_FROM, EMAIL_PASS, or EMAIL_TO in environment.")
            sys.exit(1)
        try:
            send_email(
                subject="trendfilter.py test email",
                body=f"Test email from trendfilter.py at {run_ts}.\n"
                     f"If you can read this, SMTP works and the bot can alert you.\n",
            )
            print(f"  Test email sent to {EMAIL_TO}. Check your inbox.")
            return
        except Exception as e:
            print(f"[ABORT] Email send failed: {e}")
            sys.exit(1)

    mode = "SHADOW" if args.shadow else "LIVE"
    run_ts = datetime.now(timezone.utc).isoformat()
    iso_week = current_iso_week()

    print(f"[trendfilter] {run_ts}  mode={mode}  week={iso_week}")

    # --- Load state
    try:
        state = load_state()
    except RuntimeError as e:
        print(f"[ABORT] {e}")
        sys.exit(1)

    # --- Idempotency (skipped in shadow and when --force)
    if not args.shadow and not args.force and state.get("last_action_week") == iso_week:
        print(f"  Already ran this week ({iso_week}). Use --force to override.")
        sys.exit(0)

    # --- Fetch signal
    try:
        signal_on, close, ema50 = fetch_weekly_signal()
    except RuntimeError as e:
        print(f"[ABORT] {e}")
        log_decision({"ts": run_ts, "mode": mode, "action": "ABORT_SIGNAL", "reason": str(e)})
        sys.exit(1)

    new_signal = "bullish" if signal_on else "bearish"
    last_signal = state.get("last_signal")
    flipped = last_signal is not None and last_signal != new_signal

    print(f"  Close=${close:.4f}  EMA50=${ema50:.4f}  "
          f"Signal={new_signal.upper()}  Flipped={flipped}")

    subject, body = build_email(signal_on, close, ema50, last_signal, flipped)
    print(f"  Subject: {subject}")

    record = {
        "ts": run_ts,
        "week": iso_week,
        "mode": mode,
        "close": round(close, 4),
        "ema50": round(ema50, 4),
        "signal": new_signal,
        "last_signal": last_signal,
        "flipped": flipped,
        "email_sent": False,
    }

    # --- Shadow exit
    if args.shadow:
        print("  [SHADOW] Skipping email and state update.")
        print(f"\n--- email body that WOULD be sent ---\n{body}")
        log_decision(record)
        return

    # --- Send email (state is only updated on success, so retries are safe)
    try:
        send_email(subject, body)
        record["email_sent"] = True
        print(f"  Email sent to {EMAIL_TO}.")
    except Exception as e:
        print(f"  [WARN] Email send failed: {e}")
        record["email_error"] = str(e)
        log_decision(record)
        sys.exit(1)

    # --- Persist state
    state["last_signal"] = new_signal
    state["last_action_week"] = iso_week
    state["last_close"] = round(close, 4)
    state["last_ema50"] = round(ema50, 4)
    save_state(state)
    log_decision(record)
    print("Done.")


if __name__ == "__main__":
    main()