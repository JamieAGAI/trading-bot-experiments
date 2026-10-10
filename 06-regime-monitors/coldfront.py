#!/usr/bin/env python3
"""
================================================================================
  COLDFRONT  —  BTC Momentum Signal Monitor (alert-only)
================================================================================
Long-only BTC momentum strategy. Live monitor with Gmail alerts.
Built as a sibling to your existing SOL bot (main.py): same Binance data
source, same pandas_ta indicators, same email setup, same --shadow convention.

STRATEGY (validated walk-forward OOS 2022-2026: +0.58%/trade, 58% win,
PF 1.65, t=2.52, max DD -17%, ~28 trades/yr):
  ENTRY  (on a CLOSED 4h bar): RSI(14) > 70  AND  close > EMA200
  EXIT   : 2*ATR stop  OR  3*ATR target  OR  48h (12 bar) time stop
  Long only. One position at a time. No pyramiding.

WHAT IT DOES:
  - Wakes shortly after each 4h bar close, pulls BTCUSDT 4h klines from Binance.
  - Evaluates the signal only on CLOSED bars (the in-progress bar is dropped).
  - Emails you on BUY and on EXIT (stop/target/time).
  - Persists position state to JSON so it survives restarts.
  - ALERT-ONLY: it does NOT place orders. You execute on Coinbase yourself.
    (An execution hook is marked at the bottom for when you trust it.)

RUN:
  pip install python-binance pandas pandas_ta python-dotenv
  # .env (same keys your SOL bot uses; ONLY these three are needed here):
  #   EMAIL_FROM=you@gmail.com
  #   EMAIL_TO=you@gmail.com
  #   EMAIL_PASS=your_16_char_gmail_APP_password
  python coldfront.py            # live alerts
  python coldfront.py --shadow   # test mode: alerts prefixed [SHADOW], own state file

SECURITY: needs NO wallet key, NO RPC, NO Jupiter/exchange API key. It only
reads public Binance market data and sends email. Keep it that way until you
deliberately wire up execution.
================================================================================
"""
import os
import sys
import json
import time
import smtplib
import logging
import datetime as dt
from email.mime.text import MIMEText

import pandas as pd

try:
    import pandas_ta as ta
    HAVE_TA = True
except Exception:
    HAVE_TA = False

from binance.client import Client
from dotenv import load_dotenv

# ------------------------------------------------------------------ config ----
load_dotenv()

SHADOW_MODE = '--shadow' in sys.argv

EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO   = os.getenv("EMAIL_TO")
EMAIL_PASS = os.getenv("EMAIL_PASS")

SYMBOL        = "BTCUSDT"
INTERVAL_4H   = Client.KLINE_INTERVAL_4HOUR
LOOKBACK_DAYS = 120                      # plenty for EMA200 on 4h bars

# Strategy parameters (locked to the validated backtest)
RSI_LEN         = 14
RSI_ENTRY       = 70
EMA_TREND       = 200
ATR_LEN         = 14
ATR_MULT_STOP   = 2.0
ATR_MULT_TARGET = 3.0
MAX_HOLD_BARS   = 12                      # 12 * 4h = 48h

STATE_FILE = "coldfront_shadow.json" if SHADOW_MODE else "coldfront_state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("coldfront.log")],
)
logger = logging.getLogger("coldfront")


# --------------------------------------------------------------- indicators ---
def _rsi_fallback(close, period=14):
    d = close.diff(); g = d.where(d > 0, 0.0); l = -d.where(d < 0, 0.0)
    ag = g.ewm(com=period - 1, min_periods=period).mean()
    al = l.ewm(com=period - 1, min_periods=period).mean()
    return 100 - (100 / (1 + ag / al))

def _atr_fallback(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_indicators(df):
    df = df.copy()
    if HAVE_TA:
        df["ema200"] = ta.ema(df["close"], length=EMA_TREND)
        df["rsi"]    = ta.rsi(df["close"], length=RSI_LEN)
        df["atr"]    = ta.atr(df["high"], df["low"], df["close"], length=ATR_LEN)
    else:
        df["ema200"] = df["close"].ewm(span=EMA_TREND, adjust=False).mean()
        df["rsi"]    = _rsi_fallback(df["close"], RSI_LEN)
        df["atr"]    = _atr_fallback(df, ATR_LEN)
    return df.dropna()


# ------------------------------------------------------------------- data -----
def _klines_to_df(klines):
    """Same shape/logic as your main.py: build df and drop the in-progress bar."""
    df = pd.DataFrame(klines, columns=[
        'timestamp','open','high','low','close','volume',
        'close_time','quote_av','trades','tb_base_av','tb_quote_av','ignore'])
    df[['open','high','low','close','volume']] = \
        df[['open','high','low','close','volume']].astype(float)
    df = df.set_index(pd.to_datetime(df['timestamp'], unit='ms', utc=True))
    df.index.name = 'timestamp'
    df = df[['open','high','low','close','volume']]
    if len(df) >= 2:
        interval = df.index[-1] - df.index[-2]
    else:
        interval = pd.Timedelta(hours=4)
    if df.index[-1] + interval > pd.Timestamp.now('UTC'):
        df = df.iloc[:-1]          # drop still-forming bar -> no lookahead
    return df

def fetch_4h():
    for attempt in range(3):
        try:
            client = Client()
            end = dt.datetime.now(dt.timezone.utc)
            start = end - dt.timedelta(days=LOOKBACK_DAYS)
            klines = client.get_historical_klines(
                SYMBOL, INTERVAL_4H,
                start_str=start.strftime("%Y-%m-%d %H:%M:%S"),
                end_str=end.strftime("%Y-%m-%d %H:%M:%S"))
            df = _klines_to_df(klines)
            if len(df) < EMA_TREND + 5:
                raise RuntimeError(f"only {len(df)} closed bars, need >{EMA_TREND+5}")
            return df
        except Exception as e:
            logger.warning(f"Fetch attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(10)
    raise RuntimeError("Data fetch failed after 3 retries")


# ------------------------------------------------------------------ state -----
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"in_position": False, "entry": None, "stop": None, "target": None,
            "entry_ts": None, "bars_held": 0, "last_bar_seen": None}

def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2, default=str)


# ------------------------------------------------------------------ email -----
def send_alert(subject, body):
    if SHADOW_MODE:
        subject = f"[SHADOW] {subject}"
    if not EMAIL_FROM or not EMAIL_TO or not EMAIL_PASS:
        logger.warning("Email config missing -- skipping alert")
        logger.info("ALERT (not sent): %s\n%s", subject, body)
        return
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        logger.info("Alert sent!")
    except Exception as e:
        logger.warning(f"Failed to send alert: {e}")


# ------------------------------------------------------------- core logic -----
def evaluate(df, state):
    """Act on the latest CLOSED bar. Returns (event, details)."""
    feat = compute_indicators(df)
    bar = feat.iloc[-1]
    bar_ts = feat.index[-1].isoformat()
    price = float(bar["close"])

    if state.get("last_bar_seen") == bar_ts:
        return None, {"note": "bar already processed", "price": price,
                      "rsi": round(float(bar["rsi"]), 1)}
    state["last_bar_seen"] = bar_ts

    rsi_v = float(bar["rsi"]); atr_v = float(bar["atr"])
    above = price > float(bar["ema200"])

    # ---- manage open position ----
    if state["in_position"]:
        state["bars_held"] += 1
        hi, lo = float(bar["high"]), float(bar["low"])
        reason = exit_px = None
        if lo <= state["stop"]:                       # worst-case: stop first
            reason, exit_px = "STOP", state["stop"]
        elif hi >= state["target"]:
            reason, exit_px = "TARGET", state["target"]
        elif state["bars_held"] >= MAX_HOLD_BARS:
            reason, exit_px = "TIME (48h)", price
        if reason:
            pnl = (exit_px - state["entry"]) / state["entry"] * 100
            d = {"reason": reason, "exit": exit_px, "entry": state["entry"],
                 "pnl_pct": round(pnl, 2), "bars_held": state["bars_held"],
                 "bar_ts": bar_ts}
            state.update({"in_position": False, "entry": None, "stop": None,
                          "target": None, "entry_ts": None, "bars_held": 0})
            return "EXIT", d
        return None, {"note": f"holding {state['bars_held']}/{MAX_HOLD_BARS}",
                      "price": price, "rsi": round(rsi_v, 1)}

    # ---- flat: look for entry ----
    if rsi_v > RSI_ENTRY and above:
        entry, stop = price, price - ATR_MULT_STOP * atr_v
        target = price + ATR_MULT_TARGET * atr_v
        state.update({"in_position": True, "entry": entry, "stop": stop,
                      "target": target, "entry_ts": bar_ts, "bars_held": 0})
        return "BUY", {"entry": round(entry, 2), "stop": round(stop, 2),
                       "target": round(target, 2), "rsi": round(rsi_v, 1),
                       "atr": round(atr_v, 2), "bar_ts": bar_ts}

    return None, {"note": "flat, no signal" + ("" if above else " (below EMA200)"),
                  "price": price, "rsi": round(rsi_v, 1)}


def buy_body(d):
    return (f"COLDFRONT — BTC BUY SIGNAL\n{'='*40}\n"
            f"Bar closed: {d['bar_ts']}\n"
            f"Trigger: RSI(14)={d['rsi']} > 70 AND price > EMA200\n\n"
            f"Suggested entry: ${d['entry']:,.2f}\n"
            f"Stop  (2xATR):   ${d['stop']:,.2f}\n"
            f"Target(3xATR):   ${d['target']:,.2f}\n"
            f"Time stop:       48h from entry\n\n"
            f"ACTION: open a long BTC position on Coinbase. Risk ~1% of book "
            f"to the stop. One position at a time.\n")

def exit_body(d):
    tag = "PROFIT" if d["pnl_pct"] >= 0 else "LOSS"
    return (f"COLDFRONT — BTC EXIT ({d['reason']})\n{'='*40}\n"
            f"Bar closed: {d['bar_ts']}\n"
            f"Entry: ${d['entry']:,.2f}   Exit: ${d['exit']:,.2f}\n"
            f"Result: {d['pnl_pct']:+.2f}%  ({tag})\n"
            f"Held: {d['bars_held']} bars (~{d['bars_held']*4}h)\n\n"
            f"ACTION: close your BTC long on Coinbase.\n")


# ------------------------------------------------------------------- loop ------
def seconds_to_next_bar():
    """Sleep until ~1 min after the next 4h boundary (00,04,08,12,16,20 UTC)."""
    now = dt.datetime.now(dt.timezone.utc)
    hours_into = now.hour % 4
    next_boundary = (now.replace(minute=0, second=0, microsecond=0)
                     - dt.timedelta(hours=hours_into)
                     + dt.timedelta(hours=4))
    return max(60, (next_boundary - now).total_seconds() + 60)

def run_once(state):
    df = fetch_4h()
    event, d = evaluate(df, state)
    save_state(state)
    if event == "BUY":
        logger.info("BUY @ %s (rsi %s)", d["entry"], d["rsi"])
        send_alert("🟢 COLDFRONT: BTC BUY signal", buy_body(d))
        # --- EXECUTION HOOK (disabled). To auto-trade, place Coinbase order here.
    elif event == "EXIT":
        logger.info("EXIT %s pnl=%.2f%%", d["reason"], d["pnl_pct"])
        send_alert(f"🔴 COLDFRONT: BTC EXIT ({d['reason']})", exit_body(d))
        # --- EXECUTION HOOK (disabled). place Coinbase SELL here.
    else:
        logger.info("No action — %s | price=%s rsi=%s",
                    d.get("note",""), d.get("price"), d.get("rsi"))

def main():
    logger.info("COLDFRONT start | %s 4h | %s | pandas_ta=%s",
                SYMBOL, "SHADOW" if SHADOW_MODE else "LIVE", HAVE_TA)
    state = load_state()
    logger.info("State: in_position=%s", state["in_position"])
    send_alert("COLDFRONT started",
               f"Monitor running ({'SHADOW' if SHADOW_MODE else 'LIVE'}).\n"
               f"In position: {state['in_position']}\n"
               f"Watching {SYMBOL} 4h, alert-only.")
    run_once(state)                       # evaluate immediately on boot
    while True:
        sleep_s = seconds_to_next_bar()
        logger.info("Sleeping %.0f min until next 4h bar close.", sleep_s/60)
        time.sleep(sleep_s)
        try:
            run_once(state)
        except Exception as e:
            logger.error("Loop error: %s", e)   # log, don't crash; retry next bar

if __name__ == "__main__":
    main()
