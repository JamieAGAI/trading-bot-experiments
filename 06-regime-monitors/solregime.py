#!/usr/bin/env python3
"""
================================================================================
  SOLREGIME  —  SOL/USDT Strong Bull Regime Monitor (alert-only)
================================================================================
Implements the "Bull Regime" strategy — a relaxed version of the original
strong-bull-only filter (weekly RSI > 50 instead of > 60). Backtests showed
this captures trending moves the strict filter sat out, improving performance
on BTC and on SOL's later (post-2021) years, while keeping drawdown low.

STRATEGY:
  LONG  (BUY alert) when regime is bull OR strong_bull
  FLAT  (SELL alert) when regime exits to bull_chop / bear / chop
  (To revert to the original behaviour, set LONG_REGIMES = {"strong_bull"}.)

REGIME DEFINITIONS (no lookahead — uses completed bars only):
  strong_bull: weekly_close > weekly_EMA50  AND  weekly_RSI14 > 60
               AND  daily_close > daily_EMA50 > daily_EMA200
  bull:        weekly_close > weekly_EMA50  AND  weekly_RSI14 > 50
               AND  daily_close > daily_EMA50 > daily_EMA200
  bull_chop:   weekly_close > weekly_EMA50  AND  weekly_RSI14 > 50
               AND NOT (daily_close < daily_EMA50 < daily_EMA200)
  bear:        weekly_close < weekly_EMA50  AND  weekly_RSI14 < 50
               AND  daily_close < daily_EMA50 < daily_EMA200
  chop:        everything else

WHAT IT DOES:
  - Wakes shortly after each 4h bar close, pulls SOLUSDT 4h klines from Binance.
  - Resamples to daily and weekly, uses only COMPLETED bars for regime.
  - Emails you on regime transitions that cross the strong_bull boundary.
  - Persists state to JSON so it survives restarts.
  - ALERT-ONLY: does NOT place orders. You execute manually on your exchange.
  - Sends a daily heartbeat so you know it's running.

RUN:
  python solregime.py            # live alerts
  python solregime.py --shadow   # test mode: [SHADOW] prefix, separate state file

SETUP (.env keys needed):
  EMAIL_FROM=you@gmail.com
  EMAIL_TO=you@gmail.com
  EMAIL_PASS=your_16_char_gmail_APP_password
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
from binance.client import Client
from dotenv import load_dotenv

# ------------------------------------------------------------------ config ----
load_dotenv()

SHADOW_MODE = "--shadow" in sys.argv

EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO   = os.getenv("EMAIL_TO")
EMAIL_PASS = os.getenv("EMAIL_PASS")

SYMBOL       = "SOLUSDT"
INTERVAL_4H  = Client.KLINE_INTERVAL_4HOUR
FETCH_DAYS   = 430           # 4h bars for 430 days -> ~61 weekly bars + daily EMA200 warmup
HEARTBEAT_H  = 24            # send a status email every N hours

# Which regimes count as "in a long position":
#   {"strong_bull"}          -> original, most selective (weekly RSI > 60)
#   {"bull", "strong_bull"}  -> relaxed (weekly RSI > 50); better in BTC + SOL's later years
LONG_REGIMES = {"bull", "strong_bull"}

STATE_FILE = "solregime_shadow.json" if SHADOW_MODE else "solregime_state.json"
LOG_FILE   = "solregime_shadow.log"  if SHADOW_MODE else "solregime.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
# Force stdout to handle utf-8 on Windows (avoids cp1252 UnicodeEncodeError)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
log = logging.getLogger("solregime")


# ------------------------------------------------------------------- data -----
def _klines_to_df(klines):
    """Parse Binance kline list, drop the in-progress (live) bar."""
    df = pd.DataFrame(klines, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","quote_av","trades","tb_base","tb_quote","ignore",
    ])
    df[["open","high","low","close","volume"]] = \
        df[["open","high","low","close","volume"]].astype(float)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    df = df[["open","high","low","close","volume"]]
    df.sort_index(inplace=True)
    # Drop the still-forming bar
    if len(df) >= 2:
        bar_span = df.index[-1] - df.index[-2]
    else:
        bar_span = pd.Timedelta(hours=4)
    now_utc = pd.Timestamp.now("UTC")
    if df.index[-1] + bar_span > now_utc:
        df = df.iloc[:-1]
    return df

def fetch_4h():
    """Fetch FETCH_DAYS of closed SOLUSDT 4h bars. Retries 3×."""
    for attempt in range(3):
        try:
            client = Client()
            end   = dt.datetime.now(dt.timezone.utc)
            start = end - dt.timedelta(days=FETCH_DAYS)
            klines = client.get_historical_klines(
                SYMBOL, INTERVAL_4H,
                start_str=start.strftime("%Y-%m-%d %H:%M:%S"),
                end_str=end.strftime("%Y-%m-%d %H:%M:%S"),
            )
            df = _klines_to_df(klines)
            log.info("Fetched %d closed 4h bars (%s to %s)",
                     len(df), df.index[0].date(), df.index[-1].date())
            return df
        except Exception as e:
            log.warning("Fetch attempt %d/3 failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(15)
    raise RuntimeError("Data fetch failed after 3 retries")


# ---------------------------------------------------------------- indicators --
def _rsi(close, n=14):
    d  = close.diff()
    g  = d.clip(lower=0).ewm(com=n - 1, min_periods=n).mean()
    l  = (-d).clip(lower=0).ewm(com=n - 1, min_periods=n).mean()
    return 100 - 100 / (1 + g / l)

def compute_regime(df_4h):
    """
    Build daily and weekly frames from the 4h data.
    Use only COMPLETED daily and weekly bars (exclude in-progress bar).
    Returns: (regime_str, context_dict) or (None, error_dict) on failure.
    """
    now_utc   = pd.Timestamp.now("UTC").normalize()   # midnight today UTC
    # Monday of this week (Mon = weekday 0)
    this_week_mon = now_utc - pd.Timedelta(days=now_utc.dayofweek)

    # ── Resample to daily and weekly
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    daily  = df_4h.resample("1D").agg(agg).dropna()
    weekly = df_4h.resample("W-MON").agg(agg).dropna()

    # ── Exclude in-progress bars (strict: bars BEFORE today / this week's Mon)
    daily_c  = daily[daily.index.normalize()  < now_utc].copy()
    weekly_c = weekly[weekly.index.normalize() < this_week_mon].copy()

    if len(daily_c) < 205:
        return None, {"error": f"need 205+ daily bars, have {len(daily_c)}"}
    if len(weekly_c) < 55:
        return None, {"error": f"need 55+ weekly bars, have {len(weekly_c)}"}

    # ── Daily indicators
    daily_c["ema50"]  = daily_c["close"].ewm(span=50,  adjust=False).mean()
    daily_c["ema200"] = daily_c["close"].ewm(span=200, adjust=False).mean()

    # ── Weekly indicators
    weekly_c["ema50"] = weekly_c["close"].ewm(span=50, adjust=False).mean()
    weekly_c["rsi"]   = _rsi(weekly_c["close"])

    d = daily_c.iloc[-1]
    w = weekly_c.iloc[-1]

    if pd.isna(d["ema200"]) or pd.isna(w["ema50"]) or pd.isna(w["rsi"]):
        return None, {"error": "NaN in indicators (not enough warm-up)"}

    dc = float(d["close"]); de50 = float(d["ema50"]); de200 = float(d["ema200"])
    wc = float(w["close"]); we50 = float(w["ema50"]); wrsi  = float(w["rsi"])

    bull_weekly = wc > we50 and wrsi > 50
    bear_weekly = wc < we50 and wrsi < 50
    bull_daily  = dc > de50 > de200
    bear_daily  = dc < de50 < de200

    if   bull_weekly and bull_daily and wrsi > 60: regime = "strong_bull"
    elif bull_weekly and bull_daily:               regime = "bull"
    elif bull_weekly and not bear_daily:           regime = "bull_chop"
    elif bear_weekly and bear_daily:               regime = "bear"
    else:                                          regime = "chop"

    sol_price = float(df_4h["close"].iloc[-1])

    ctx = {
        "regime":     regime,
        "sol_price":  round(sol_price, 4),
        "daily_date": str(daily_c.index[-1].date()),
        "daily_close":round(dc,   2),
        "daily_ema50": round(de50,  2),
        "daily_ema200":round(de200, 2),
        "daily_bull":  bull_daily,
        "weekly_date": str(weekly_c.index[-1].date()),
        "weekly_close":round(wc,   2),
        "weekly_ema50":round(we50,  2),
        "weekly_rsi":  round(wrsi,  1),
        "weekly_bull_weekly": bull_weekly,
        "bar_ts":      str(df_4h.index[-1]),
        # Gap metrics
        "pct_to_weekly_ema50": round((we50 - wc) / wc * 100, 1),
        "daily_ema_gap":       round((de50 - de200) / de200 * 100, 1),
    }
    return regime, ctx


# ------------------------------------------------------------------ state -----
def _empty_state():
    return {
        "in_strong_bull":    False,
        "last_regime":       None,
        "entry_price":       None,
        "entry_bar_ts":      None,
        "last_bar_seen":     None,
        "last_heartbeat_ts": None,
    }

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
            # back-fill any missing keys
            base = _empty_state()
            for k, v in base.items():
                s.setdefault(k, v)
            return s
        except Exception as e:
            log.warning("Failed to load state (%s), starting fresh", e)
    return _empty_state()

def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2, default=str)


# ------------------------------------------------------------------ email -----
def send_alert(subject, body):
    prefix = "[SHADOW] " if SHADOW_MODE else ""
    subject = prefix + subject
    if not (EMAIL_FROM and EMAIL_TO and EMAIL_PASS):
        log.warning("Email config missing — printing instead:")
        log.info("SUBJECT: %s\n%s", subject, body)
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = EMAIL_FROM
        msg["To"]      = EMAIL_TO
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
            srv.login(EMAIL_FROM, EMAIL_PASS)
            srv.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        log.info("Alert sent: %s", subject)
    except Exception as e:
        log.warning("Email failed: %s", e)


# --------------------------------------------------------------- email bodies --
_SEP = "=" * 52

def _regime_block(ctx):
    """Compact summary of all regime inputs — included in every alert."""
    in_sb   = ctx["in_strong_bull"] if "in_strong_bull" in ctx else (ctx["regime"] == "strong_bull")
    d_align = "BULLISH (close > EMA50 > EMA200)" if ctx["daily_bull"] else "not bullish"
    w_bull  = "YES" if ctx["weekly_bull_weekly"] else "NO"
    sb_gap  = ctx["pct_to_weekly_ema50"]
    arrow   = "above" if sb_gap < 0 else f"+{sb_gap}% below"
    return f"""\
REGIME INPUTS ({ctx['bar_ts'][:16]} UTC)
  SOL price:       ${ctx['sol_price']:,.4f}

  WEEKLY (bar: {ctx['weekly_date']}):
    Close:         ${ctx['weekly_close']:,.2f}
    EMA50:         ${ctx['weekly_ema50']:,.2f}  ({arrow} weekly EMA50)
    RSI14:         {ctx['weekly_rsi']}  (>50 bull_weekly, >60 strong_bull)
    Bull weekly:   {w_bull}

  DAILY (bar: {ctx['daily_date']}):
    Close:         ${ctx['daily_close']:,.2f}
    EMA50:         ${ctx['daily_ema50']:,.2f}
    EMA200:        ${ctx['daily_ema200']:,.2f}
    Alignment:     {d_align}

  REGIME:  {ctx['regime'].upper().replace('_',' ')}"""

def buy_body(ctx, prev_regime):
    return f"""\
SOLREGIME — SOL BUY SIGNAL
{_SEP}
Regime changed: {prev_regime.upper()} → {ctx['regime'].upper()}
Timestamp: {ctx['bar_ts'][:16]} UTC

{_regime_block(ctx)}

ACTION: Go LONG SOL.
  - Trend regime is bullish (weekly close > EMA50, weekly RSI > 50, daily aligned).
  - Hold until SELL alert (regime exits bull / strong_bull).
  - Suggested: execute on your exchange at market or limit near current price.

Strategy: Bull Regime (weekly RSI > 50, daily close > EMA50 > EMA200)
"""

def sell_body(ctx, prev_regime, entry_price):
    if entry_price:
        pnl = (ctx["sol_price"] - entry_price) / entry_price * 100
        pnl_line = f"Approx P&L vs entry (${entry_price:,.4f}): {pnl:+.1f}%"
    else:
        pnl_line = "Entry price not recorded (state missing)."
    return f"""\
SOLREGIME — SOL SELL / EXIT SIGNAL
{_SEP}
Regime changed: {prev_regime.upper()} → {ctx['regime'].upper()}
Timestamp: {ctx['bar_ts'][:16]} UTC

{_regime_block(ctx)}

{pnl_line}

ACTION: Go FLAT — close your SOL position.
  Regime has exited the bull zone. Risk/reward no longer favourable.
  {"⚠️  BEAR regime confirmed. High-priority exit." if ctx['regime'] == "bear" else ""}
"""

def heartbeat_body(ctx, state):
    pos = "LONG (in SOL)" if state["in_strong_bull"] else "FLAT (no position)"
    entry_info = (f"  Entry price:     ${state['entry_price']:,.4f}"
                  if state["entry_price"] else "  Entry price:     —")
    entry_ts   = (f"  Entry bar:       {state['entry_bar_ts']}"
                  if state["entry_bar_ts"] else "")
    return f"""\
SOLREGIME — Daily Heartbeat
{_SEP}
Monitor is running normally.

POSITION: {pos}
{entry_info}
{entry_ts}

{_regime_block(ctx)}

Next LONG entry requires:
  Weekly RSI > 50    (now: {ctx['weekly_rsi']})
  Weekly close > EMA50  (now ${ctx['weekly_close']:,.2f} vs EMA50 ${ctx['weekly_ema50']:,.2f})
  Daily close > EMA50 > EMA200  (now: {ctx['daily_bull']})
"""


# ------------------------------------------------------------- core logic -----
def run_once(state):
    """Fetch data, classify regime, update state, send alerts if changed."""
    try:
        df = fetch_4h()
    except RuntimeError as e:
        log.error("Skipping cycle: %s", e)
        return

    regime, ctx = compute_regime(df)

    if regime is None:
        log.error("Regime compute failed: %s", ctx.get("error"))
        return

    bar_ts = ctx["bar_ts"]
    if state.get("last_bar_seen") == bar_ts:
        log.info("Bar already processed (%s) | regime=%s | price=$%.4f",
                 bar_ts[:16], regime, ctx["sol_price"])
        _maybe_heartbeat(state, ctx)
        return

    state["last_bar_seen"] = bar_ts
    prev_regime = state.get("last_regime")

    log.info("Bar %s | regime=%-12s | SOL=$%.4f | wRSI=%.1f | wEMA50=$%.2f | dEMA50=$%.2f dEMA200=$%.2f",
             bar_ts[:16], regime, ctx["sol_price"],
             ctx["weekly_rsi"], ctx["weekly_ema50"],
             ctx["daily_ema50"], ctx["daily_ema200"])

    regime_changed = (prev_regime is not None and prev_regime != regime)

    # NOTE: state["in_strong_bull"] is the persisted LONG flag. With
    # LONG_REGIMES = {"bull","strong_bull"} we are long on bull-or-better.
    # Transitions WITHIN long regimes (bull <-> strong_bull) keep us long.
    is_long = regime in LONG_REGIMES

    # ── BUY: entering a long regime
    if is_long and not state["in_strong_bull"]:
        state["in_strong_bull"] = True
        state["entry_price"]    = ctx["sol_price"]
        state["entry_bar_ts"]   = bar_ts
        log.info("ENTERING %s (LONG) — sending BUY alert", regime.upper())
        send_alert(f"🟢 SOLREGIME: SOL BUY — {regime.upper().replace('_',' ')} confirmed",
                   buy_body(ctx, prev_regime or "unknown"))

    # ── SELL: exiting to a non-long regime
    elif not is_long and state["in_strong_bull"]:
        ep = state.get("entry_price")
        state["in_strong_bull"] = False
        state["entry_price"]    = None
        state["entry_bar_ts"]   = None
        emojis = {"bear":"🔴","chop":"🟡","bull_chop":"🟡"}
        emoji  = emojis.get(regime, "⚪")
        log.info("EXITING LONG to %s — sending SELL alert", regime)
        send_alert(f"{emoji} SOLREGIME: SOL SELL — Exiting to {regime.upper().replace('_',' ')}",
                   sell_body(ctx, prev_regime or "long", ep))

    elif regime_changed:
        # regime changed between non-strong_bull states — log only
        log.info("Regime: %s -> %s (no trade signal)", prev_regime, regime)

    state["last_regime"] = regime
    save_state(state)
    _maybe_heartbeat(state, ctx)


def _maybe_heartbeat(state, ctx):
    """Send a heartbeat email every HEARTBEAT_H hours."""
    last_hb = state.get("last_heartbeat_ts")
    now_str  = dt.datetime.now(dt.timezone.utc).isoformat()
    if last_hb is None:
        do_hb = True
    else:
        elapsed = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(last_hb)
        do_hb = elapsed.total_seconds() > HEARTBEAT_H * 3600
    if do_hb:
        pos = "LONG" if state["in_strong_bull"] else "FLAT"
        log.info("Sending heartbeat | position=%s | regime=%s", pos, ctx["regime"])
        send_alert(f"💓 SOLREGIME: Heartbeat | {pos} | {ctx['regime'].upper()}",
                   heartbeat_body(ctx, state))
        state["last_heartbeat_ts"] = now_str
        save_state(state)


# ------------------------------------------------------------------- loop -----
def seconds_to_next_4h_bar():
    """Seconds until 60s after the next 4h boundary (00/04/08/12/16/20 UTC)."""
    now  = dt.datetime.now(dt.timezone.utc)
    h_in = now.hour % 4
    next_boundary = (
        now.replace(minute=0, second=0, microsecond=0)
        - dt.timedelta(hours=h_in)
        + dt.timedelta(hours=4)
    )
    return max(30, (next_boundary - now).total_seconds() + 65)


def main():
    log.info("SOLREGIME starting | %s | mode=%s",
             SYMBOL, "SHADOW" if SHADOW_MODE else "LIVE")
    state = load_state()
    log.info("State loaded: in_strong_bull=%s  last_regime=%s",
             state["in_strong_bull"], state["last_regime"])

    # Startup alert
    send_alert(
        f"SOLREGIME started | {'SHADOW' if SHADOW_MODE else 'LIVE'} | "
        f"{'LONG' if state['in_strong_bull'] else 'FLAT'}",
        f"SOLREGIME monitor is running.\n"
        f"Mode: {'SHADOW (no real trades)' if SHADOW_MODE else 'LIVE'}\n"
        f"Symbol: {SYMBOL} (4h regime check)\n"
        f"Current state: {'LONG (in SOL)' if state['in_strong_bull'] else 'FLAT'}\n"
        f"Last known regime: {state.get('last_regime') or 'unknown'}\n\n"
        f"Alerts: BUY on bull-regime entry, SELL on exit to non-bull.\n"
        f"Heartbeat every {HEARTBEAT_H}h.\n",
    )

    # Evaluate immediately on boot
    run_once(state)

    while True:
        sleep_s = seconds_to_next_4h_bar()
        log.info("Sleeping %.0f min until next 4h bar.", sleep_s / 60)
        time.sleep(sleep_s)
        try:
            run_once(state)
        except Exception as e:
            log.error("Unhandled error in loop: %s", e, exc_info=True)
            # Don't crash — retry next bar


if __name__ == "__main__":
    main()