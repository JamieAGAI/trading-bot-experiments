# Copyright (c) 2026 James Staples. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, distribution, or derivative works are prohibited.
# No license is granted for training machine learning models on this code.
import os
import json
import time
import re as _re
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from binance.client import Client
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText
from time import sleep

load_dotenv()

# ====================== CONFIG ======================
CHECK_INTERVAL    = 3600
MODEL             = "grok-4.20-beta-0309-reasoning"
SYMBOL            = "SOLUSDT"
DECISIONS_FILE    = "advisor_decisions.jsonl"
TREND_REPORT_FILE = "trend_reports.jsonl"

# How many 4H candles forward to measure outcome (12 = 2 days, 24 = 4 days)
OUTCOME_LOOKFORWARD_BARS = 12

# ── Shadow Perps Mode ────────────────────────────────────────────────────────
# Logs a virtual (fake) Jupiter Perps trade every hour using the exact same
# report data Advisor already generates.  Zero extra API calls.  Zero risk.
# Disable by setting to False.
SHADOW_PERPS_MODE  = True
SHADOW_PERPS_FILE  = "advisor_shadow_perps.jsonl"
# Max 4H bars to search forward when replaying path (48h = 12 bars)
SHADOW_LOOKFORWARD = 12

client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.x.ai/v1")
binance = Client(api_key=os.getenv("BINANCE_API_KEY"), api_secret=os.getenv("BINANCE_API_SECRET"))

EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO   = os.getenv("EMAIL_TO")
EMAIL_PASS = os.getenv("EMAIL_PASS")
ADVISOR_EMAIL_ENABLED = False  # Set True to receive advisor emails


# ====================== RETRY ======================
def with_retry(func, max_attempts=3, backoff=3):
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            err_str = str(e).lower()
            is_transient = any(k in err_str for k in ("connection", "timeout", "ssl", "rate limit"))
            if is_transient and attempt < max_attempts - 1:
                print(f"⚠️  Retry {attempt+1}/{max_attempts} in {backoff}s...")
                sleep(backoff)
                backoff *= 2
            else:
                print(f"❌ All retries failed: {e}")
                raise


# ====================== DATA ======================
def get_klines(interval, limit=400):
    """Fetch klines with automatic pagination for limits > 1000."""
    def _fetch():
        if limit <= 1000:
            klines = binance.futures_klines(symbol=SYMBOL, interval=interval, limit=limit)
        else:
            # Chunk into max 1000-bar requests, walking backwards from now
            chunks = []
            remaining = limit
            end_time = None
            while remaining > 0:
                chunk_limit = min(1000, remaining)
                params = {
                    "symbol": SYMBOL,
                    "interval": interval,
                    "limit": chunk_limit
                }
                if end_time is not None:
                    params["endTime"] = int(end_time.timestamp() * 1000) - 1
                raw = binance.futures_klines(**params)
                if not raw:
                    break
                chunks = raw + chunks  # prepend to maintain chronological order
                # Set endTime to just before the oldest bar we just got
                end_time = pd.to_datetime(raw[0][0], unit="ms", utc=True)
                remaining -= len(raw)
            klines = chunks[-limit:]  # take the most recent 'limit' bars

        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'n', 'tbav', 'tbqav', 'ignore'
        ])
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        return df.set_index('open_time')

    return with_retry(_fetch)


def get_perp_data():
    def _fetch():
        ticker         = binance.futures_ticker(symbol=SYMBOL)
        current_price  = float(ticker['lastPrice'])
        funding        = binance.futures_funding_rate(symbol=SYMBOL, limit=3)
        rates          = [float(f['fundingRate']) * 100 for f in funding]
        avg_funding    = round(sum(rates) / len(rates), 4)
        latest_funding = rates[0]
        oi             = binance.futures_open_interest(symbol=SYMBOL)
        oi_value       = float(oi['openInterest']) * current_price / 1_000_000
        return {
            "funding_rate_pct":    round(latest_funding, 4),
            "funding_avg_3period": avg_funding,
            "open_interest_m":     round(oi_value, 1),
        }
    return with_retry(_fetch)


# ====================== TA FUNCTIONS ======================
def calculate_rsi(df, period=14):
    if len(df) < period + 10:
        return 50.0
    delta    = df['close'].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    rsi      = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 1)


def detect_rsi_divergence(df, lookback=25):
    if len(df) < lookback + 14:
        return "none"
    close  = df['close'].iloc[-lookback:].reset_index(drop=True)
    volume = df['volume'].iloc[-lookback:].reset_index(drop=True)
    delta    = df['close'].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs       = avg_gain / avg_loss
    rsi_full = 100 - (100 / (1 + rs))
    rsi      = rsi_full.iloc[-lookback:].reset_index(drop=True)

    def find_swing_lows(series, window=3):
        lows = []
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i - window:i + window + 1].min():
                lows.append((i, float(series.iloc[i])))
        deduped = []
        for idx, val in lows:
            if deduped and idx - deduped[-1][0] <= window:
                if val < deduped[-1][1]:
                    deduped[-1] = (idx, val)
            else:
                deduped.append((idx, val))
        return deduped

    def find_swing_highs(series, window=3):
        highs = []
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i - window:i + window + 1].max():
                highs.append((i, float(series.iloc[i])))
        deduped = []
        for idx, val in highs:
            if deduped and idx - deduped[-1][0] <= window:
                if val > deduped[-1][1]:
                    deduped[-1] = (idx, val)
            else:
                deduped.append((idx, val))
        return deduped

    p_lows  = find_swing_lows(close)
    p_highs = find_swing_highs(close)
    r_lows  = find_swing_lows(rsi)
    r_highs = find_swing_highs(rsi)

    if len(p_lows) >= 2 and len(r_lows) >= 2:
        pl1, pl2 = p_lows[-2], p_lows[-1]
        rl1, rl2 = r_lows[-2], r_lows[-1]
        vol_ok   = volume.iloc[pl2[0]] > volume.iloc[pl1[0]]
        if pl2[1] < pl1[1] and rl2[1] > rl1[1] and vol_ok:
            return "regular_bullish"
        if pl2[1] > pl1[1] and rl2[1] < rl1[1] and vol_ok:
            return "hidden_bullish"

    if len(p_highs) >= 2 and len(r_highs) >= 2:
        ph1, ph2 = p_highs[-2], p_highs[-1]
        rh1, rh2 = r_highs[-2], r_highs[-1]
        vol_ok   = volume.iloc[ph2[0]] > volume.iloc[ph1[0]]
        if ph2[1] > ph1[1] and rh2[1] < rh1[1] and vol_ok:
            return "regular_bearish"
        if ph2[1] < ph1[1] and rh2[1] > rh1[1] and vol_ok:
            return "hidden_bearish"

    return "none"


def calculate_macd(df):
    ema12     = df['close'].ewm(span=12, adjust=False).mean()
    ema26     = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal    = macd_line.ewm(span=9, adjust=False).mean()
    hist      = macd_line - signal
    last, prev         = macd_line.iloc[-1], macd_line.iloc[-2]
    sig_last, sig_prev = signal.iloc[-1], signal.iloc[-2]
    if last > sig_last and prev <= sig_prev:
        cross = "bullish_cross"
    elif last < sig_last and prev >= sig_prev:
        cross = "bearish_cross"
    else:
        cross = "no_cross"
    improving = hist.iloc[-1] > hist.iloc[-2]
    return f"{cross} | hist {hist.iloc[-1]:.4f} ({'improving' if improving else 'deteriorating'})"


def calculate_ema_status(df):
    ema50  = df['close'].ewm(span=50,  adjust=False).mean().iloc[-1]
    ema200 = df['close'].ewm(span=200, adjust=False).mean().iloc[-1]
    price  = df['close'].iloc[-1]
    if price > ema50 > ema200:
        return "above_both_bullish"
    elif price < ema50 < ema200:
        return "below_both_bearish"
    elif price > ema50 and ema50 < ema200:
        return "above_ema50_mixed"
    else:
        return "mixed"


def calculate_atr(df, period=14):
    high_low   = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close  = np.abs(df['low']  - df['close'].shift())
    tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return round(tr.rolling(window=period).mean().iloc[-1], 2)


def get_fib_levels(df):
    recent  = df.tail(120)
    high    = recent['high'].max()
    low     = recent['low'].min()
    diff    = high - low
    levels  = {
        "0.000": round(low, 2),
        "0.236": round(low + 0.236 * diff, 2),
        "0.382": round(low + 0.382 * diff, 2),
        "0.500": round(low + 0.500 * diff, 2),
        "0.618": round(low + 0.618 * diff, 2),
        "0.786": round(low + 0.786 * diff, 2),
        "1.000": round(high, 2),
    }
    price   = df['close'].iloc[-1]
    closest = min(levels, key=lambda k: abs(levels[k] - price))
    fib_str = " | ".join(f"{k}=${v}" for k, v in levels.items()
                         if k in ("0.236", "0.382", "0.500", "0.618", "0.786"))
    return f"nearest={closest} (${levels[closest]:.2f}) | {fib_str}", levels


def get_macro_regime_from_data(df_daily, df_weekly):
    try:
        w_price  = df_weekly['close'].iloc[-1]
        w_ema50  = df_weekly['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        w_rsi    = calculate_rsi(df_weekly)
        d_price  = df_daily['close'].iloc[-1]
        d_ema50  = df_daily['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        d_ema200 = df_daily['close'].ewm(span=200, adjust=False).mean().iloc[-1]
        if w_price < w_ema50 and w_rsi < 50 and d_price < d_ema50 < d_ema200:
            return "bear"
        elif w_price > w_ema50 and w_rsi > 55 and d_price > d_ema50 > d_ema200:
            return "bull"
        else:
            return "chop"
    except Exception:
        return "unknown"


def build_ta_summary(df_4h, df_1h, df_daily, df_weekly):
    fib_label_4h, fib_levels_4h = get_fib_levels(df_4h)
    fib_label_1h, _             = get_fib_levels(df_1h)

    rsi_4h       = calculate_rsi(df_4h)
    rsi_1h       = calculate_rsi(df_1h)
    div_4h       = detect_rsi_divergence(df_4h, lookback=25)
    div_1h       = detect_rsi_divergence(df_1h, lookback=18)
    macd_4h      = calculate_macd(df_4h)
    ema_4h       = calculate_ema_status(df_4h)
    atr_4h       = calculate_atr(df_4h)
    vol_4h       = float(df_4h['volume'].iloc[-2]) if len(df_4h) >= 2 else 0.0
    vol_4h_ma20  = float(df_4h['volume'].iloc[:-1].rolling(20, min_periods=1).mean().iloc[-1]) if len(df_4h) >= 2 else 0.0
    vol_ratio_4h = round(vol_4h / vol_4h_ma20, 3) if vol_4h_ma20 > 0 else 1.0
    vol_1h       = float(df_1h['volume'].iloc[-2]) if len(df_1h) >= 2 else 0.0
    vol_1h_ma20  = float(df_1h['volume'].iloc[:-1].rolling(20, min_periods=1).mean().iloc[-1]) if len(df_1h) >= 2 else 0.0
    vol_ratio_1h = round(vol_1h / vol_1h_ma20, 3) if vol_1h_ma20 > 0 else 1.0

    price        = df_4h['close'].iloc[-1]
    d_rsi        = calculate_rsi(df_daily)
    d_ema50      = round(df_daily['close'].ewm(span=50,  adjust=False).mean().iloc[-1], 2)
    d_ema200     = round(df_daily['close'].ewm(span=200, adjust=False).mean().iloc[-1], 2)
    d_price      = round(df_daily['close'].iloc[-1], 2)
    w_rsi        = calculate_rsi(df_weekly)
    local_regime = get_macro_regime_from_data(df_daily, df_weekly)

    summary = (
        f"Local Regime (TA): {local_regime.upper()}\n"
        f"Weekly RSI: {w_rsi} | Daily RSI: {d_rsi} | "
        f"Daily EMA50: ${d_ema50} | Daily EMA200: ${d_ema200} | Daily Price: ${d_price}\n"
        f"4H: RSI={rsi_4h} | Div={div_4h} | MACD={macd_4h} | EMA={ema_4h} | {fib_label_4h}\n"
        f"1H: RSI={rsi_1h} | Div={div_1h} | {fib_label_1h}\n"
        f"ATR(14)=${atr_4h}"
    )
    return summary, rsi_4h, rsi_1h, div_4h, div_1h, atr_4h, price, local_regime, fib_label_4h, ema_4h, macd_4h, vol_ratio_4h, vol_ratio_1h, vol_4h


# ====================== PHASE 1: SIGNAL OUTCOME TRACKER ======================
def tag_outcomes_in_decisions(df_4h_full):
    """
    Reads advisor_decisions.jsonl, finds entries that don't yet have outcome data,
    looks up what price actually did in multiple forward windows (12h, 24h, 36h, 48h),
    and writes the outcome back to the file.

    Existing field names (outcome_high_pct, outcome_low_pct, outcome_close_pct,
    outcome_tagged) represent the 48h window for back-compat. New suffixed fields:
    _12h, _24h, _36h.
    """
    if not os.path.exists(DECISIONS_FILE):
        return

    # Forward windows in 4H bars: 3 bars = 12h, 6 = 24h, 9 = 36h, 12 = 48h
    WINDOWS = {"12h": 3, "24h": 6, "36h": 9, "48h": 12}

    try:
        with open(DECISIONS_FILE, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            return

        records  = [json.loads(l) for l in lines]
        modified = False
        newly_tagged = 0

        for rec in records:
            # Skip records already fully populated with all 4 windows
            already_full = (
                rec.get("outcome_tagged") and
                "outcome_high_pct_12h" in rec and
                "outcome_high_pct_24h" in rec and
                "outcome_high_pct_36h" in rec
            )
            if already_full:
                continue

            sig_time_str = rec.get("timestamp")
            if not sig_time_str:
                continue

            # Robust tz + unit handling
            sig_time = pd.Timestamp(sig_time_str)
            if sig_time.tzinfo is None:
                sig_time = sig_time.tz_localize("UTC")
            else:
                sig_time = sig_time.tz_convert("UTC")

            # Match resolution to Binance index (ms) to prevent unit error
            sig_time = sig_time.as_unit('ms')
            idx_arr = df_4h_full.index.searchsorted(sig_time)

            if idx_arr < 1:
                continue

            fwd_start = idx_arr

            entry_price = rec.get("price")
            if not entry_price or entry_price <= 0:
                continue

            # Populate each window if we have enough forward data
            rec_modified = False
            for label, bars in WINDOWS.items():
                end = fwd_start + bars
                if end > len(df_4h_full):
                    continue  # not enough forward data yet for this window
                field_suffix = "" if label == "48h" else f"_{label}"
                # Skip this window if already populated
                if f"outcome_high_pct{field_suffix}" in rec:
                    continue
                fwd_slice = df_4h_full.iloc[fwd_start:end]
                high_pct  = round((fwd_slice['high'].max()  - entry_price) / entry_price * 100, 3)
                low_pct   = round((fwd_slice['low'].min()   - entry_price) / entry_price * 100, 3)
                close_pct = round((fwd_slice['close'].iloc[-1] - entry_price) / entry_price * 100, 3)
                rec[f"outcome_high_pct{field_suffix}"]  = high_pct
                rec[f"outcome_low_pct{field_suffix}"]   = low_pct
                rec[f"outcome_close_pct{field_suffix}"] = close_pct
                rec_modified = True

            # Mark outcome_tagged=True only when the 48h window is filled
            if "outcome_close_pct" in rec and not rec.get("outcome_tagged"):
                rec["outcome_tagged"] = True
                newly_tagged += 1
                rec_modified = True

            if rec_modified:
                modified = True

        if modified:
            with open(DECISIONS_FILE, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
            tagged_count = sum(1 for r in records if r.get("outcome_tagged"))
            print(f"   📊 Outcome tagging: {tagged_count} total tagged (48h), {newly_tagged} newly tagged this run")

    except Exception as e:
        print(f"   ⚠️  Outcome tagging error: {e}")


def get_signal_performance_summary():
    """
    Reads tagged outcomes from advisor_decisions.jsonl and computes
    win-rate statistics broken down by regime and bias.

    A signal is a 'win' if:
      - LONG:  outcome_high_pct  >= +1.5% (price moved up enough to profit)
      - SHORT: outcome_low_pct   <= -1.5% (price moved down enough to profit)
      - NEUTRAL: excluded from win/loss stats

    Returns a compact string injected into the AI prompt.
    """
    if not os.path.exists(DECISIONS_FILE):
        return "No historical signals yet — building track record."

    try:
        with open(DECISIONS_FILE, "r") as f:
            records = [json.loads(l) for l in f if l.strip()]

        tagged = [r for r in records if r.get("outcome_tagged")]
        if len(tagged) < 8:
            return f"Track record building ({len(tagged)} outcomes so far — need 8+ for stats)."

        # ── Per-regime, per-bias stats ──────────────────────────────────────
        stats = {}  # key: (regime, bias), value: {"wins": n, "total": n, "avg_close": []}

        for r in tagged:
            regime = r.get("local_regime", "unknown")
            bias   = r.get("bias", "neutral").lower()
            if bias == "neutral":
                continue

            high_pct  = r.get("outcome_high_pct", 0)
            low_pct   = r.get("outcome_low_pct",  0)
            close_pct = r.get("outcome_close_pct", 0)

            if bias == "long":
                win = high_pct >= 1.5
            elif bias == "short":
                win = low_pct <= -1.5
            else:
                continue

            key = (regime, bias)
            if key not in stats:
                stats[key] = {"wins": 0, "total": 0, "avg_close": []}
            stats[key]["wins"]      += int(win)
            stats[key]["total"]     += 1
            stats[key]["avg_close"].append(close_pct)

        if not stats:
            return "No directional signals with outcomes yet."

        lines = []
        for (regime, bias), s in sorted(stats.items()):
            wr       = round(s["wins"] / s["total"] * 100)
            avg_cls  = round(sum(s["avg_close"]) / len(s["avg_close"]), 2)
            n        = s["total"]
            quality  = "✅" if wr >= 55 else "⚠️" if wr >= 45 else "❌"
            lines.append(
                f"  {quality} {regime.upper()} {bias.upper()}: "
                f"{n} signals | WR {wr}% | avg net {avg_cls:+.2f}% over {OUTCOME_LOOKFORWARD_BARS*4}h"
            )

        total_dir = sum(s["total"] for s in stats.values())
        total_win = sum(s["wins"]  for s in stats.values())
        overall   = round(total_win / total_dir * 100) if total_dir else 0

        return (
            f"Signal track record ({total_dir} directional, {OUTCOME_LOOKFORWARD_BARS*4}h window):\n"
            + "\n".join(lines)
            + f"\n  Overall directional WR: {overall}%"
        )

    except Exception as e:
        return f"Performance summary error: {e}"


# ====================== PHASE 2: HISTORICAL EDGE CALCULATOR ======================
def _get_fast_regime(df_slice):
    """
    Lightweight regime check used inside the backtest loop — avoids
    loading weekly/daily data for every historical bar.
    Uses the 4H slice itself with a 200-bar EMA proxy.
    """
    try:
        if len(df_slice) < 50:
            return "unknown"
        price   = df_slice['close'].iloc[-1]
        ema50   = df_slice['close'].ewm(span=50,  adjust=False).mean().iloc[-1]
        # Use a shorter RSI proxy (14-period on 4H ≈ 56h)
        rsi_val = calculate_rsi(df_slice)
        if price < ema50 and rsi_val < 50:
            return "bear"
        elif price > ema50 and rsi_val > 55:
            return "bull"
        else:
            return "chop"
    except Exception:
        return "unknown"


def get_historical_edge(df_4h_long, current_regime, div_4h, macd_4h, ema_4h, atr_now):
    """
    Scans df_4h_long (2000 bars ≈ 11 months) for past bars where the same
    signal combination was present, then measures forward price action.

    Matching criteria:
      - Same regime (bear/bull/chop)
      - Same divergence type on 4H
      - Same MACD direction (bullish/bearish/none)
      - Same EMA stack

    Forward measurement (OUTCOME_LOOKFORWARD_BARS):
      - long_win:  max gain ≥ 1.5% within window
      - short_win: max drop ≤ -1.5% within window
      - Avg max gain and avg max drawdown across all matches

    Returns a formatted string for the prompt, or a short "insufficient data" note.
    """
    if df_4h_long is None or len(df_4h_long) < 200:
        return "Historical edge: insufficient data (need 200+ bars)."

    # Extract MACD direction from string (bullish_cross / bearish_cross / no_cross)
    if "bullish_cross" in macd_4h:
        macd_dir = "bullish_cross"
    elif "bearish_cross" in macd_4h:
        macd_dir = "bearish_cross"
    else:
        macd_dir = "no_cross"

    # Whether MACD hist is improving
    macd_improving = "improving" in macd_4h

    lookback  = 100   # bars used to compute TA state at each historical point
    fwd       = OUTCOME_LOOKFORWARD_BARS
    matches   = []
    start_idx = max(lookback, 200)  # need enough history for EMA200
    end_idx   = len(df_4h_long) - fwd  # leave room for forward window

    for i in range(start_idx, end_idx):
        try:
            past_slice = df_4h_long.iloc[i - lookback: i + 1].copy()
            if len(past_slice) < lookback:
                continue

            # Fast regime proxy on the 4H slice
            past_regime = _get_fast_regime(past_slice)
            if past_regime != current_regime:
                continue

            past_div  = detect_rsi_divergence(past_slice, lookback=25)
            past_macd = calculate_macd(past_slice)
            past_ema  = calculate_ema_status(past_slice)

            # MACD direction match
            if "bullish_cross" in past_macd:
                past_macd_dir = "bullish_cross"
            elif "bearish_cross" in past_macd:
                past_macd_dir = "bearish_cross"
            else:
                past_macd_dir = "no_cross"

            past_macd_improving = "improving" in past_macd

            # Match: regime + divergence type + EMA stack + MACD direction
            # (regime already filtered above)
            div_match    = past_div  == div_4h
            ema_match    = past_ema  == ema_4h
            macd_match   = past_macd_dir == macd_dir and past_macd_improving == macd_improving

            if not (div_match and ema_match and macd_match):
                continue

            # Forward window
            entry_price = past_slice['close'].iloc[-1]
            fwd_slice   = df_4h_long.iloc[i + 1: i + 1 + fwd]
            if len(fwd_slice) < fwd:
                continue

            max_up   = (fwd_slice['high'].max()  - entry_price) / entry_price * 100
            max_down = (fwd_slice['low'].min()   - entry_price) / entry_price * 100
            net_cls  = (fwd_slice['close'].iloc[-1] - entry_price) / entry_price * 100

            matches.append({
                "max_up":   round(max_up,   2),
                "max_down": round(max_down, 2),
                "net_cls":  round(net_cls,  2),
            })

        except Exception:
            continue

    if not matches:
        return (
            f"Historical edge: no exact matches for {current_regime.upper()} + "
            f"div={div_4h} + macd={macd_dir} + ema={ema_4h} in last ~11 months. "
            f"No base rate available — rely on current TA signals only."
        )

    n          = len(matches)
    long_wins  = sum(1 for m in matches if m["max_up"]   >=  1.5)
    short_wins = sum(1 for m in matches if m["max_down"] <= -1.5)
    avg_up     = round(sum(m["max_up"]   for m in matches) / n, 2)
    avg_down   = round(sum(m["max_down"] for m in matches) / n, 2)
    avg_net    = round(sum(m["net_cls"]  for m in matches) / n, 2)
    long_wr    = round(long_wins  / n * 100)
    short_wr   = round(short_wins / n * 100)

    # Confidence signal — is the base rate meaningful?
    edge_note = ""
    if long_wr >= 60:
        edge_note = " ← LONG edge confirmed historically"
    elif short_wr >= 60:
        edge_note = " ← SHORT edge confirmed historically"
    elif long_wr <= 35:
        edge_note = " ← historically weak for longs"
    elif short_wr <= 35:
        edge_note = " ← historically weak for shorts"

    return (
        f"Historical edge ({n} matches | {current_regime.upper()} + div={div_4h} + "
        f"macd={macd_dir} + ema={ema_4h} | fwd={fwd * 4}h):\n"
        f"  LONG win rate:  {long_wr}% ({long_wins}/{n} moved ≥+1.5%){edge_note}\n"
        f"  SHORT win rate: {short_wr}% ({short_wins}/{n} dropped ≤-1.5%)\n"
        f"  Avg max gain: +{avg_up}% | Avg max drawdown: {avg_down}% | Avg net: {avg_net:+.2f}%"
    )


# ====================== TREND AGENT CONTEXT ======================
def load_latest_trend_report():
    if not os.path.exists(TREND_REPORT_FILE):
        return "Trend report unavailable — run trendagent.py for macro context."
    try:
        with open(TREND_REPORT_FILE, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return "Trend report empty."
        report = json.loads(lines[-1])
        return (
            f"Macro Regime: {report.get('regime', '?')} | "
            f"Confidence: {report.get('confidence', '?')}/10\n"
            f"Summary: {report.get('summary', '')}\n"
            f"1M Outlook: {report.get('outlook_1m', '')}\n"
            f"Bull Invalidation: {report.get('bull_invalidation', '?')} | "
            f"Bear Invalidation: {report.get('bear_invalidation', '?')}\n"
            f"Recommendation: {report.get('recommendation', '')}\n"
            f"Report timestamp: {report.get('timestamp', '?')}"
        )
    except Exception as e:
        return f"Trend report load error: {e}"


# ====================== REGIME GATE ======================
def build_regime_gate(local_regime, trend_context):
    is_macro_bear = "bear" in trend_context.lower()

    if local_regime == "bear" or is_macro_bear:
        return """🔴 REGIME GATE: BEAR MARKET
The macro trend agent and/or local TA confirms a bear market regime.

LONG TRADES:
  Counter-trend bounces only. Require ALL THREE: confirmed regular_bullish divergence on 4H,
  MACD bullish cross (not just histogram improving), AND price at major Fib support.
  Confidence cap: 6.5/10 maximum. Targets: next Fib only.

SHORT TRADES:
  This is the primary direction. Valid setups:
  • hidden_bearish or regular_bearish divergence on 4H
  • Price failing at resistance (EMA50, EMA200, key Fib) with bearish candle
  • MACD bearish cross or deteriorating histogram after a failed bounce
  • Positive funding after a relief rally = flush risk
  Shorts in bear regime get full confidence scoring.

FUNDING IN BEAR: Persistent negative funding = trend confirmation, NOT squeeze setup.
  Only treat as long signal if RSI <30 AND divergence confirmed simultaneously."""

    elif local_regime == "chop":
        return """🟡 REGIME GATE: CHOP / TRANSITIONING
Both directions valid but require strict confluence.

LONG: confirmed bullish divergence 4H + MACD cross/improving 3+ bars + major Fib support. Target = next Fib only.
SHORT: confirmed bearish divergence 4H + MACD bearish / deteriorating + failing at resistance. Target = next Fib down.
FUNDING: persistent neg = leans short but weak. Persistent pos = leans long but weak. Neither alone is enough."""

    else:
        return """🟢 REGIME GATE: BULL / TRANSITIONING BULLISH
Favorable for longs. Standard confluence rules. Targets can extend 2–3 Fib levels.
Shorts only on clear regular_bearish divergence + EMA rejection + positive funding.
FUNDING: persistent pos = crowded longs = caution. Persistent neg in bull = dip is buyable."""


# ====================== EMAIL ======================
def send_advisor_email(decision, price, ta_summary, perp, trend_context,
                       signal_perf, historical_edge):
    if not ADVISOR_EMAIL_ENABLED:
        return
    if not all([EMAIL_FROM, EMAIL_TO, EMAIL_PASS]):
        return
    bias      = decision.get('bias', 'neutral').upper()
    emoji     = "🟢" if bias == "LONG" else "🔴" if bias == "SHORT" else "⚪"
    price_str = f"${price:.2f}" if price else "N/A"
    subject   = f"SOL Advisor — {emoji} {bias} | Conf {decision.get('confidence',0)}/10 @ {price_str}"

    body = f"""SOL Perp Bias Advisor (Jupiter) — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC

{emoji} BIAS: {bias} | Confidence: {decision.get('confidence', 0)}/10
Regime alignment: {decision.get('regime_alignment', 'N/A')}
Long score: {decision.get('long_score','?')} | Short score: {decision.get('short_score','?')}

📝 {decision.get('reason', '')}

🎯 Entry:        {decision.get('entry_note', '')}
📍 Leverage:     {decision.get('suggested_leverage', '')}
🛑 Stop:         {decision.get('stop_loss', '')}
🎯 TP:           {decision.get('take_profit', '')}
❌ Invalidation: {decision.get('invalidation', '')}
⚠️  Risks:       {' | '.join(decision.get('key_risks', []))}

=== HISTORICAL EDGE (this exact setup) ===
{historical_edge}

=== SIGNAL TRACK RECORD ===
{signal_perf}

=== MACRO CONTEXT (Trend Agent) ===
{trend_context}

=== TA DASHBOARD ===
{ta_summary}

=== PERPS ===
Funding: {perp.get('funding_rate_pct', 0):+.4f}%/8h (avg 3p: {perp.get('funding_avg_3period',0):+.4f}%) | OI: ${perp.get('open_interest_m', 0)}M
"""
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From']    = EMAIL_FROM
        msg['To']      = EMAIL_TO
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_FROM, EMAIL_PASS)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print("📧 Email sent")
    except Exception as e:
        print(f"Email failed: {e}")


# ====================== AI DECISION ======================
def get_ai_decision(ta_summary, perp, price, trend_context, regime_gate,
                    local_regime, signal_perf, historical_edge):
    def _call():
        avg_f          = perp.get('funding_avg_3period', perp['funding_rate_pct'])
        latest_f       = perp['funding_rate_pct']
        persistent_neg = avg_f < -0.01
        persistent_pos = avg_f > 0.01

        if local_regime == "bear":
            if persistent_neg:
                funding_note = (
                    f"Funding avg (3 periods): {avg_f:+.4f}% — PERSISTENTLY NEGATIVE in bear. "
                    f"Trend confirmation, NOT a squeeze setup. "
                    f"Only a contrarian long signal if RSI<30 + confirmed divergence simultaneously."
                )
            elif persistent_pos:
                funding_note = (
                    f"Funding avg (3 periods): {avg_f:+.4f}% — POSITIVE after bear move. "
                    f"Crowded longs on relief rally = flush risk. Reinforces short bias."
                )
            else:
                funding_note = f"Funding: {latest_f:+.4f}% (mixed — not persistent, low weight)"
        else:
            if persistent_neg:
                funding_note = (
                    f"Funding avg (3 periods): {avg_f:+.4f}% — persistent negative = crowded shorts. "
                    f"Squeeze risk if price moves up with confluence."
                )
            elif persistent_pos:
                funding_note = (
                    f"Funding avg (3 periods): {avg_f:+.4f}% — persistent positive = crowded longs. "
                    f"Flush risk. Slight short lean or caution on new longs."
                )
            else:
                funding_note = f"Funding: {latest_f:+.4f}% (neutral — low weight)"

        prompt = f"""You are a disciplined SOL perpetual futures bias advisor for Jupiter Perps.
Give a clear Long, Short, or Neutral bias that a trader can act on confidently.
You are fully bidirectional — evaluate both directions with equal rigour every run.

Current price: ${price:.2f}

{regime_gate}

=== MACRO CONTEXT (Trend Agent — highest priority) ===
{trend_context}

=== HISTORICAL EDGE (this exact signal combination on 4H — deterministic Python backtest) ===
{historical_edge}
Use this to calibrate confidence: if historical long WR is 65%+ for this setup, lean toward long.
If historical short WR is 65%+ and current bias is long, weight that contradiction seriously.
If no matches exist, rely on current TA signals only.

=== SIGNAL TRACK RECORD (your past calls on this system) ===
{signal_perf}
If your LONG win rate in BEAR regime is below 40%, treat that as a structural warning against new longs.

=== TA DASHBOARD ===
{ta_summary}

=== PERP DATA ===
{funding_note}
Open Interest: ${perp['open_interest_m']}M

=== SIGNAL SCORING ===
Start both long and short scores at 5.0. Add and subtract below.

LONG score additions:
  +2.5  regular_bullish divergence confirmed on 4H (price lower low + RSI higher low, volume confirmed)
  +2.0  hidden_bullish divergence confirmed on 4H
  +1.5  bullish divergence confirmed on 1H as well (multi-TF)
  +1.5  MACD bullish CROSS on 4H (not just histogram improving)
  +0.75 MACD histogram improving 3+ consecutive bars
  +1.0  Price above daily EMA50
  +1.0  Persistent negative funding + RSI <35 in neutral/bull regime only
  +0.5  1H RSI reclaiming 50 from below

LONG penalties:
  -3.0  Bear regime confirmed (trendagent or local TA) — hard cap long score at 6.0
  -2.0  Below daily EMA50 AND EMA200 with no confirmed divergence
  -1.5  MACD bearish cross or histogram deteriorating
  -1.0  No divergence on 4H at all

SHORT score additions:
  +2.5  regular_bearish divergence confirmed on 4H (price higher high + RSI lower high, volume confirmed)
  +2.0  hidden_bearish divergence confirmed on 4H
  +1.5  bearish divergence confirmed on 1H as well
  +1.5  MACD bearish CROSS on 4H
  +0.75 MACD histogram deteriorating 3+ consecutive bars
  +1.0  Price below daily EMA50 AND EMA200 (confirmed structure)
  +1.0  Persistent positive funding after a rally
  +0.5  Price failing at EMA50/EMA200/major Fib with bearish candle

SHORT penalties:
  -2.5  Bull regime confirmed
  -1.5  RSI below 30 (oversold — poor R:R for new shorts)
  -1.0  No divergence on 4H at all

THRESHOLDS:
  ≥7.5  → output that direction, high confidence
  6.5–7.4 → output that direction, moderate confidence
  5.5–6.4 → Neutral — say which direction is closer and what one signal would flip it
  ≤5.4  → Neutral

RULES:
  - Only count signals clearly present NOW. "Possibly forming" = zero points.
  - If historical edge shows <40% WR for a direction, require score ≥7.5 to output it.
  - Confidence 8+ requires multi-TF confirmation. Never output 9+ unless exceptionally clean.
  - ATR stops: 1.5–2.0x. Targets: 2.5–3.5x. In bear/chop cap targets at next Fib only.
  - Add invalidation level to every signal — the exact price that proves you wrong.

Output ONLY valid JSON — no preamble, no markdown:
{{
  "bias": "long | short | neutral",
  "confidence": 6.5,
  "long_score": 5.5,
  "short_score": 7.0,
  "regime_alignment": "short aligned with bear | long counter-trend bounce | neutral",
  "reason": "2-3 sentences. Name the specific signals. No vague statements.",
  "entry_note": "Specific level or condition on Jupiter. Actionable.",
  "suggested_leverage": "3x",
  "stop_loss": "$XX.XX — why (e.g. above 4H swing high / 1.5x ATR)",
  "take_profit": "primary $XX.XX | stretch $XX.XX",
  "invalidation": "Exact price or signal that proves this bias wrong",
  "key_risks": ["risk1", "risk2"]
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    return with_retry(_call)


# ====================== SHADOW PERPS MODE ======================
def _extract_price(text):
    """
    Pull the first bare dollar amount from a string like '$84.50 — above 4H swing high'.
    Returns float or None.  Handles: $84.50  84.50  $84  84
    """
    if not text:
        return None
    try:
        m = _re.search(r'\$?([\d]+\.[\d]+|[\d]+)', str(text))
        if m:
            val = float(m.group(1))
            # Sanity-check: must look like a SOL price (between $1 and $10,000)
            if 1.0 <= val <= 10_000.0:
                return round(val, 4)
    except Exception:
        pass
    return None


def _extract_leverage(text):
    """
    Pull leverage integer from strings like '3x', '5x leverage', '3–5x'.
    Returns int or None.
    """
    if not text:
        return None
    try:
        m = _re.search(r'(\d+)x', str(text))
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def _extract_tp_prices(take_profit_str):
    """
    Parse 'primary $XX.XX | stretch $XX.XX' into (primary_price, stretch_price).
    Returns (float|None, float|None).
    """
    primary = stretch = None
    if not take_profit_str:
        return primary, stretch
    try:
        parts = str(take_profit_str).lower().split("|")
        for part in parts:
            val = _extract_price(part)
            if val is None:
                continue
            if "stretch" in part:
                stretch = val
            elif "primary" in part or primary is None:
                primary = val
    except Exception:
        pass
    return primary, stretch


def log_shadow_perp(decision, price, local_regime, perp,
                    rsi_4h, div_4h, macd_4h, ema_4h, atr_4h,
                    vol_ratio_4h=1.0, vol_ratio_1h=1.0, vol_4h=0.0):
    """
    Write one virtual perps trade record to SHADOW_PERPS_FILE.
    Uses the exact levels the AI report already calculated — no new API calls.
    The record is later updated by tag_shadow_perp_outcomes().
    """
    if not SHADOW_PERPS_MODE:
        return

    bias = decision.get("bias", "neutral").lower()
    if bias == "neutral":
        # No directional trade to shadow
        return

    leverage       = _extract_leverage(decision.get("suggested_leverage", ""))
    stop_price     = _extract_price(decision.get("stop_loss", ""))
    tp_primary, tp_stretch = _extract_tp_prices(decision.get("take_profit", ""))
    invalidation   = _extract_price(decision.get("invalidation", ""))

    # Risk % per unit at entry (for later P&L calculation)
    risk_pct  = None
    reward_pct_primary = None
    reward_pct_stretch = None
    rr_primary = None

    if stop_price and price and price > 0:
        if bias == "long":
            risk_pct = round((price - stop_price) / price * 100, 3) if stop_price < price else None
        else:  # short
            risk_pct = round((stop_price - price) / price * 100, 3) if stop_price > price else None

    if tp_primary and price and price > 0:
        if bias == "long":
            reward_pct_primary = round((tp_primary - price) / price * 100, 3)
        else:
            reward_pct_primary = round((price - tp_primary) / price * 100, 3)

    if tp_stretch and price and price > 0:
        if bias == "long":
            reward_pct_stretch = round((tp_stretch - price) / price * 100, 3)
        else:
            reward_pct_stretch = round((price - tp_stretch) / price * 100, 3)

    if risk_pct and risk_pct > 0 and reward_pct_primary:
        rr_primary = round(reward_pct_primary / risk_pct, 2)

    rec = {
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "price":                round(price, 4),
        "bias":                 bias,
        "confidence":           decision.get("confidence", 0),
        "local_regime":         local_regime,
        # Levels from the report
        "leverage":             leverage,
        "stop_price":           stop_price,
        "tp_primary":           tp_primary,
        "tp_stretch":           tp_stretch,
        "invalidation_price":   invalidation,
        # Pre-calculated R metrics at entry
        "risk_pct":             risk_pct,
        "reward_pct_primary":   reward_pct_primary,
        "reward_pct_stretch":   reward_pct_stretch,
        "rr_primary":           rr_primary,
        # Market context at signal time
        "funding_rate":         perp.get("funding_rate_pct", 0),
        "funding_avg_3period":  perp.get("funding_avg_3period", 0),
        "open_interest_m":      perp.get("open_interest_m", 0),
        "rsi_4h":               round(rsi_4h, 1),
        "div_4h":               div_4h,
        "macd_4h":              macd_4h,
        "ema_4h":               ema_4h,
        "atr_4h":               round(atr_4h, 4),
        "vol_ratio_4h":         round(vol_ratio_4h, 3),
        "vol_ratio_1h":         round(vol_ratio_1h, 3),
        "vol_4h_raw":           round(vol_4h, 2),
        # Raw strings for reference
        "raw_stop_str":         decision.get("stop_loss", ""),
        "raw_tp_str":           decision.get("take_profit", ""),
        "raw_leverage_str":     decision.get("suggested_leverage", ""),
        # Outcome fields — filled later by tag_shadow_perp_outcomes()
        "outcome_tagged":       False,
        "outcome_result":       None,   # "tp_primary" | "tp_stretch" | "stop" | "timeout"
        "outcome_exit_price":   None,
        "outcome_hold_bars":    None,   # number of 4H bars until exit
        "outcome_hold_hours":   None,
        "outcome_pnl_pct":      None,   # raw % move (unleveraged)
        "outcome_pnl_leveraged_pct": None,   # pnl_pct * leverage
        "outcome_funding_cost": None,   # accumulated funding_rate * hold_hours/8 * leverage
        "outcome_net_pct":      None,   # pnl_leveraged_pct - funding_cost
        "outcome_closest_to_tp_pct": None,
    }

    try:
        with open(SHADOW_PERPS_FILE, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"   👻 Shadow perp logged: {bias.upper()} @ ${price:.2f} | "
              f"stop={stop_price} | tp={tp_primary} | {leverage}x")
    except Exception as e:
        print(f"   ⚠️  Shadow perp log error: {e}")


def tag_shadow_perp_outcomes(df_4h_full):
    """
    Replays the price path for each untagged shadow perp record.

    Exit logic (checked bar-by-bar in order):
      1. Stop hit  → outcome = 'stop',      exit at stop_price
      2. TP primary hit → outcome = 'tp_primary', exit at tp_primary
      3. TP stretch hit → outcome = 'tp_stretch', exit at tp_stretch
      4. Timeout (SHADOW_LOOKFORWARD bars) → outcome = 'timeout', exit at close

    Funding cost approximation:
      Each 4H bar = 4 hours.  Funding is charged every 8h (twice per bar interval pair).
      accumulated_funding = funding_rate_at_entry * (hold_hours / 8) * leverage
      This is a conservative single-rate estimate — the replay script Jamie builds later
      can swap in hourly logged rates for precision.
    """
    if not SHADOW_PERPS_MODE:
        return
    if not os.path.exists(SHADOW_PERPS_FILE):
        return

    try:
        with open(SHADOW_PERPS_FILE, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return

        records  = [json.loads(l) for l in lines]
        modified = False

        for rec in records:
            if rec.get("outcome_tagged"):
                continue

            # Must have a valid directional bias and entry price
            bias  = rec.get("bias", "neutral")
            price = rec.get("price")
            if bias == "neutral" or not price or price <= 0:
                rec["outcome_tagged"] = True   # skip neutral entries cleanly
                modified = True
                continue

            stop_price  = rec.get("stop_price")
            tp_primary  = rec.get("tp_primary")
            tp_stretch  = rec.get("tp_stretch")
            leverage    = rec.get("leverage") or 1
            funding_rate = rec.get("funding_rate", 0)   # % per 8h at entry

            # If we have no stop AND no TP, we can't replay — skip for now
            if stop_price is None and tp_primary is None:
                continue

            sig_time_str = rec.get("timestamp")
            if not sig_time_str:
                continue

            # Resolve timestamp to index — same pattern as tag_outcomes_in_decisions()
            sig_time = pd.Timestamp(sig_time_str)
            if sig_time.tzinfo is None:
                sig_time = sig_time.tz_localize("UTC")
            else:
                sig_time = sig_time.tz_convert("UTC")
            sig_time = sig_time.as_unit('ms')

            idx_arr = df_4h_full.index.searchsorted(sig_time)
            if idx_arr < 1:
                continue

            fwd_start = idx_arr
            fwd_end   = min(fwd_start + SHADOW_LOOKFORWARD, len(df_4h_full))

            # Need at least 1 forward bar
            if fwd_start >= fwd_end:
                continue

            # Check whether enough forward data exists yet
            if fwd_end - fwd_start < SHADOW_LOOKFORWARD and fwd_end == len(df_4h_full):
                # Not enough history yet — wait for a future run
                continue

            result       = "timeout"
            exit_price   = None
            hold_bars    = SHADOW_LOOKFORWARD
            closest_to_tp_pct = None

            for bar_offset in range(fwd_end - fwd_start):
                bar = df_4h_full.iloc[fwd_start + bar_offset]
                bar_high = bar['high']
                bar_low  = bar['low']

                if tp_primary is not None:
                    if bias == "long" and bar_high > price:
                        pct = round((bar_high - price) / (tp_primary - price) * 100, 2)
                        if closest_to_tp_pct is None or pct > closest_to_tp_pct:
                            closest_to_tp_pct = pct
                    elif bias == "short" and bar_low < price:
                        pct = round((price - bar_low) / (price - tp_primary) * 100, 2)
                        if closest_to_tp_pct is None or pct > closest_to_tp_pct:
                            closest_to_tp_pct = pct

                if bias == "long":
                    # Check stop first (conservative — assume worst fills in same bar)
                    if stop_price is not None and bar_low <= stop_price:
                        result     = "stop"
                        exit_price = stop_price
                        hold_bars  = bar_offset + 1
                        break
                    if tp_stretch is not None and bar_high >= tp_stretch:
                        result     = "tp_stretch"
                        exit_price = tp_stretch
                        hold_bars  = bar_offset + 1
                        break
                    if tp_primary is not None and bar_high >= tp_primary:
                        result     = "tp_primary"
                        exit_price = tp_primary
                        hold_bars  = bar_offset + 1
                        break

                else:  # short
                    if stop_price is not None and bar_high >= stop_price:
                        result     = "stop"
                        exit_price = stop_price
                        hold_bars  = bar_offset + 1
                        break
                    if tp_stretch is not None and bar_low <= tp_stretch:
                        result     = "tp_stretch"
                        exit_price = tp_stretch
                        hold_bars  = bar_offset + 1
                        break
                    if tp_primary is not None and bar_low <= tp_primary:
                        result     = "tp_primary"
                        exit_price = tp_primary
                        hold_bars  = bar_offset + 1
                        break

            # Timeout exit — use the close of the last forward bar
            if result == "timeout":
                exit_price = float(df_4h_full.iloc[fwd_start + hold_bars - 1]['close'])

            if exit_price is None or exit_price <= 0:
                continue

            # Raw (unleveraged) P&L %
            if bias == "long":
                pnl_pct = round((exit_price - price) / price * 100, 4)
            else:
                pnl_pct = round((price - exit_price) / price * 100, 4)

            hold_hours       = hold_bars * 4
            pnl_lev          = round(pnl_pct * leverage, 4)
            # funding_cost is the drag on the leveraged position
            # positive cost = paid out, negative = received (rare)
            funding_cost     = round(funding_rate * (hold_hours / 8) * leverage, 4)
            net_pct          = round(pnl_lev - funding_cost, 4)

            rec["outcome_tagged"]            = True
            rec["outcome_result"]            = result
            rec["outcome_exit_price"]        = round(exit_price, 4)
            rec["outcome_hold_bars"]         = hold_bars
            rec["outcome_hold_hours"]        = hold_hours
            rec["outcome_pnl_pct"]           = pnl_pct
            rec["outcome_pnl_leveraged_pct"] = pnl_lev
            rec["outcome_funding_cost"]      = funding_cost
            rec["outcome_net_pct"]           = net_pct
            rec["outcome_closest_to_tp_pct"] = closest_to_tp_pct
            modified = True

        if modified:
            with open(SHADOW_PERPS_FILE, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
            tagged_count = sum(1 for r in records if r.get("outcome_tagged"))
            print(f"   👻 Shadow perp outcomes: {tagged_count} total tagged")

    except Exception as e:
        print(f"   ⚠️  Shadow perp outcome tagging error: {e}")


# ====================== DECISION LOGGING ======================
def log_decision(decision, price, local_regime, rsi_4h, rsi_1h, div_4h, div_1h,
                 macd_4h, ema_4h, fib_label_4h, atr_4h, perp,
                 fear_greed=None, btc_change=None,
                 vol_ratio_4h=1.0, vol_ratio_1h=1.0, vol_4h=0.0):
    """Log every decision to DECISIONS_FILE for outcome tagging later."""
    # Parse macd_cross and macd_hist from the macd_4h string
    # Format: "bullish_cross | hist 0.0042 (improving)"
    try:
        macd_cross = macd_4h.split(" | ")[0].strip()
    except Exception:
        macd_cross = "no_cross"
    try:
        macd_hist = round(float(macd_4h.split("hist ")[1].split(" ")[0]), 4)
    except Exception:
        macd_hist = 0

    # Parse fib_closest from fib_label_4h string
    # Format: "nearest=0.618 ($XXX.XX) | ..."
    try:
        fib_closest = fib_label_4h.split("nearest=")[1].split(" ")[0]
    except Exception:
        fib_closest = ""

    rec = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "price":          round(price, 4),
        "local_regime":   local_regime,
        "outcome_tagged": False,
        # ── Extended data fields ──
        "rsi_4h":         round(rsi_4h, 1),
        "rsi_1h":         round(rsi_1h, 1),
        "div_4h":         div_4h,
        "div_1h":         div_1h,
        "macd_cross":     macd_cross,
        "macd_hist":      macd_hist,
        "ema_status":     ema_4h,
        "fib_closest":    fib_closest,
        "atr":            round(atr_4h, 4),
        "funding_rate":   perp.get("funding_rate_pct", 0),
        "open_interest":  perp.get("open_interest_m", 0),
        "fear_greed":     fear_greed,
        "btc_change":     btc_change,
        "vol_ratio_4h":   round(vol_ratio_4h, 3),
        "vol_ratio_1h":   round(vol_ratio_1h, 3),
        "vol_4h_raw":     round(vol_4h, 2),
        **decision,
    }
    with open(DECISIONS_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ====================== MAIN LOOP ======================
if __name__ == "__main__":
    print("🚀 SOL Perp Advisor (Jupiter) — Bidirectional + Self-Backtesting")
    print(f"   Model: {MODEL} | Interval: {CHECK_INTERVAL // 60}min\n")

    while True:
        try:
            print("📡 Fetching data...")
            df_4h     = get_klines("4h",  400)
            df_1h     = get_klines("1h",  500)
            df_daily  = get_klines("1d",  200)
            df_weekly = get_klines("1w",  100)
            perp      = get_perp_data()

            # Long history for backtest (2000 x 4H bars ≈ 11 months)
            print("   📈 Fetching extended 4H history for backtest...")
            df_4h_long = get_klines("4h", 2000)

            # ── Phase 1: tag outcomes on past signals ──────────────────────
            print("   🏷️  Tagging signal outcomes...")
            tag_outcomes_in_decisions(df_4h_long)
            tag_shadow_perp_outcomes(df_4h_long)
            signal_perf = get_signal_performance_summary()
            print(f"   Track record: {signal_perf}")

            # ── Build TA summary ───────────────────────────────────────────
            (ta_summary, rsi_4h, rsi_1h, div_4h, div_1h,
             atr_4h, price, local_regime,
             fib_label_4h, ema_4h, macd_4h,
             vol_ratio_4h, vol_ratio_1h, vol_4h) = build_ta_summary(df_4h, df_1h, df_daily, df_weekly)

            # ── Phase 2: historical edge for this exact setup ──────────────
            print("   🔍 Computing historical edge...")
            historical_edge = get_historical_edge(
                df_4h_long, local_regime, div_4h, macd_4h, ema_4h, atr_4h
            )
            print(f"   Edge: {historical_edge}")

            trend_context = load_latest_trend_report()
            regime_gate   = build_regime_gate(local_regime, trend_context)

            print(f"\n📊 TA Dashboard:\n{ta_summary}")
            print(f"   Funding: {perp['funding_rate_pct']:+.4f}% "
                  f"(avg 3p: {perp['funding_avg_3period']:+.4f}%) | OI: ${perp['open_interest_m']}M")
            print(f"   Local regime: {local_regime.upper()}\n")

            print("🤖 Querying AI...")
            decision = get_ai_decision(
                ta_summary, perp, price, trend_context, regime_gate,
                local_regime, signal_perf, historical_edge
            )

            bias       = decision.get('bias', 'neutral').upper()
            confidence = decision.get('confidence', 0)
            emoji      = "🟢" if bias == "LONG" else "🔴" if bias == "SHORT" else "⚪"

            print(f"{'='*65}")
            print(f"🕒 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
            print(f"{emoji} BIAS: {bias} | Confidence: {confidence}/10")
            print(f"   Long score: {decision.get('long_score','?')} | "
                  f"Short score: {decision.get('short_score','?')}")
            print(f"   Regime alignment: {decision.get('regime_alignment','?')}")
            print(f"📝 {decision.get('reason', '')}")
            print(f"🎯 Entry:        {decision.get('entry_note', '')}")
            print(f"📍 Leverage:     {decision.get('suggested_leverage', '')}")
            print(f"🛑 Stop:         {decision.get('stop_loss', '')}")
            print(f"🎯 TP:           {decision.get('take_profit', '')}")
            print(f"❌ Invalidation: {decision.get('invalidation', '')}")
            print(f"⚠️  Risks:        {' | '.join(decision.get('key_risks', []))}")
            print(f"{'='*65}\n")

            send_advisor_email(decision, price, ta_summary, perp,
                               trend_context, signal_perf, historical_edge)

            # Log this decision (outcome will be tagged on a future run)
            log_decision(
                decision, price, local_regime,
                rsi_4h, rsi_1h, div_4h, div_1h,
                macd_4h, ema_4h, fib_label_4h, atr_4h, perp,
                vol_ratio_4h=vol_ratio_4h, vol_ratio_1h=vol_ratio_1h, vol_4h=vol_4h,
            )
            # Shadow Perps Mode — virtual trade log (no real position opened)
            log_shadow_perp(
                decision, price, local_regime, perp,
                rsi_4h, div_4h, macd_4h, ema_4h, atr_4h,
                vol_ratio_4h=vol_ratio_4h, vol_ratio_1h=vol_ratio_1h, vol_4h=vol_4h,
            )

        except Exception as e:
            print(f"❌ Error: {e}")
            try:
                send_advisor_email(
                    {"bias": "neutral", "reason": f"CRASHED: {e}", "key_risks": []},
                    None, "TA unavailable",
                    {"funding_rate_pct": 0, "funding_avg_3period": 0, "open_interest_m": 0},
                    "Trend report unavailable", "N/A", "N/A"
                )
            except Exception:
                pass

        print(f"⏳ Next run in {CHECK_INTERVAL // 60} minutes...\n")
        time.sleep(CHECK_INTERVAL)
