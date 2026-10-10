"""
advisor_v2.py — single-file deterministic perp signal backtester.

Replaces the LLM-as-trader with a pure-Python rules engine and a walk-forward
backtest harness that replays history bar-by-bar with no lookahead.
Symbol-agnostic but configured for SOLUSDT 4H by default.

Usage:
    pip install pandas numpy python-binance python-dotenv pyarrow

    python advisor_v2.py                    # live loop (default)
    python advisor_v2.py --check            # one-shot check
    python advisor_v2.py --backtest         # backtest, DEFAULT_BACKTEST_DAYS days
    python advisor_v2.py --backtest --days 365

All tuning is in the USER CONFIG block at the top of the file.

Layout (top-to-bottom):
    USER CONFIG  (the only place you tune anything)
    1. Internal config + imports
    2. Data loader (Binance klines + funding, parquet cache)
    3. Indicators (RSI, MACD, divergence, EMA, ATR, fib, regime)
    4. Rules engine (Snapshot, Decision, scoring, decide)
    5. Backtest harness (build_snapshot, simulate_outcome, run_backtest)
    6. Analysis (summarize, walk_forward_split)
    7. Live + loop modes
    8. CLI entrypoint
"""
import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from time import sleep
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

# ====================================================================
# ▼▼▼ USER CONFIG — TUNE HERE ▼▼▼
# Everything in this block used to be a CLI flag. Now it's baked in.
# Override at runtime only via the 3 flags in main(): --backtest, --check, --symbol.
# ====================================================================
# What & where
SYMBOL                = "SOLUSDT"
INTERVAL              = "4h"           # primary timeframe ("1h", "4h", "1d")
DEFAULT_BACKTEST_DAYS = 730            # 2 years; override with --days N

# Trade timing
FORWARD_BARS          = 12             # max bars to hold (12 × 4h = 48h)
COOLDOWN_BARS         = 6              # min bars between signals (24h on 4h TF)
CHECK_MINUTES         = 60             # loop-mode poll interval

# Scoring thresholds
THRESHOLD_HIGH        = 7.0            # high-confidence fire
THRESHOLD_MOD         = 7.0            # moderate-confidence fire
# Geometry
ATR_STOP_MULT         = 1.5
TP_ATR_MULT_PRIMARY   = 3.0
TP_ATR_MULT_STRETCH   = 4.0
MIN_RR                = 1.5            # trades below this R:R get rejected

# Sizing
LEVERAGE_DEFAULT      = 3
LEVERAGE_HIGH_CONF    = 3              # used when confidence >= 8.0
POSITION_SIZE_PCT     = 100.0           # % of normal position per trade.
                                       # 100 = full size, 50 = half size, etc.
                                       # Scales P&L AND funding proportionally.

# Volatility-targeted sizing
# When True, position size is computed per-trade so each stop-out costs
# ~VOL_TARGET_LOSS_PCT of equity, regardless of ATR. High-vol bars get
# smaller size; calm bars get larger size. Caps prevent extremes.
# When False, falls back to fixed POSITION_SIZE_PCT.
VOL_TARGET_MODE       = True
VOL_TARGET_LOSS_PCT   = 5.0        # net % equity lost per stopped trade
VOL_SIZE_MIN          = 10.0        # floor on per-trade size %
VOL_SIZE_MAX          = 100.0      # ceiling (don't oversize in ultra-calm vol)

# Caps
LONG_BEAR_CAP         = 6.0            # max long score when regime == bear

# Cache freshness (live mode uses LIVE_CACHE_MAX_AGE_HOURS, backtest uses the loose default)
LIVE_CACHE_MAX_AGE_HOURS = 1.5         # in live mode, refetch if cache older than this
# ====================================================================
# ▲▲▲ END USER CONFIG ▲▲▲
# ====================================================================


# ====================================================================
# 1. INTERNAL CONFIG
# ====================================================================
CACHE_DIR = Path(os.getenv("ADVISOR_CACHE_DIR", "data_cache"))
CACHE_DIR.mkdir(exist_ok=True)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Client(
            api_key=os.getenv("BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_API_SECRET"),
        )
    return _client


def _interval_to_ms(interval: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    return int(interval[:-1]) * units[interval[-1]]


def _interval_hours(interval: str) -> float:
    units = {"m": 1 / 60, "h": 1, "d": 24, "w": 168}
    return int(interval[:-1]) * units[interval[-1]]


# ====================================================================
# 2. DATA LOADER — Binance klines + funding with parquet caching
# ====================================================================
def fetch_klines(symbol, interval, start, end, use_cache=True,
                 max_cache_staleness_hours: Optional[float] = None) -> pd.DataFrame:
    """Fetch klines for [start, end). Paginates as needed. Cached to parquet.

    max_cache_staleness_hours: if set, the cached file must cover data up to within
    this many hours of `end` to be reused. Live mode passes a small value (e.g. 1.5h)
    to ensure recent bars are always refetched. Backtest leaves it as None (uses the
    default 2-day tolerance, which is fine since history doesn't change).
    """
    cache_path = CACHE_DIR / f"{symbol}_{interval}_klines.parquet"

    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if max_cache_staleness_hours is not None:
            freshness_floor = end - timedelta(hours=max_cache_staleness_hours)
        else:
            freshness_floor = end - timedelta(days=2)
        if not cached.empty and cached.index.min() <= start and cached.index.max() >= freshness_floor:
            return cached.loc[(cached.index >= start) & (cached.index < end)].copy()

    client = _get_client()
    all_chunks = []
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    bar_ms = _interval_to_ms(interval)
    fail_count = 0

    while cursor_ms < end_ms:
        try:
            chunk = client.futures_klines(
                symbol=symbol, interval=interval,
                startTime=cursor_ms, endTime=end_ms, limit=1000,
            )
            fail_count = 0
        except Exception as e:
            fail_count += 1
            if fail_count > 5:
                raise
            print(f"  klines fetch failed ({e}), retrying...")
            sleep(3)
            continue

        if not chunk:
            break
        all_chunks.extend(chunk)
        cursor_ms = int(chunk[-1][0]) + bar_ms
        if len(chunk) < 1000:
            break
        sleep(0.1)

    if not all_chunks:
        return pd.DataFrame()

    df = pd.DataFrame(all_chunks, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "n", "tbav", "tbqav", "ignore",
    ])
    df = df[["open_time", "open", "high", "low", "close", "volume"]].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    if use_cache:
        if cache_path.exists():
            old = pd.read_parquet(cache_path)
            df = pd.concat([old, df]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        df.to_parquet(cache_path)

    return df.loc[(df.index >= start) & (df.index < end)].copy()


def fetch_funding(symbol, start, end, use_cache=True) -> pd.DataFrame:
    """Historical funding rates. Index = funding_time UTC, column = funding_rate (decimal)."""
    cache_path = CACHE_DIR / f"{symbol}_funding.parquet"

    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not cached.empty and cached.index.min() <= start and cached.index.max() >= end - timedelta(days=2):
            return cached.loc[(cached.index >= start) & (cached.index < end)].copy()

    client = _get_client()
    all_chunks = []
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    fail_count = 0

    while cursor_ms < end_ms:
        try:
            chunk = client.futures_funding_rate(
                symbol=symbol, startTime=cursor_ms, endTime=end_ms, limit=1000,
            )
            fail_count = 0
        except Exception as e:
            fail_count += 1
            if fail_count > 5:
                raise
            print(f"  funding fetch failed ({e}), retrying...")
            sleep(3)
            continue

        if not chunk:
            break
        all_chunks.extend(chunk)
        cursor_ms = int(chunk[-1]["fundingTime"]) + 1
        if len(chunk) < 1000:
            break
        sleep(0.1)

    if not all_chunks:
        return pd.DataFrame()

    df = pd.DataFrame([{
        "funding_time": pd.to_datetime(int(f["fundingTime"]), unit="ms", utc=True),
        "funding_rate": float(f["fundingRate"]),
    } for f in all_chunks])
    df = df.set_index("funding_time").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    if use_cache:
        if cache_path.exists():
            old = pd.read_parquet(cache_path)
            df = pd.concat([old, df]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        df.to_parquet(cache_path)

    return df.loc[(df.index >= start) & (df.index < end)].copy()


# ====================================================================
# 3. INDICATORS — pure functions, no lookahead enforced by caller
# ====================================================================
def rsi(df, period=14):
    """Wilder-style RSI. Returns latest float (50.0 if insufficient data)."""
    if len(df) < period + 10:
        return 50.0
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return float((100 - (100 / (1 + rs))).iloc[-1])


def macd(df, fast=12, slow=26, signal=9):
    """Returns dict with cross / hist / hist_improving / 3-bar trend flags.

    The 3-bar trend flags exist because the single-bar `hist_improving` flag is too
    loose for use in scoring: "hist not improving" includes every flat/sleepy bar,
    which previously gave shorts a +0.75 credit and longs a -1.5 penalty on
    *every* quiet bar. The 3-bar flags require an actual trend (3 consecutive
    changes in the same direction) before crediting/penalizing.
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig

    if len(macd_line) < 2:
        return {"cross": "no_cross", "hist": 0.0, "hist_improving": False,
                "hist_3bar_improving": False, "hist_3bar_deteriorating": False}

    last, prev = macd_line.iloc[-1], macd_line.iloc[-2]
    sig_last, sig_prev = sig.iloc[-1], sig.iloc[-2]

    if last > sig_last and prev <= sig_prev:
        cross = "bullish_cross"
    elif last < sig_last and prev >= sig_prev:
        cross = "bearish_cross"
    else:
        cross = "no_cross"

    improving = bool(hist.iloc[-1] > hist.iloc[-2])

    # 3-bar trend: 4 most recent hist values, 3 consecutive moves in same direction.
    if len(hist) >= 4:
        h = hist.iloc[-4:].values
        hist_3bar_improving = bool(h[1] > h[0] and h[2] > h[1] and h[3] > h[2])
        hist_3bar_deteriorating = bool(h[1] < h[0] and h[2] < h[1] and h[3] < h[2])
    else:
        hist_3bar_improving = False
        hist_3bar_deteriorating = False

    return {
        "cross": cross,
        "hist": float(hist.iloc[-1]),
        "hist_improving": improving,
        "hist_3bar_improving": hist_3bar_improving,
        "hist_3bar_deteriorating": hist_3bar_deteriorating,
    }


def ema_status(df, fast_span=50, slow_span=200):
    """One of: above_both_bullish, below_both_bearish, mixed."""
    ema_f = df["close"].ewm(span=fast_span, adjust=False).mean().iloc[-1]
    ema_s = df["close"].ewm(span=slow_span, adjust=False).mean().iloc[-1]
    price = df["close"].iloc[-1]
    if price > ema_f > ema_s:
        return "above_both_bullish"
    if price < ema_f < ema_s:
        return "below_both_bearish"
    return "mixed"


def atr(df, period=14):
    """Average true range, last value."""
    if len(df) < period + 1:
        return 0.0
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    val = tr.rolling(window=period).mean().iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def fib_levels(df, lookback=120):
    """Return dict: levels, nearest level name, nearest_price."""
    recent = df.tail(lookback)
    high = float(recent["high"].max())
    low = float(recent["low"].min())
    diff = high - low
    levels = {
        "0.000": low,
        "0.236": low + 0.236 * diff,
        "0.382": low + 0.382 * diff,
        "0.500": low + 0.500 * diff,
        "0.618": low + 0.618 * diff,
        "0.786": low + 0.786 * diff,
        "1.000": high,
    }
    price = float(df["close"].iloc[-1])
    nearest = min(levels, key=lambda k: abs(levels[k] - price))
    return {
        "levels": {k: round(v, 4) for k, v in levels.items()},
        "nearest": nearest,
        "nearest_price": round(levels[nearest], 4),
    }


def rsi_divergence(df, lookback=25, swing_window=3, min_volume_confirm=True):
    """
    Returns one of: regular_bullish, hidden_bullish, regular_bearish,
    hidden_bearish, none.
    """
    if len(df) < lookback + 14:
        return "none"

    close = df["close"].iloc[-lookback:].reset_index(drop=True)
    volume = df["volume"].iloc[-lookback:].reset_index(drop=True)

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_full = 100 - (100 / (1 + rs))
    rsi_recent = rsi_full.iloc[-lookback:].reset_index(drop=True)

    def _swings(series, kind):
        out = []
        for i in range(swing_window, len(series) - swing_window):
            window = series.iloc[i - swing_window: i + swing_window + 1]
            if kind == "low" and window.min() == series.iloc[i]:
                out.append((i, float(series.iloc[i])))
            elif kind == "high" and window.max() == series.iloc[i]:
                out.append((i, float(series.iloc[i])))
        deduped = []
        for idx, val in out:
            if deduped and idx - deduped[-1][0] <= swing_window:
                if (kind == "low" and val < deduped[-1][1]) or (kind == "high" and val > deduped[-1][1]):
                    deduped[-1] = (idx, val)
            else:
                deduped.append((idx, val))
        return deduped

    p_lows = _swings(close, "low")
    p_highs = _swings(close, "high")
    r_lows = _swings(rsi_recent, "low")
    r_highs = _swings(rsi_recent, "high")

    if len(p_lows) >= 2 and len(r_lows) >= 2:
        pl1, pl2 = p_lows[-2], p_lows[-1]
        rl1, rl2 = r_lows[-2], r_lows[-1]
        vol_ok = (not min_volume_confirm) or volume.iloc[pl2[0]] > volume.iloc[pl1[0]]
        if pl2[1] < pl1[1] and rl2[1] > rl1[1] and vol_ok:
            return "regular_bullish"
        if pl2[1] > pl1[1] and rl2[1] < rl1[1] and vol_ok:
            return "hidden_bullish"

    if len(p_highs) >= 2 and len(r_highs) >= 2:
        ph1, ph2 = p_highs[-2], p_highs[-1]
        rh1, rh2 = r_highs[-2], r_highs[-1]
        vol_ok = (not min_volume_confirm) or volume.iloc[ph2[0]] > volume.iloc[ph1[0]]
        if ph2[1] > ph1[1] and rh2[1] < rh1[1] and vol_ok:
            return "regular_bearish"
        if ph2[1] < ph1[1] and rh2[1] > rh1[1] and vol_ok:
            return "hidden_bearish"

    return "none"


def detect_regime(df_daily, df_weekly):
    """Macro regime from daily + weekly. Returns: bull, bear, chop, unknown."""
    try:
        w_price = df_weekly["close"].iloc[-1]
        w_ema50 = df_weekly["close"].ewm(span=50, adjust=False).mean().iloc[-1]
        w_rsi = rsi(df_weekly)
        d_price = df_daily["close"].iloc[-1]
        d_ema50 = df_daily["close"].ewm(span=50, adjust=False).mean().iloc[-1]
        d_ema200 = df_daily["close"].ewm(span=200, adjust=False).mean().iloc[-1]
        if w_price < w_ema50 and w_rsi < 50 and d_price < d_ema50 < d_ema200:
            return "bear"
        if w_price > w_ema50 and w_rsi > 55 and d_price > d_ema50 > d_ema200:
            return "bull"
        return "chop"
    except Exception:
        return "unknown"


def volume_ratio(df, period=20):
    """Last bar volume / N-bar moving avg of prior bars."""
    if len(df) < 2:
        return 1.0
    last = float(df["volume"].iloc[-1])
    ma = float(df["volume"].iloc[:-1].rolling(period, min_periods=1).mean().iloc[-1])
    return last / ma if ma > 0 else 1.0


# ====================================================================
# 4. RULES ENGINE — replaces the LLM with deterministic scoring
# ====================================================================
@dataclass
class Snapshot:
    """All inputs the rules engine needs at one point in time."""
    timestamp: object
    symbol: str
    interval: str
    price: float

    # Primary timeframe
    rsi_main: float
    div_main: str
    macd_cross_main: str
    macd_hist_main: float
    macd_improving_main: bool                 # 1-bar; logged for audit only
    macd_hist_3bar_improving: bool            # used by scorer
    macd_hist_3bar_deteriorating: bool        # used by scorer
    ema_status_main: str
    atr_main: float
    vol_ratio_main: float

    # 1H confirmation
    rsi_1h: float
    div_1h: str
    vol_ratio_1h: float

    # Daily structure  (daily_ema200 = None when fewer than 200 daily bars available)
    daily_price: float
    daily_ema50: float
    daily_ema200: Optional[float]
    daily_rsi: float

    # Weekly + regime
    weekly_rsi: float
    regime: str

    # Fib (logging)
    fib_nearest: str
    fib_nearest_price: float

    # Funding (percent, e.g. 0.01 = 0.01%)
    funding_rate: float
    funding_avg_3p: float

    open_interest_m: float = 0.0


@dataclass
class Decision:
    bias: str
    confidence: float
    long_score: float
    short_score: float
    entry_price: float
    stop_price: float
    tp_primary: float
    tp_stretch: float
    leverage: int
    risk_pct: float
    reward_pct_primary: float
    rr_primary: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class RulesConfig:
    """Tunable parameters. Defaults come from the USER CONFIG block at top of file."""
    threshold_high: float = THRESHOLD_HIGH
    threshold_mod: float = THRESHOLD_MOD

    atr_stop_mult: float = ATR_STOP_MULT
    tp_atr_mult_primary: float = TP_ATR_MULT_PRIMARY
    tp_atr_mult_stretch: float = TP_ATR_MULT_STRETCH
    min_rr: float = MIN_RR

    leverage_default: int = LEVERAGE_DEFAULT
    leverage_high_conf: int = LEVERAGE_HIGH_CONF
    position_size_pct: float = POSITION_SIZE_PCT

    long_bear_cap: float = LONG_BEAR_CAP

    # Volatility-targeted sizing
    vol_target_mode: bool = VOL_TARGET_MODE
    vol_target_loss_pct: float = VOL_TARGET_LOSS_PCT
    vol_size_min: float = VOL_SIZE_MIN
    vol_size_max: float = VOL_SIZE_MAX


def score_long(s: Snapshot) -> Tuple[float, List[str]]:
    score = 5.0
    reasons = []

    if s.div_main == "regular_bullish":
        score += 2.5; reasons.append("+2.50 reg_bullish_main")
    elif s.div_main == "hidden_bullish":
        score += 2.0; reasons.append("+2.00 hidden_bullish_main")

    if s.div_1h in ("regular_bullish", "hidden_bullish"):
        score += 1.5; reasons.append("+1.50 bullish_div_1h")

    # MACD bullish — fresh cross OR confirmed 3-bar improvement
    if s.macd_cross_main == "bullish_cross":
        score += 1.5; reasons.append("+1.50 macd_bull_cross")
    elif s.macd_hist_3bar_improving:
        score += 0.75; reasons.append("+0.75 macd_hist_improving_3bar")

    if s.daily_price > s.daily_ema50:
        score += 1.0; reasons.append("+1.00 above_daily_ema50")

    if s.regime != "bear" and s.funding_avg_3p < -0.01 and s.rsi_main < 35:
        score += 1.0; reasons.append("+1.00 neg_funding+oversold")

    if 50.0 <= s.rsi_1h <= 55.0:
        score += 0.5; reasons.append("+0.50 rsi_1h_reclaim_50")

    # Penalties
    if s.regime == "bear":
        score -= 3.0; reasons.append("-3.00 bear_regime")

    # Below both daily EMAs without bullish div. Only fires when EMA200 is available.
    if (s.daily_ema200 is not None
            and s.daily_price < s.daily_ema50
            and s.daily_price < s.daily_ema200
            and s.div_main not in ("regular_bullish", "hidden_bullish")):
        score -= 2.0; reasons.append("-2.00 below_daily_emas_no_div")

    # MACD bearish — only on a *real* signal, not the default "no_cross + not improving"
    # state which the old version misread as deteriorating.
    if s.macd_cross_main == "bearish_cross":
        score -= 1.5; reasons.append("-1.50 macd_bear_cross")
    elif s.macd_hist_3bar_deteriorating:
        score -= 1.5; reasons.append("-1.50 macd_hist_deteriorating_3bar")

    if s.div_main == "none":
        score -= 1.0; reasons.append("-1.00 no_div_main")

    return score, reasons



def decide(s: Snapshot, cfg: Optional[RulesConfig] = None) -> Optional[Decision]:
    """Return a long Decision or None. Shorting removed — see USER CONFIG."""
    if cfg is None:
        cfg = RulesConfig()

    # Skip chop regime — backtest shows no edge here (train −2.77%, test +0.24%)
    if s.regime == "chop":
        return None

    long_s, long_reasons = score_long(s)

    if s.regime == "bear":
        long_s = min(long_s, cfg.long_bear_cap)

    if long_s < cfg.threshold_mod:
        return None

    confidence = min(long_s, 9.5)

    if s.atr_main <= 0:
        return None

    stop_dist = cfg.atr_stop_mult * s.atr_main
    tp1_dist  = cfg.tp_atr_mult_primary * s.atr_main
    tp2_dist  = cfg.tp_atr_mult_stretch * s.atr_main

    stop = s.price - stop_dist
    tp1  = s.price + tp1_dist
    tp2  = s.price + tp2_dist

    risk_pct   = (s.price - stop) / s.price * 100
    reward_pct = (tp1 - s.price)  / s.price * 100
    rr = reward_pct / risk_pct if risk_pct > 0 else 0.0

    if rr < cfg.min_rr:
        return None

    leverage = cfg.leverage_high_conf if confidence >= 8.0 else cfg.leverage_default

    return Decision(
        bias="long",
        confidence=round(confidence, 2),
        long_score=round(long_s, 2),
        short_score=0.0,   # kept on Decision for JSON-log compat; always 0 now
        entry_price=round(s.price, 4),
        stop_price=round(stop, 4),
        tp_primary=round(tp1, 4),
        tp_stretch=round(tp2, 4),
        leverage=leverage,
        risk_pct=round(risk_pct, 3),
        reward_pct_primary=round(reward_pct, 3),
        rr_primary=round(rr, 2),
        reasons=long_reasons,
    )


# ====================================================================
# 5. BACKTEST HARNESS — walk-forward, no lookahead
# ====================================================================
def build_snapshot(symbol, interval, df_main, df_1h, df_daily, df_weekly,
                   funding_df, current_time) -> Optional[Snapshot]:
    """Build Snapshot from already-sliced dfs (must be <= current_time)."""
    if len(df_main) < 200 or len(df_daily) < 50:
        return None

    price = float(df_main["close"].iloc[-1])
    macd_main = macd(df_main)
    fib = fib_levels(df_main)

    if df_1h is not None and len(df_1h) >= 30:
        r_1h = rsi(df_1h)
        d_1h = rsi_divergence(df_1h, lookback=18)
        v_1h = volume_ratio(df_1h)
    else:
        r_1h, d_1h, v_1h = 50.0, "none", 1.0

    daily_price = float(df_daily["close"].iloc[-1])
    daily_ema50 = float(df_daily["close"].ewm(span=50, adjust=False).mean().iloc[-1])
    if len(df_daily) >= 200:
        daily_ema200 = float(df_daily["close"].ewm(span=200, adjust=False).mean().iloc[-1])
    else:
        # Insufficient history — don't silently substitute EMA50. Scorers gate on None.
        daily_ema200 = None
    daily_rsi = rsi(df_daily)

    if df_weekly is not None and len(df_weekly) >= 20:
        weekly_rsi = rsi(df_weekly)
        regime = detect_regime(df_daily, df_weekly)
    else:
        weekly_rsi, regime = 50.0, "unknown"

    funding_rate = funding_avg_3p = 0.0
    if funding_df is not None and len(funding_df) > 0:
        recent = funding_df[funding_df.index <= current_time].tail(3)
        if len(recent) >= 1:
            funding_rate = float(recent["funding_rate"].iloc[-1]) * 100
            funding_avg_3p = float(recent["funding_rate"].mean()) * 100

    return Snapshot(
        timestamp=current_time, symbol=symbol, interval=interval, price=price,
        rsi_main=rsi(df_main),
        div_main=rsi_divergence(df_main, lookback=25),
        macd_cross_main=macd_main["cross"],
        macd_hist_main=macd_main["hist"],
        macd_improving_main=macd_main["hist_improving"],
        macd_hist_3bar_improving=macd_main["hist_3bar_improving"],
        macd_hist_3bar_deteriorating=macd_main["hist_3bar_deteriorating"],
        ema_status_main=ema_status(df_main),
        atr_main=atr(df_main),
        vol_ratio_main=volume_ratio(df_main),
        rsi_1h=r_1h, div_1h=d_1h, vol_ratio_1h=v_1h,
        daily_price=daily_price, daily_ema50=daily_ema50,
        daily_ema200=daily_ema200, daily_rsi=daily_rsi,
        weekly_rsi=weekly_rsi, regime=regime,
        fib_nearest=fib["nearest"], fib_nearest_price=fib["nearest_price"],
        funding_rate=funding_rate, funding_avg_3p=funding_avg_3p,
    )


def simulate_outcome(decision, df_forward, funding_df, max_bars, interval_hours,
                     position_size_pct: float = 100.0):
    """
    Forward replay starting at the bar AFTER entry. Long-only.
    Conservative same-bar fills: stop checked before TP1, TP1 before TP2.

    position_size_pct scales final P&L and funding cost proportionally
    (e.g. 50.0 means each trade uses half normal size — both wins and losses halved).
    """
    if df_forward is None or len(df_forward) == 0:
        return None
    bars_to_use = min(max_bars, len(df_forward))
    if bars_to_use == 0:
        return None

    entry, stop = decision.entry_price, decision.stop_price
    tp1, tp2 = decision.tp_primary, decision.tp_stretch
    leverage = decision.leverage

    result = "timeout"
    exit_price = None
    hold_bars = bars_to_use
    closest_to_tp_pct = None

    for offset in range(bars_to_use):
        bar = df_forward.iloc[offset]
        bar_high = float(bar["high"])
        bar_low  = float(bar["low"])

        if tp1 > entry and bar_high > entry:
            pct = (bar_high - entry) / (tp1 - entry) * 100
            if closest_to_tp_pct is None or pct > closest_to_tp_pct:
                closest_to_tp_pct = pct

        # Stop first (conservative); TP1 before TP2 (TP1 is closer to entry)
        if bar_low <= stop:
            result, exit_price, hold_bars = "stop", stop, offset + 1; break
        if bar_high >= tp1:
            result, exit_price, hold_bars = "tp_primary", tp1, offset + 1; break
        if bar_high >= tp2:
            result, exit_price, hold_bars = "tp_stretch", tp2, offset + 1; break

    if exit_price is None:
        exit_price = float(df_forward.iloc[hold_bars - 1]["close"])

    pnl_pct = (exit_price - entry) / entry * 100
    pnl_lev = pnl_pct * leverage
    hold_hours = hold_bars * interval_hours

    # Funding (longs pay when funding rate is positive)
    funding_cost = 0.0
    if funding_df is not None and len(funding_df) > 0:
        entry_time = df_forward.index[0]
        exit_time  = df_forward.index[hold_bars - 1]
        events = funding_df[(funding_df.index >= entry_time) & (funding_df.index <= exit_time)]
        for _, row in events.iterrows():
            rate_pct = float(row["funding_rate"]) * 100
            funding_cost += rate_pct * leverage   # sign always + for longs

    size_mult = position_size_pct / 100.0
    net_pct = (pnl_lev - funding_cost) * size_mult

    return {
        "outcome_result": result,
        "outcome_exit_price": round(exit_price, 4),
        "outcome_hold_bars": hold_bars,
        "outcome_hold_hours": round(hold_hours, 2),
        "outcome_pnl_pct": round(pnl_pct, 4),
        "outcome_pnl_leveraged_pct": round(pnl_lev * size_mult, 4),
        "outcome_funding_cost": round(funding_cost * size_mult, 4),
        "outcome_net_pct": round(net_pct, 4),
        "outcome_closest_to_tp_pct": round(closest_to_tp_pct, 2) if closest_to_tp_pct is not None else None,
        "outcome_position_size_pct": position_size_pct,
    }


def run_backtest(symbol, primary_interval, start, end,
                 rules_config=None, forward_bars=12, cooldown_bars=6,
                 use_cache=True, verbose=True) -> pd.DataFrame:
    """Walk every primary-TF bar in [start, end), generate trades."""
    if rules_config is None:
        rules_config = RulesConfig()

    interval_h = _interval_hours(primary_interval)
    pad_days = max(60, int(220 * interval_h / 24))
    fetch_start = start - timedelta(days=pad_days)

    if verbose:
        print(f"Fetching {symbol} {primary_interval} from {fetch_start.date()} to {end.date()}")

    df_main = fetch_klines(symbol, primary_interval, fetch_start, end, use_cache=use_cache)
    df_1h = (fetch_klines(symbol, "1h", fetch_start, end, use_cache=use_cache)
             if primary_interval != "1h" else df_main)
    df_daily = fetch_klines(symbol, "1d", fetch_start - timedelta(days=200), end, use_cache=use_cache)
    df_weekly = fetch_klines(symbol, "1w", fetch_start - timedelta(days=400), end, use_cache=use_cache)
    funding_df = fetch_funding(symbol, fetch_start, end, use_cache=use_cache)

    if verbose:
        print(f"  {primary_interval}: {len(df_main)} | 1h: {len(df_1h)} | 1d: {len(df_daily)} | "
              f"1w: {len(df_weekly)} | funding: {len(funding_df)}")

    if df_main.empty:
        print("  no data, aborting")
        return pd.DataFrame()

    walk_bars = df_main[(df_main.index >= start) & (df_main.index < end)]
    if verbose:
        print(f"Walking {len(walk_bars)} bars...")

    trades = []
    cooldown_until = None
    last_progress = 0

    for i, bar_time in enumerate(walk_bars.index):
        if cooldown_until is not None and bar_time < cooldown_until:
            continue

        m = df_main.loc[df_main.index <= bar_time]
        h = df_1h.loc[df_1h.index <= bar_time]
        d = df_daily.loc[df_daily.index <= bar_time]
        w = df_weekly.loc[df_weekly.index <= bar_time]

        snap = build_snapshot(symbol, primary_interval, m, h, d, w, funding_df, bar_time)
        if snap is None:
            continue

        decision = decide(snap, rules_config)
        if decision is None:
            continue

        df_fwd = df_main.loc[df_main.index > bar_time].iloc[:forward_bars]

        # Per-trade sizing: either fixed or vol-targeted
        if rules_config.vol_target_mode and decision.risk_pct > 0:
            # Solve: target_loss = size_pct/100 * leverage * risk_pct
            raw_size = (rules_config.vol_target_loss_pct
                        / (decision.risk_pct * decision.leverage) * 100)
            size_pct = max(rules_config.vol_size_min,
                           min(rules_config.vol_size_max, raw_size))
        else:
            size_pct = rules_config.position_size_pct

        outcome = simulate_outcome(decision, df_fwd, funding_df, forward_bars, interval_h,
                                   position_size_pct=size_pct)
        if outcome is None:
            continue

        rec = {
            "timestamp": bar_time.isoformat(),
            "symbol": symbol, "interval": primary_interval,
            "price": snap.price,
            "bias": decision.bias,
            "confidence": decision.confidence,
            "long_score": decision.long_score,
            "short_score": decision.short_score,
            "local_regime": snap.regime,
            "leverage": decision.leverage,
            "stop_price": decision.stop_price,
            "tp_primary": decision.tp_primary,
            "tp_stretch": decision.tp_stretch,
            "risk_pct": decision.risk_pct,
            "reward_pct_primary": decision.reward_pct_primary,
            "rr_primary": decision.rr_primary,
            "rsi_main": round(snap.rsi_main, 1),
            "div_main": snap.div_main,
            "macd_cross_main": snap.macd_cross_main,
            "macd_hist_main": round(snap.macd_hist_main, 4),
            "macd_improving_main": snap.macd_improving_main,
            "ema_status_main": snap.ema_status_main,
            "atr_main": round(snap.atr_main, 4),
            "vol_ratio_main": round(snap.vol_ratio_main, 3),
            "rsi_1h": round(snap.rsi_1h, 1),
            "div_1h": snap.div_1h,
            "funding_rate": round(snap.funding_rate, 4),
            "funding_avg_3p": round(snap.funding_avg_3p, 4),
            "fib_nearest": snap.fib_nearest,
            "reasons": "; ".join(decision.reasons),
            **outcome,
        }
        trades.append(rec)

        # Lock out new signals until BOTH (a) the previous trade has actually exited
        # AND (b) the configured cooldown has elapsed. Without this, when forward_bars
        # exceeds cooldown_bars, a new trade can fire while the previous one is still
        # notionally open, double-counting in cumulative P&L.
        actual_hold_bars = outcome["outcome_hold_bars"]
        exit_time     = bar_time + timedelta(hours=actual_hold_bars * interval_h)
        cooldown_time = bar_time + timedelta(hours=cooldown_bars   * interval_h)
        cooldown_until = max(exit_time, cooldown_time)

        if verbose and i - last_progress >= 500:
            print(f"  bar {i}/{len(walk_bars)} ({i / len(walk_bars) * 100:.0f}%) | signals={len(trades)}")
            last_progress = i

    df_out = pd.DataFrame(trades)
    if verbose:
        print(f"Done. {len(df_out)} trades.")
    return df_out


# ====================================================================
# 6. ANALYZE — performance summary + walk-forward split
# ====================================================================
def summarize(df: pd.DataFrame, label: str = "Backtest") -> dict:
    """Print a compact performance summary, return key stats."""
    if len(df) == 0:
        print(f"\n=== {label}: NO TRADES ===\n")
        return {}

    n = len(df)
    nets = df["outcome_net_pct"].astype(float).values
    raws = df["outcome_pnl_pct"].astype(float).values

    wins = int((nets > 0).sum())
    tp_hits = int(df["outcome_result"].isin(["tp_primary", "tp_stretch"]).sum())
    stops = int((df["outcome_result"] == "stop").sum())
    timeouts = int((df["outcome_result"] == "timeout").sum())

    mean_net = float(nets.mean())
    median_net = float(np.median(nets))
    sum_net = float(nets.sum())
    sharpe_ish = mean_net / nets.std() if nets.std() > 0 else 0.0

    win_rate_raw = float((raws > 0).sum() / n * 100)
    win_rate_net = wins / n * 100

    rrs = df["rr_primary"].dropna().astype(float).values
    median_rr = float(np.median(rrs)) if len(rrs) else 0.0

    cum = nets.cumsum()
    peak = np.maximum.accumulate(cum)
    drawdown = cum - peak
    max_dd = float(drawdown.min())

    print(f"\n{'=' * 68}")
    print(f"{label}")
    print(f"{'=' * 68}")
    print(f"Trades:          {n}")
    print(f"TP hits:         {tp_hits} ({tp_hits / n * 100:.1f}%)")
    print(f"Stops:           {stops} ({stops / n * 100:.1f}%)")
    print(f"Timeouts:        {timeouts} ({timeouts / n * 100:.1f}%)")
    print(f"Net win rate:    {win_rate_net:.1f}%")
    print(f"Raw win rate:    {win_rate_raw:.1f}%")
    print(f"Mean net %:      {mean_net:+.3f}%")
    print(f"Median net %:    {median_net:+.3f}%")
    print(f"Sum net %:       {sum_net:+.2f}%   (equal-$-risk cumulative)")
    print(f"Sharpe-ish:      {sharpe_ish:.3f}   (per-trade mean / stdev)")
    print(f"Max drawdown:    {max_dd:+.2f}%")
    print(f"Median R:R:      {median_rr:.2f}")

    print("\nBy direction:")
    for b in ["long", "short"]:
        sub = df[df["bias"] == b]
        if len(sub) == 0:
            continue
        sn = sub["outcome_net_pct"].astype(float).values
        st = sub["outcome_result"].isin(["tp_primary", "tp_stretch"]).sum()
        print(f"  {b.upper():5s} n={len(sub):4d} TP={int(st):3d} ({st / len(sub) * 100:.0f}%)  "
              f"mean={sn.mean():+.2f}%  sum={sn.sum():+.2f}%")

    print("\nBy regime:")
    for r in sorted(df["local_regime"].dropna().unique()):
        sub = df[df["local_regime"] == r]
        sn = sub["outcome_net_pct"].astype(float).values
        st = sub["outcome_result"].isin(["tp_primary", "tp_stretch"]).sum()
        print(f"  {str(r).upper():8s} n={len(sub):4d} TP={int(st):3d} ({st / len(sub) * 100:.0f}%)  "
              f"mean={sn.mean():+.2f}%")

    print("\nBy confidence:")
    for lo, hi in [(0, 6.5), (6.5, 7.0), (7.0, 7.5), (7.5, 99)]:
        sub = df[(df["confidence"] >= lo) & (df["confidence"] < hi)]
        if len(sub) == 0:
            continue
        sn = sub["outcome_net_pct"].astype(float).values
        st = sub["outcome_result"].isin(["tp_primary", "tp_stretch"]).sum()
        bucket = f"{lo:.1f}-{hi:.1f}" if hi < 99 else f">={lo:.1f}"
        print(f"  conf {bucket:12s} n={len(sub):4d} TP={int(st):3d} ({st / len(sub) * 100:.0f}%)  "
              f"mean={sn.mean():+.2f}%")

    print(f"{'=' * 68}\n")

    return {
        "n": n, "mean_net": mean_net, "sum_net": sum_net,
        "win_rate_net": win_rate_net, "tp_hit_rate": tp_hits / n * 100,
        "max_dd": max_dd, "median_rr": median_rr, "sharpe_ish": sharpe_ish,
    }


def walk_forward_split(df, train_frac=0.7):
    """Time-based train/test split (no shuffling)."""
    if "timestamp" not in df.columns:
        raise ValueError("df needs 'timestamp' column")
    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = int(len(df) * train_frac)
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


# ====================================================================
# 7. LIVE MODE — manual on-demand signal check
# ====================================================================
def format_live_output(snap: Snapshot, decision: Optional[Decision]) -> str:
    """Pretty-print current state and decision."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"{snap.symbol} {snap.interval} Advisor — {snap.timestamp:%Y-%m-%d %H:%M UTC}")
    lines.append(f"{'=' * 70}")
    lines.append(f"Price: ${snap.price:.4f}")
    lines.append("")

    daily_pos = "above" if snap.daily_price > snap.daily_ema50 else "below"
    lines.append(f"Regime: {snap.regime.upper()}  (weekly RSI {snap.weekly_rsi:.1f}, "
                 f"daily price {daily_pos} EMA50)")
    lines.append(f"  {snap.interval}:  RSI {snap.rsi_main:.1f} | div: {snap.div_main} | "
                 f"MACD: {snap.macd_cross_main} | EMA: {snap.ema_status_main}")
    lines.append(f"  1H:  RSI {snap.rsi_1h:.1f} | div: {snap.div_1h}")
    lines.append(f"  ATR: ${snap.atr_main:.4f} | Vol ratio: {snap.vol_ratio_main:.2f}x")
    lines.append(f"  Fib nearest: {snap.fib_nearest} (${snap.fib_nearest_price:.2f})")
    lines.append("")
    lines.append(f"Funding: {snap.funding_rate:+.4f}% latest | {snap.funding_avg_3p:+.4f}% (3-period avg)")
    lines.append("")

    if decision is None:
        lines.append("NO SIGNAL — confluence does not meet threshold criteria")
    else:
        emoji = "[LONG]" if decision.bias == "long" else "[SHORT]"
        lines.append(f"{emoji} {decision.bias.upper()} signal | confidence {decision.confidence:.2f}")
        lines.append(f"  Score: {decision.long_score:.2f}")
        lines.append("")
        lines.append(f"  Entry:    ${decision.entry_price:.4f}")
        lines.append(f"  Stop:     ${decision.stop_price:.4f}  ({decision.risk_pct:.2f}% risk)")
        lines.append(f"  TP1:      ${decision.tp_primary:.4f}  ({decision.reward_pct_primary:.2f}% reward, R:R {decision.rr_primary})")
        lines.append(f"  TP2:      ${decision.tp_stretch:.4f}  (stretch)")
        lines.append(f"  Leverage: {decision.leverage}x suggested")
        lines.append("")
        lines.append("  Reasons:")
        for r in decision.reasons:
            lines.append(f"    {r}")

    lines.append(f"{'=' * 70}")
    return "\n".join(lines)


def tag_live_outcomes(log_path, df_main, funding_df, forward_bars, interval_h):
    """Retroactively tag outcomes for prior signals where enough forward data now exists."""
    if not Path(log_path).exists():
        return 0, 0

    with open(log_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    if not records:
        return 0, 0

    newly_tagged = 0
    modified = False

    for r in records:
        if r.get("outcome_tagged"):
            continue

        sig_time = pd.Timestamp(r["timestamp"])
        if sig_time.tzinfo is None:
            sig_time = sig_time.tz_localize("UTC")
        else:
            sig_time = sig_time.tz_convert("UTC")

        # Forward bars after the signal
        fwd = df_main[df_main.index > sig_time]
        if len(fwd) < forward_bars:
            # Need to confirm enough wall-clock time has passed for a timeout
            hours_passed = (datetime.now(timezone.utc) - sig_time).total_seconds() / 3600
            if hours_passed < forward_bars * interval_h:
                continue  # not enough time yet; check again next run

        # Build a Decision-like namespace for simulate_outcome
        from types import SimpleNamespace
        decision_proxy = SimpleNamespace(
            entry_price=r["price"],
            stop_price=r["stop_price"],
            tp_primary=r["tp_primary"],
            tp_stretch=r["tp_stretch"],
            bias=r["bias"],
            leverage=r["leverage"],
        )

        outcome = simulate_outcome(decision_proxy, fwd.iloc[:forward_bars],
                                   funding_df, forward_bars, interval_h,
                                   position_size_pct=POSITION_SIZE_PCT)
        if outcome is None:
            continue

        r["outcome_tagged"] = True
        r.update(outcome)
        newly_tagged += 1
        modified = True

    if modified:
        with open(log_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")

    total_tagged = sum(1 for r in records if r.get("outcome_tagged"))
    return newly_tagged, total_tagged


def append_live_signal(log_path, snap, decision):
    """Append signal to forward log. Dedup: skip if same bar already logged."""
    if Path(log_path).exists():
        with open(log_path) as f:
            existing = [json.loads(l) for l in f if l.strip()]
        if existing:
            last_ts = pd.Timestamp(existing[-1]["timestamp"])
            cur_ts = pd.Timestamp(snap.timestamp)
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            if cur_ts.tzinfo is None:
                cur_ts = cur_ts.tz_localize("UTC")
            if last_ts == cur_ts:
                return False  # already logged this bar

    rec = {
        "timestamp": snap.timestamp.isoformat() if hasattr(snap.timestamp, "isoformat") else str(snap.timestamp),
        "symbol": snap.symbol,
        "interval": snap.interval,
        "price": snap.price,
        "bias": decision.bias,
        "confidence": decision.confidence,
        "long_score": decision.long_score,
        "short_score": decision.short_score,
        "local_regime": snap.regime,
        "leverage": decision.leverage,
        "stop_price": decision.stop_price,
        "tp_primary": decision.tp_primary,
        "tp_stretch": decision.tp_stretch,
        "risk_pct": decision.risk_pct,
        "reward_pct_primary": decision.reward_pct_primary,
        "rr_primary": decision.rr_primary,
        "rsi_main": round(snap.rsi_main, 1),
        "div_main": snap.div_main,
        "macd_cross_main": snap.macd_cross_main,
        "macd_hist_main": round(snap.macd_hist_main, 4),
        "macd_improving_main": snap.macd_improving_main,
        "ema_status_main": snap.ema_status_main,
        "atr_main": round(snap.atr_main, 4),
        "vol_ratio_main": round(snap.vol_ratio_main, 3),
        "rsi_1h": round(snap.rsi_1h, 1),
        "div_1h": snap.div_1h,
        "funding_rate": round(snap.funding_rate, 4),
        "funding_avg_3p": round(snap.funding_avg_3p, 4),
        "fib_nearest": snap.fib_nearest,
        "reasons": "; ".join(decision.reasons),
        "outcome_tagged": False,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return True


def print_live_log_stats(log_path):
    """Brief stats on the running forward log."""
    if not Path(log_path).exists():
        print("\nForward log: empty (this is your first signal)")
        return

    with open(log_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    if not records:
        print("\nForward log: empty")
        return

    n_total = len(records)
    tagged = [r for r in records if r.get("outcome_tagged")]
    n_tagged = len(tagged)
    pending = n_total - n_tagged

    print(f"\nForward log: {n_total} signals total | {n_tagged} with outcomes | {pending} pending")

    if n_tagged >= 5:
        wins = sum(1 for r in tagged if r.get("outcome_net_pct", 0) > 0)
        tp_hits = sum(1 for r in tagged if r.get("outcome_result") in ("tp_primary", "tp_stretch"))
        net_sum = sum(r.get("outcome_net_pct", 0) for r in tagged)
        net_mean = net_sum / n_tagged

        print(f"  Win rate: {wins}/{n_tagged} ({wins / n_tagged * 100:.0f}%) | "
              f"TP hits: {tp_hits} ({tp_hits / n_tagged * 100:.0f}%)")
        print(f"  Mean net: {net_mean:+.2f}%/trade  Cumulative: {net_sum:+.1f}%")


def _drop_incomplete_last_bar(df, interval):
    """If the last bar's interval hasn't fully elapsed yet, drop it.

    Binance returns the currently developing bar from futures_klines, and its OHLC
    keeps changing until the bar closes. Running RSI/MACD/divergence on that moving
    target causes signal flicker. Always drop it so we only decide on closed bars.
    """
    if df is None or df.empty:
        return df
    last_open = df.index[-1]
    bar_close_time = last_open + timedelta(hours=_interval_hours(interval))
    if datetime.now(timezone.utc) < bar_close_time:
        return df.iloc[:-1].copy()
    return df


def live_mode(args, cfg: RulesConfig):
    """Run a single on-demand check. Print state, log signal if any, tag old outcomes."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=240)  # 8 months — plenty of indicator history

    print(f"Fetching latest {args.symbol} {args.interval} data...")

    # Live mode passes max_cache_staleness_hours so we don't decide on stale data.
    fresh = None if args.no_cache else LIVE_CACHE_MAX_AGE_HOURS
    df_main = fetch_klines(args.symbol, args.interval, start, end,
                           use_cache=not args.no_cache,
                           max_cache_staleness_hours=fresh)
    df_1h = (fetch_klines(args.symbol, "1h", start, end,
                          use_cache=not args.no_cache,
                          max_cache_staleness_hours=fresh)
             if args.interval != "1h" else df_main)
    df_daily = fetch_klines(args.symbol, "1d",
                            start - timedelta(days=200), end,
                            use_cache=not args.no_cache,
                            max_cache_staleness_hours=fresh)
    df_weekly = fetch_klines(args.symbol, "1w",
                             start - timedelta(days=400), end,
                             use_cache=not args.no_cache,
                             max_cache_staleness_hours=fresh)
    funding_df = fetch_funding(args.symbol, start, end, use_cache=not args.no_cache)

    if df_main.empty:
        print("No data fetched")
        return

    # Drop the currently-developing bar on every timeframe. Indicators on a
    # not-yet-closed bar mutate as the bar develops, causing flicker.
    df_main   = _drop_incomplete_last_bar(df_main,   args.interval)
    if args.interval != "1h":
        df_1h = _drop_incomplete_last_bar(df_1h,   "1h")
    df_daily  = _drop_incomplete_last_bar(df_daily,  "1d")
    df_weekly = _drop_incomplete_last_bar(df_weekly, "1w")

    if df_main is None or df_main.empty:
        print("No closed bars available after dropping in-progress bar")
        return

    interval_h = _interval_hours(args.interval)
    bar_time = df_main.index[-1]

    snap = build_snapshot(args.symbol, args.interval, df_main, df_1h,
                          df_daily, df_weekly, funding_df, bar_time)
    if snap is None:
        print("Insufficient data to build snapshot")
        return

    decision = decide(snap, cfg)

    log_path = args.log_file or f"live_signals_{args.symbol}_{args.interval}.jsonl"

    # Tag prior outcomes first
    newly, total = tag_live_outcomes(log_path, df_main, funding_df,
                                     args.forward_bars, interval_h)
    if newly > 0:
        print(f"Tagged {newly} new outcomes from prior signals "
              f"({total} total tagged)\n")

    # Print current state
    print(format_live_output(snap, decision))

    # Log new signal (with dedup)
    if decision is not None and not args.no_log:
        added = append_live_signal(log_path, snap, decision)
        if added:
            print(f"\nSignal logged to {log_path}")
        else:
            print(f"\nSignal already logged for this bar — no duplicate added")

    # Brief log summary
    print_live_log_stats(log_path)


def loop_mode(args, cfg: RulesConfig):
    """
    Run live_mode forever in a loop, sleeping `check_minutes` between iterations.
    Errors in any one iteration are caught and printed; the loop continues.
    """
    from time import sleep
    sleep_seconds = max(60, int(args.check_minutes * 60))

    print(f"\n{'#' * 70}")
    print(f"# Live loop mode — {args.symbol} {args.interval}")
    print(f"# Config: threshold_mod={cfg.threshold_mod}, "
          f"tp_primary={cfg.tp_atr_mult_primary}x ATR, size={cfg.position_size_pct}%")
    print(f"# Checking every {args.check_minutes} minutes")
    print(f"# Logging to: live_signals_{args.symbol}_{args.interval}.jsonl")
    print(f"# Press Ctrl+C to stop")
    print(f"{'#' * 70}\n")

    iteration = 0
    while True:
        iteration += 1
        try:
            print(f"\n--- Check #{iteration} at {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} ---")
            live_mode(args, cfg)
        except KeyboardInterrupt:
            print("\n\nStopped by user. Forward log preserved.")
            return
        except Exception as e:
            print(f"\nError this iteration: {e}")
            print("(continuing — next check on schedule)")

        next_run = datetime.now(timezone.utc) + timedelta(seconds=sleep_seconds)
        print(f"\nNext check in {args.check_minutes} min (at {next_run:%H:%M UTC})...")
        try:
            sleep(sleep_seconds)
        except KeyboardInterrupt:
            print("\n\nStopped by user. Forward log preserved.")
            return


# ====================================================================
# 8. CLI ENTRYPOINT — kept intentionally tiny.
# Everything except mode lives in the USER CONFIG block at the top.
# ====================================================================
def _runtime_args(symbol: str):
    """Synthesize the args namespace that live_mode/loop_mode expect from baked-in constants."""
    from types import SimpleNamespace
    return SimpleNamespace(
        symbol=symbol,
        interval=INTERVAL,
        forward_bars=FORWARD_BARS,
        cooldown_bars=COOLDOWN_BARS,
        check_minutes=CHECK_MINUTES,
        no_cache=False,
        no_log=False,
        log_file=None,
    )


def main():
    p = argparse.ArgumentParser(
        description="SOL perp signal advisor + backtester.\n"
                    "Defaults to live-loop mode. All tuning is in the USER CONFIG block "
                    "at the top of the file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python advisor_v2.py                    # live loop (default)\n"
            f"  python advisor_v2.py --check            # one-shot check, no loop\n"
            f"  python advisor_v2.py --backtest         # backtest, {DEFAULT_BACKTEST_DAYS} days\n"
            f"  python advisor_v2.py --backtest --days 365\n"
            "  python advisor_v2.py --symbol BTCUSDT --check\n"
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--backtest", action="store_true",
                      help=f"Run a backtest over the last --days (default {DEFAULT_BACKTEST_DAYS})")
    mode.add_argument("--check", action="store_true",
                      help="Run a single live check and exit (no loop)")
    p.add_argument("--days", type=int, default=DEFAULT_BACKTEST_DAYS,
                   help=f"(--backtest) days of history. Default {DEFAULT_BACKTEST_DAYS}")
    p.add_argument("--symbol", default=SYMBOL,
                   help=f"Override symbol from USER CONFIG. Default {SYMBOL}")

    args = p.parse_args()
    cfg = RulesConfig()  # picks up everything from USER CONFIG

    # ── Live loop (default) ──
    if not args.backtest and not args.check:
        loop_mode(_runtime_args(args.symbol), cfg)
        return

    # ── One-shot live check ──
    if args.check:
        live_mode(_runtime_args(args.symbol), cfg)
        return

    # ── Backtest ──
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    out_path = f"backtest_{args.symbol}_{INTERVAL}.jsonl"

    print(f"Backtest config:")
    print(f"  symbol={args.symbol}  interval={INTERVAL}  days={args.days}")
    print(f"  thresholds: high={cfg.threshold_high} mod={cfg.threshold_mod}  min_rr={cfg.min_rr}")
    print(f"  geom: stop={cfg.atr_stop_mult}xATR tp1={cfg.tp_atr_mult_primary}xATR tp2={cfg.tp_atr_mult_stretch}xATR")
    print(f"  forward_bars={FORWARD_BARS}  cooldown_bars={COOLDOWN_BARS}")
    print()

    df = run_backtest(
        symbol=args.symbol,
        primary_interval=INTERVAL,
        start=start, end=end,
        rules_config=cfg,
        forward_bars=FORWARD_BARS,
        cooldown_bars=COOLDOWN_BARS,
        use_cache=True,
        verbose=True,
    )

    if df.empty:
        print("No trades generated.")
        return

    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            rec = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"\nSaved {len(df)} trades to {out_path}")

    summarize(df, label=f"{args.symbol} {INTERVAL} ({args.days}d, full)")

    if len(df) >= 20:
        train, test = walk_forward_split(df, train_frac=0.7)
        summarize(train, label="Train (first 70%)")
        summarize(test, label="Test (last 30% — out of sample)")


if __name__ == "__main__":
    main()