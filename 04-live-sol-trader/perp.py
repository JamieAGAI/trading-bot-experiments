"""
swing_perp.py — leveraged perp version of v10 swing bot for SOLUSDT.

Three modes:
  - default (backtest):       python swing_perp.py [--years N] [--leverage X]
  - one-shot manual check:    python swing_perp.py --live
  - continuous shadow loop:   python swing_perp.py --loop [--email]

Strategy is a faithful port of v10 (your main.py): pullback-buy in confirmed
uptrends, breakeven shift at +1R, trailing stop after +2R, take-profit at
+2.5R, trend/daily/timeout exits.

Perp-specific extensions: leverage, funding cost, liquidation guardrail,
lower fees (0.04% perp taker).

Email setup: put EMAIL_FROM / EMAIL_TO / EMAIL_PASS in .env. Use Gmail app
password (not your real password). Pass --email to enable alerts.

Forward shadow log: live_swing_perp_SOLUSDT.jsonl
State file:         live_swing_state.json
"""

import argparse
import json
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from time import sleep
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONSTANTS — STRATEGY (port of v10 main.py)
# ============================================================================
SYMBOL              = 'SOLUSDT'
INTERVAL_4H         = '4h'
INTERVAL_1D         = '1d'

EMA_TREND           = 35
EMA_VALUE           = 15
EMA_DAILY           = 20
RSI_LEN             = 14
ATR_LEN             = 14

REWARD_RATIO        = 3.0
ATR_BUFFER          = 0.45
MAX_HOLD_BARS       = 50

RSI_MIN             = 53
RSI_MAX             = 65
VOLUME_MULTIPLIER   = 1.6
VALUE_ZONE_PCT      = 0.05

TRAIL_ACTIVATION_R  = 3
TRAIL_ATR_MULT      = 3.0
TRAIL_LOOKBACK_BARS = 3

# ============================================================================
# CONSTANTS — PERP-SPECIFIC
# ============================================================================
LEVERAGE            = 3.0
POSITION_SIZE       = 1.0
MAX_RISK_PCT        = 0.10
FEE_PCT_PER_SIDE    = 0.0004
USE_FUNDING_COST    = True
LIQUIDATION_BUFFER  = 1.05

# ============================================================================
# CONSTANTS — CAPITAL / BACKTEST / LIVE
# ============================================================================
INITIAL_CAPITAL     = 5000
BACKTEST_YEARS      = 2

CACHE_DIR  = Path(os.getenv("ADVISOR_CACHE_DIR", "data_cache"))
CACHE_DIR.mkdir(exist_ok=True)

STATE_FILE = "live_swing_state.json"
LOG_FILE   = f"live_swing_perp_{SYMBOL}.jsonl"
CHECK_MINUTES = 60

# ============================================================================
# CONSTANTS — EMAIL (set in .env)
# ============================================================================
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO   = os.getenv("EMAIL_TO")
EMAIL_PASS = os.getenv("EMAIL_PASS")
SMTP_HOST  = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT  = int(os.getenv("SMTP_PORT", "587"))


# ============================================================================
# DATA LOADER (Binance futures only)
# ============================================================================
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


def fetch_klines(symbol, interval, start, end, use_cache=True) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"{symbol}_{interval}_klines.parquet"
    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not cached.empty and cached.index.min() <= start and cached.index.max() >= end - timedelta(days=2):
            return cached.loc[(cached.index >= start) & (cached.index < end)].copy()

    client = _get_client()
    all_chunks = []
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    bar_ms = _interval_to_ms(interval)
    fail = 0

    while cursor_ms < end_ms:
        try:
            chunk = client.futures_klines(symbol=symbol, interval=interval,
                                          startTime=cursor_ms, endTime=end_ms, limit=1000)
            fail = 0
        except Exception as e:
            fail += 1
            if fail > 5:
                raise
            print(f"  klines fetch failed ({e}), retry...")
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
    cache_path = CACHE_DIR / f"{symbol}_funding.parquet"
    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not cached.empty and cached.index.min() <= start and cached.index.max() >= end - timedelta(days=2):
            return cached.loc[(cached.index >= start) & (cached.index < end)].copy()

    client = _get_client()
    all_chunks = []
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    fail = 0

    while cursor_ms < end_ms:
        try:
            chunk = client.futures_funding_rate(symbol=symbol, startTime=cursor_ms,
                                                endTime=end_ms, limit=1000)
            fail = 0
        except Exception as e:
            fail += 1
            if fail > 5:
                raise
            print(f"  funding fetch failed ({e}), retry...")
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


# ============================================================================
# INDICATORS + SIGNAL
# ============================================================================
def compute_indicators(df_4h, df_daily) -> pd.DataFrame:
    df = df_4h.copy()
    df['ema_trend'] = ta.ema(df['close'], length=EMA_TREND)
    df['ema_value'] = ta.ema(df['close'], length=EMA_VALUE)
    df['rsi']       = ta.rsi(df['close'], length=RSI_LEN)
    df['atr']       = ta.atr(df['high'], df['low'], df['close'], length=ATR_LEN)
    df['vol_ma']    = df['volume'].rolling(20, min_periods=1).mean()
    df['bullish']   = (df['close'] > df['open']).astype(int)

    daily = df_daily.copy()
    daily['ema_daily']  = ta.ema(daily['close'], length=EMA_DAILY)
    daily['daily_bull'] = (daily['close'] > daily['ema_daily']).astype(int)

    if daily.index.tz is None:
        daily.index = daily.index.tz_localize('UTC')
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    # Shift by 1 day to prevent look-ahead bias: 4H bars on day X see
    # daily_bull from day X-1's close, not day X's (which isn't known yet)
    df['daily_bull'] = daily['daily_bull'].shift(1).reindex(df.index, method='ffill')
    return df.dropna()


def buy_signal(df, i) -> bool:
    if i < 1:
        return False
    c, p1 = df.iloc[i], df.iloc[i-1]
    recent_low = df.iloc[max(0, i-8):i+1]['low'].min()
    in_value_zone = recent_low <= c['ema_value'] * (1.0 + VALUE_ZONE_PCT)
    return (
        c['daily_bull'] == 1 and c['close'] > c['ema_trend'] and in_value_zone
        and c['low'] > p1['low'] and c['close'] > p1['high'] and c['bullish'] == 1
        and RSI_MIN < c['rsi'] < RSI_MAX
        and c['volume'] >= c['vol_ma'] * VOLUME_MULTIPLIER
        and c['close'] > c['ema_value']
    )


# ============================================================================
# BACKTEST
# ============================================================================
@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    risk_per_unit: float
    units: float
    margin_used: float
    exposure: float
    leverage: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    bars_held: int = 0
    at_be: bool = False
    used_trail: bool = False
    gross_pnl: float = 0.0
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    funding_cost: float = 0.0
    net_pnl: float = 0.0
    capital_after: float = 0.0
    liquidated: bool = False


def backtest_perp(df, funding_df, initial_capital=INITIAL_CAPITAL,
                  leverage=LEVERAGE, position_size=POSITION_SIZE,
                  fee_pct=FEE_PCT_PER_SIDE, max_risk_pct=MAX_RISK_PCT,
                  use_funding=USE_FUNDING_COST, verbose=True) -> tuple:
    capital = initial_capital
    in_position = False
    skipped_max_risk = skipped_liquidation = 0
    entry_price = stop_price = target_price = 0.0
    risk_per_unit = be_trigger = trail_stop = 0.0
    units = margin_used = exposure = 0.0
    entry_bar_idx = 0
    entry_time = None
    at_be = used_trail = False
    trades = []

    for i in range(8, len(df) - 1):
        bar = df.iloc[i]

        if not in_position and buy_signal(df, i):
            next_open = df.iloc[i+1]['open']
            atr_val   = bar['atr']
            swing_low = df.iloc[i-2:i+1]['low'].min()
            sp = swing_low - (ATR_BUFFER * atr_val)
            r_unit = next_open - sp
            risk_pct = r_unit / next_open if next_open > 0 else 1.0

            if r_unit <= 0 or risk_pct > max_risk_pct:
                skipped_max_risk += 1
                continue
            if leverage > 1.0:
                liq_price = next_open * (1.0 - 1.0 / leverage)
                if sp < liq_price * LIQUIDATION_BUFFER:
                    skipped_liquidation += 1
                    continue

            entry_price, stop_price = next_open, sp
            risk_per_unit = r_unit
            target_price = entry_price + (REWARD_RATIO * risk_per_unit)
            be_trigger = entry_price + risk_per_unit
            trail_stop = stop_price
            margin_used = capital * position_size
            exposure = margin_used * leverage
            units = exposure / entry_price
            entry_bar_idx = i + 1
            entry_time = df.index[i+1]
            at_be = used_trail = False
            in_position = True

        elif in_position:
            cur = df.iloc[i]
            bars_held = i - entry_bar_idx

            if not at_be and cur['high'] >= be_trigger:
                stop_price = entry_price
                at_be = True

            if cur['close'] > entry_price + (TRAIL_ACTIVATION_R * risk_per_unit):
                ws = max(0, i - (TRAIL_LOOKBACK_BARS - 1))
                new_trail = df.iloc[ws:i+1]['high'].max() - (TRAIL_ATR_MULT * cur['atr'])
                trail_stop = max(trail_stop, new_trail)
                used_trail = trail_stop > stop_price

            effective_stop = max(stop_price, trail_stop)
            exit_price = exit_reason = None
            liquidated = False

            if leverage > 1.0:
                liq_price = entry_price * (1.0 - 1.0 / leverage)
                if cur['low'] <= liq_price and liq_price > effective_stop:
                    exit_price, exit_reason, liquidated = liq_price, 'LIQUIDATION', True

            if exit_price is None:
                if cur['low'] <= effective_stop:
                    exit_price = effective_stop
                    if used_trail and cur['low'] <= trail_stop and trail_stop > stop_price:
                        exit_reason = 'Trailing Stop'
                    elif at_be:
                        exit_reason = 'Breakeven Stop'
                    else:
                        exit_reason = 'Stop Loss'
                elif cur['high'] >= target_price:
                    exit_price, exit_reason = target_price, 'Take Profit'
                elif i > 0 and cur['close'] < cur['ema_trend'] and df.iloc[i-1]['close'] < df.iloc[i-1]['ema_trend']:
                    exit_price, exit_reason = cur['close'], 'Trend Exit'
                elif cur['daily_bull'] == 0:
                    exit_price, exit_reason = cur['close'], 'Daily Trend Exit'
                elif bars_held >= MAX_HOLD_BARS:
                    exit_price, exit_reason = cur['close'], 'Timeout'

            if exit_price is not None:
                price_pct = (exit_price - entry_price) / entry_price
                gross_pnl = price_pct * exposure
                entry_fee = entry_price * units * fee_pct
                exit_fee  = exit_price  * units * fee_pct
                funding_cost = 0.0
                if use_funding and funding_df is not None and len(funding_df) > 0:
                    events = funding_df[(funding_df.index >= entry_time) & (funding_df.index <= df.index[i])]
                    for _, row in events.iterrows():
                        funding_cost += float(row['funding_rate']) * units * entry_price

                net_pnl = -margin_used if liquidated else (gross_pnl - entry_fee - exit_fee - funding_cost)
                capital += net_pnl

                trades.append(Trade(
                    entry_time=entry_time, entry_price=entry_price,
                    stop_price=stop_price, target_price=target_price,
                    risk_per_unit=risk_per_unit, units=units,
                    margin_used=margin_used, exposure=exposure, leverage=leverage,
                    exit_time=df.index[i], exit_price=exit_price, exit_reason=exit_reason,
                    bars_held=bars_held, at_be=at_be, used_trail=used_trail,
                    gross_pnl=gross_pnl, entry_fee=entry_fee, exit_fee=exit_fee,
                    funding_cost=funding_cost, net_pnl=net_pnl,
                    capital_after=capital, liquidated=liquidated,
                ))
                in_position = False

                if capital <= initial_capital * 0.05:
                    if verbose:
                        print(f"  *** account zeroed ({capital:.2f}) — stopping ***")
                    break

    if in_position:
        last = df.iloc[-1]
        exit_price = last['close']
        price_pct = (exit_price - entry_price) / entry_price
        gross_pnl = price_pct * exposure
        entry_fee = entry_price * units * fee_pct
        exit_fee  = exit_price  * units * fee_pct
        funding_cost = 0.0
        if use_funding and funding_df is not None and len(funding_df) > 0:
            events = funding_df[(funding_df.index >= entry_time) & (funding_df.index <= df.index[-1])]
            for _, row in events.iterrows():
                funding_cost += float(row['funding_rate']) * units * entry_price
        net_pnl = gross_pnl - entry_fee - exit_fee - funding_cost
        capital += net_pnl
        trades.append(Trade(
            entry_time=entry_time, entry_price=entry_price,
            stop_price=stop_price, target_price=target_price,
            risk_per_unit=risk_per_unit, units=units,
            margin_used=margin_used, exposure=exposure, leverage=leverage,
            exit_time=df.index[-1], exit_price=exit_price, exit_reason='EOD',
            bars_held=len(df) - 1 - entry_bar_idx, at_be=at_be, used_trail=used_trail,
            gross_pnl=gross_pnl, entry_fee=entry_fee, exit_fee=exit_fee,
            funding_cost=funding_cost, net_pnl=net_pnl,
            capital_after=capital, liquidated=False,
        ))

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    summary = {
        'final_capital': capital, 'initial_capital': initial_capital,
        'total_return_pct': (capital / initial_capital - 1) * 100,
        'n_trades': len(trades),
        'skipped_max_risk': skipped_max_risk,
        'skipped_liquidation': skipped_liquidation,
    }
    return trades_df, None, summary


def summarize_trades(trades_df, summary, label="Backtest"):
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    print(f"Initial capital: ${summary['initial_capital']:,.2f}")
    print(f"Final capital:   ${summary['final_capital']:,.2f}")
    print(f"Total return:    {summary['total_return_pct']:+.2f}%")
    print(f"Trades fired:    {summary['n_trades']}")
    print(f"Skipped (risk):  {summary['skipped_max_risk']}")
    print(f"Skipped (liq):   {summary['skipped_liquidation']}")
    if len(trades_df) == 0:
        print("\nNo trades.\n")
        return
    wins = trades_df[trades_df['net_pnl'] > 0]
    losses = trades_df[trades_df['net_pnl'] <= 0]
    wr = len(wins) / len(trades_df) * 100
    print(f"\nWin rate:        {wr:.1f}% ({len(wins)} W / {len(losses)} L)")
    print(f"Avg win:         ${wins['net_pnl'].mean():,.2f}" if len(wins) else "Avg win: —")
    print(f"Avg loss:        ${losses['net_pnl'].mean():,.2f}" if len(losses) else "Avg loss: —")
    if len(wins) and len(losses):
        pf = abs(wins['net_pnl'].sum() / losses['net_pnl'].sum())
        print(f"Profit factor:   {pf:.2f}")
    eq = np.concatenate([[summary['initial_capital']], trades_df['capital_after'].values])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    print(f"Max drawdown:    {dd.min():.2f}%")
    pnl_pcts = trades_df['net_pnl'] / trades_df['margin_used'] * 100
    print(f"Mean trade %:    {pnl_pcts.mean():+.2f}%")
    print(f"Median trade %:  {pnl_pcts.median():+.2f}%")
    print(f"Best trade %:    {pnl_pcts.max():+.2f}%")
    print(f"Worst trade %:   {pnl_pcts.min():+.2f}%")
    print("\nExit reasons:")
    for r, n in trades_df['exit_reason'].value_counts().items():
        sub = trades_df[trades_df['exit_reason'] == r]
        print(f"  {r:18s}  n={n:3d}  avg_pnl=${sub['net_pnl'].mean():+9.2f}  avg_bars={sub['bars_held'].mean():.1f}")
    tf = trades_df['funding_cost'].sum()
    fe = (trades_df['entry_fee'] + trades_df['exit_fee']).sum()
    print(f"\nGross: ${trades_df['gross_pnl'].sum():+,.2f}  Fees: -${fe:,.2f}  Funding: -${tf:,.2f}")
    print(f"Net:   ${trades_df['net_pnl'].sum():+,.2f}\n{'='*72}\n")


def save_trades(trades_df, out_path):
    if len(trades_df) == 0:
        return
    rec = trades_df.copy()
    for col in ['entry_time', 'exit_time']:
        if col in rec.columns:
            rec[col] = rec[col].astype(str)
    with open(out_path, 'w') as f:
        for _, row in rec.iterrows():
            f.write(json.dumps(row.to_dict(), default=str) + '\n')
    print(f"Saved {len(rec)} trades to {out_path}")


# ============================================================================
# EMAIL
# ============================================================================
def send_alert(subject: str, body: str) -> bool:
    if not (EMAIL_FROM and EMAIL_TO and EMAIL_PASS):
        print(f"  [email disabled — set EMAIL_FROM/EMAIL_TO/EMAIL_PASS in .env]")
        return False
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From']    = EMAIL_FROM
        msg['To']      = EMAIL_TO
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls()
            s.login(EMAIL_FROM, EMAIL_PASS)
            s.send_message(msg)
        print(f"  ✓ Email sent: {subject}")
        return True
    except Exception as e:
        print(f"  ✗ Email failed: {e}")
        return False


def format_entry_alert(state, bar_time):
    subject = f"SOL Perp: NEW LONG @ ${state['entry_price']:.4f} ({state['leverage']}x)"
    body = (
        f"NEW LONG SIGNAL — {bar_time:%Y-%m-%d %H:%M UTC}\n"
        f"{'='*50}\n"
        f"Symbol:   {SYMBOL}\n"
        f"Entry:    ${state['entry_price']:.4f}\n"
        f"Stop:     ${state['stop_price']:.4f}  ({(state['stop_price']/state['entry_price']-1)*100:+.2f}%)\n"
        f"TP:       ${state['target_price']:.4f}  ({(state['target_price']/state['entry_price']-1)*100:+.2f}%, R:R 2.5)\n"
        f"BE trig:  ${state['be_trigger']:.4f}  (stop moves to entry here)\n\n"
        f"Leverage: {state['leverage']}x\n"
        f"Margin:   ${state['margin_used']:,.2f}\n"
        f"Units:    {state['units']:.4f} SOL\n\n"
        f"Risk at stop (leveraged): "
        f"{(state['stop_price']/state['entry_price']-1)*100*state['leverage']:+.2f}% of margin\n\n"
        f"Place market or limit order to enter near current price."
    )
    return subject, body


def format_exit_alert(state, exit_price, exit_reason, gross_pct, net_pnl, bar_time):
    pnl_sign = "+" if net_pnl >= 0 else ""
    subject = f"SOL Perp: EXIT — {exit_reason} ({pnl_sign}{net_pnl:.2f})"
    leveraged_pct = gross_pct * state['leverage']
    hold_hours = (bar_time - pd.Timestamp(state['entry_time'])).total_seconds() / 3600
    body = (
        f"EXIT — {exit_reason} — {bar_time:%Y-%m-%d %H:%M UTC}\n"
        f"{'='*50}\n"
        f"Symbol:    {SYMBOL}\n"
        f"Entry:     ${state['entry_price']:.4f}  ({pd.Timestamp(state['entry_time']):%Y-%m-%d %H:%M})\n"
        f"Exit:      ${exit_price:.4f}\n\n"
        f"Gross %:   {gross_pct*100:+.2f}% (raw)\n"
        f"Lev %:     {leveraged_pct*100:+.2f}% (at {state['leverage']}x)\n"
        f"Net PnL:   ${net_pnl:+,.2f}\n\n"
        f"Hold:      {hold_hours:.1f}h\n"
        f"Reason:    {exit_reason}\n"
        f"Status:    {'WIN' if net_pnl > 0 else 'LOSS'}"
    )
    return subject, body


# ============================================================================
# LIVE / LOOP MODE
# ============================================================================
def load_state(path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_state(state, path):
    with open(path, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def clear_state(path):
    if Path(path).exists():
        Path(path).unlink()


def append_log(record, log_path):
    with open(log_path, 'a') as f:
        f.write(json.dumps(record, default=str) + '\n')


def print_dashboard(df, state):
    i = len(df) - 1
    bar = df.iloc[i]
    bar_time = df.index[i]
    print(f"\n{'='*70}\nSOLUSDT 4H Swing Perp — {bar_time:%Y-%m-%d %H:%M UTC}\n{'='*70}")
    print(f"Price:      ${bar['close']:.4f}")
    print(f"4H RSI:     {bar['rsi']:.1f}  (signal range: {RSI_MIN}-{RSI_MAX})")
    print(f"4H EMA15:   ${bar['ema_value']:.4f}  ({'above' if bar['close'] > bar['ema_value'] else 'below'})")
    print(f"4H EMA35:   ${bar['ema_trend']:.4f}  ({'above' if bar['close'] > bar['ema_trend'] else 'below'})")
    print(f"Daily bull: {'YES' if bar['daily_bull'] == 1 else 'NO'}")
    print(f"ATR:        ${bar['atr']:.4f}")
    print(f"Vol ratio:  {bar['volume']/bar['vol_ma']:.2f}x  (need >= {VOLUME_MULTIPLIER}x)")
    if state:
        cur = bar['close']
        pnl_pct = (cur - state['entry_price']) / state['entry_price'] * 100
        eff_stop = max(state['stop_price'], state.get('trail_stop', state['stop_price']))
        print(f"\n  IN POSITION since {state['entry_time']}")
        print(f"  Entry: ${state['entry_price']:.4f}  Stop: ${eff_stop:.4f}  TP: ${state['target_price']:.4f}")
        print(f"  PnL: {pnl_pct:+.2f}% ({pnl_pct * state['leverage']:+.2f}% at {state['leverage']}x)")
        print(f"  At BE: {state.get('at_be', False)}  Trail active: {state.get('trail_stop', 0) > state['stop_price']}")
    else:
        print("\n  FLAT — no open position")
    print(f"{'='*70}")


def live_mode(args):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=180)
    print(f"\nFetching {SYMBOL} 4H + daily...")
    df_4h = fetch_klines(SYMBOL, INTERVAL_4H, start, end, use_cache=not args.no_cache)
    df_daily = fetch_klines(SYMBOL, INTERVAL_1D, start - timedelta(days=200), end, use_cache=not args.no_cache)
    if df_4h.empty or df_daily.empty:
        print("No data")
        return

    df = compute_indicators(df_4h, df_daily)
    if len(df) < 10:
        print("Not enough bars")
        return

    state_path = args.state_file or STATE_FILE
    log_path   = args.log_file or LOG_FILE
    state = load_state(state_path)
    print_dashboard(df, state)

    i = len(df) - 1
    bar = df.iloc[i]
    bar_time = df.index[i]

    # ── Flat: check entry ───────────────────────────────────────────
    if state is None:
        if i < 8 or not buy_signal(df, i):
            print("\nNo signal this bar.")
            return
        next_open = bar['close']
        atr_val = bar['atr']
        swing_low = df.iloc[i-2:i+1]['low'].min()
        sp = swing_low - (ATR_BUFFER * atr_val)
        r_unit = next_open - sp
        risk_pct = r_unit / next_open if next_open > 0 else 1.0
        if r_unit <= 0 or risk_pct > args.max_risk_pct:
            print(f"\nSignal fired but stop too far ({risk_pct*100:.1f}%) — skipped")
            return
        if args.leverage > 1.0:
            liq = next_open * (1.0 - 1.0 / args.leverage)
            if sp < liq * LIQUIDATION_BUFFER:
                print(f"\nSignal fired but stop too close to liquidation — skipped")
                return

        margin_used = args.initial_capital * args.position_size
        exposure = margin_used * args.leverage
        units = exposure / next_open
        new_state = {
            'entry_time': bar_time.isoformat(),
            'entry_price': next_open,
            'stop_price': sp,
            'target_price': next_open + (REWARD_RATIO * r_unit),
            'risk_per_unit': r_unit,
            'be_trigger': next_open + r_unit,
            'at_be': False,
            'trail_stop': sp,
            'units': units,
            'margin_used': margin_used,
            'exposure': exposure,
            'leverage': args.leverage,
        }
        save_state(new_state, state_path)
        append_log({'event': 'entry', 'time': bar_time.isoformat(), **new_state}, log_path)
        print(f"\n✓ NEW LONG SIGNAL — entry ${next_open:.4f}, stop ${sp:.4f}, TP ${new_state['target_price']:.4f}")
        if args.email:
            subj, body = format_entry_alert(new_state, bar_time)
            send_alert(subj, body)
        return

    # ── In position: check exits + stop updates ─────────────────────
    cur = bar
    modified = False
    if not state.get('at_be') and cur['high'] >= state['be_trigger']:
        state['stop_price'] = state['entry_price']
        state['at_be'] = True
        modified = True
        print("  → Breakeven shift: stop moved to entry")
        if args.email:
            send_alert(f"SOL Perp: Breakeven Activated @ ${state['entry_price']:.4f}",
                       f"Price hit {state['be_trigger']:.4f} (+1R). Stop moved to entry.")

    if cur['close'] > state['entry_price'] + (TRAIL_ACTIVATION_R * state['risk_per_unit']):
        ws = max(0, i - (TRAIL_LOOKBACK_BARS - 1))
        new_trail = df.iloc[ws:i+1]['high'].max() - (TRAIL_ATR_MULT * cur['atr'])
        if new_trail > state.get('trail_stop', state['stop_price']):
            state['trail_stop'] = new_trail
            modified = True
            print(f"  → Trail raised to ${new_trail:.4f}")

    if modified:
        save_state(state, state_path)

    eff_stop = max(state['stop_price'], state.get('trail_stop', state['stop_price']))
    exit_price = exit_reason = None

    if cur['low'] <= eff_stop:
        exit_price = eff_stop
        if state.get('trail_stop', 0) > state['stop_price'] and cur['low'] <= state['trail_stop']:
            exit_reason = 'Trailing Stop'
        elif state.get('at_be'):
            exit_reason = 'Breakeven Stop'
        else:
            exit_reason = 'Stop Loss'
    elif cur['high'] >= state['target_price']:
        exit_price, exit_reason = state['target_price'], 'Take Profit'
    elif i > 0 and cur['close'] < cur['ema_trend'] and df.iloc[i-1]['close'] < df.iloc[i-1]['ema_trend']:
        exit_price, exit_reason = cur['close'], 'Trend Exit'
    elif cur['daily_bull'] == 0:
        exit_price, exit_reason = cur['close'], 'Daily Trend Exit'
    else:
        bars_held = int((bar_time - pd.Timestamp(state['entry_time'])).total_seconds() / 3600 / 4)
        if bars_held >= MAX_HOLD_BARS:
            exit_price, exit_reason = cur['close'], 'Timeout'

    if exit_price is None:
        print("\n  No exit triggered.")
        return

    gross_pct = (exit_price - state['entry_price']) / state['entry_price']
    gross_pnl = gross_pct * state['exposure']
    entry_fee = state['entry_price'] * state['units'] * args.fee_pct
    exit_fee  = exit_price * state['units'] * args.fee_pct
    net_pnl   = gross_pnl - entry_fee - exit_fee

    append_log({
        'event': 'exit', 'time': bar_time.isoformat(),
        'entry_time': state['entry_time'], 'entry_price': state['entry_price'],
        'exit_price': exit_price, 'exit_reason': exit_reason,
        'gross_pct': gross_pct, 'leveraged_pct': gross_pct * state['leverage'],
        'net_pnl': net_pnl, 'leverage': state['leverage'],
        'units': state['units'], 'margin_used': state['margin_used'],
        'at_be': state.get('at_be', False),
        'used_trail': state.get('trail_stop', 0) > state['stop_price'],
    }, log_path)

    print(f"\n✓ EXIT — {exit_reason} @ ${exit_price:.4f}  net=${net_pnl:+.2f}")
    if args.email:
        subj, body = format_exit_alert(state, exit_price, exit_reason, gross_pct, net_pnl, bar_time)
        send_alert(subj, body)
    clear_state(state_path)


def loop_mode(args):
    sleep_seconds = max(60, int(args.check_minutes * 60))
    print(f"\n{'#'*70}")
    print(f"# SOL Swing Perp — Loop Mode")
    print(f"# Leverage: {args.leverage}x  Position: {args.position_size:.0%}  Initial: ${args.initial_capital:,.0f}")
    print(f"# Check every {args.check_minutes} min  |  Email: {'ON' if args.email else 'OFF'}")
    print(f"# Log: {args.log_file or LOG_FILE}")
    print(f"# State: {args.state_file or STATE_FILE}")
    print(f"# Ctrl+C to stop")
    print(f"{'#'*70}\n")

    iteration = 0
    while True:
        iteration += 1
        try:
            print(f"\n--- Check #{iteration} at {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} ---")
            live_mode(args)
        except KeyboardInterrupt:
            print("\n\nStopped. State + log preserved.")
            return
        except Exception as e:
            print(f"\nError: {e}\n(continuing)")
        next_run = datetime.now(timezone.utc) + timedelta(seconds=sleep_seconds)
        print(f"\nNext check at {next_run:%H:%M UTC}...")
        try:
            sleep(sleep_seconds)
        except KeyboardInterrupt:
            print("\n\nStopped. State + log preserved.")
            return


# ============================================================================
# CLI
# ============================================================================
def main():
    global REWARD_RATIO, ATR_BUFFER, MAX_HOLD_BARS, TRAIL_ACTIVATION_R, TRAIL_ATR_MULT

    p = argparse.ArgumentParser(description="v10 swing strategy on SOLUSDT perp")
    p.add_argument("--years", type=float, default=BACKTEST_YEARS)
    p.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    p.add_argument("--leverage", type=float, default=LEVERAGE)
    p.add_argument("--position-size", type=float, default=POSITION_SIZE)
    p.add_argument("--fee-pct", type=float, default=FEE_PCT_PER_SIDE,
                   help="Fee per side (0.0004 = 0.04%% Binance perp taker)")
    p.add_argument("--max-risk-pct", type=float, default=MAX_RISK_PCT)
    p.add_argument("--no-funding", action="store_true", help="Disable funding in backtest")
    p.add_argument("--reward-ratio", type=float, default=REWARD_RATIO)
    p.add_argument("--atr-buffer", type=float, default=ATR_BUFFER)
    p.add_argument("--max-hold-bars", type=int, default=MAX_HOLD_BARS)
    p.add_argument("--trail-activation-r", type=float, default=TRAIL_ACTIVATION_R)
    p.add_argument("--trail-atr-mult", type=float, default=TRAIL_ATR_MULT)
    p.add_argument("--backtest", action="store_true", help="Run backtest instead of live bot")
    p.add_argument("--days", type=int, default=None, help="Backtest length in days (overrides --years)")
    p.add_argument("--check-minutes", type=int, default=CHECK_MINUTES, help="Live bot check interval (default 60)")
    p.add_argument("--email", action="store_true", help="Email alerts (needs EMAIL_* in .env)")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--log-file", default=None)
    p.add_argument("--state-file", default=None)
    args = p.parse_args()

    REWARD_RATIO = args.reward_ratio
    ATR_BUFFER = args.atr_buffer
    MAX_HOLD_BARS = args.max_hold_bars
    TRAIL_ACTIVATION_R = args.trail_activation_r
    TRAIL_ATR_MULT = args.trail_atr_mult

    # Default: run the bot continuously. With --backtest: run backtest.
    if not args.backtest:
        loop_mode(args)
        return

    end = datetime.now(timezone.utc)
    years = args.days / 365 if args.days else args.years
    start = end - timedelta(days=int(years * 365))
    print(f"Config:")
    print(f"  Symbol:         {SYMBOL}")
    print(f"  Years:          {years:.2f}")
    print(f"  Initial cap:    ${args.initial_capital:,.2f}")
    print(f"  Leverage:       {args.leverage}x")
    print(f"  Position size:  {args.position_size:.0%} of capital as margin")
    print(f"  Fee per side:   {args.fee_pct*100:.4f}%")
    print(f"  Max risk:       {args.max_risk_pct*100:.1f}% stop distance")
    print(f"  Funding cost:   {'ON' if not args.no_funding else 'OFF'}")
    print(f"  Reward ratio:   {args.reward_ratio}R")
    print(f"  Max hold:       {args.max_hold_bars} bars\n")

    fetch_start = start - timedelta(days=200)
    print(f"Fetching SOLUSDT data: {fetch_start.date()} to {end.date()}")
    df_4h    = fetch_klines(SYMBOL, INTERVAL_4H, fetch_start, end, use_cache=not args.no_cache)
    df_daily = fetch_klines(SYMBOL, INTERVAL_1D, fetch_start, end, use_cache=not args.no_cache)
    funding_df = fetch_funding(SYMBOL, fetch_start, end, use_cache=not args.no_cache) if not args.no_funding else None
    print(f"  4H bars: {len(df_4h)}  daily: {len(df_daily)}  funding events: {len(funding_df) if funding_df is not None else 0}")

    if df_4h.empty:
        print("No data, aborting")
        return

    df = compute_indicators(df_4h, df_daily)
    df = df.loc[df.index >= pd.Timestamp(start)]
    print(f"  After warmup + trim: {len(df)} bars")

    if len(df) < 50:
        print("Not enough bars")
        return

    trades_df, _, summary = backtest_perp(
        df, funding_df,
        initial_capital=args.initial_capital, leverage=args.leverage,
        position_size=args.position_size, fee_pct=args.fee_pct,
        max_risk_pct=args.max_risk_pct, use_funding=not args.no_funding,
        verbose=True,
    )

    label = f"{SYMBOL} v10-perp, {args.leverage}x leverage, {years:.1f}y"
    summarize_trades(trades_df, summary, label=label)
    out_path = args.out or f"backtest_{SYMBOL}_lev{args.leverage}x.jsonl"
    save_trades(trades_df, out_path)


if __name__ == "__main__":
    main()