# Copyright (c) 2026 Laura Staples. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, distribution, or derivative works are prohibited.
# No license is granted for training machine learning models on this code.
#
# =============================================================================
#  BACKTEST-ONLY VERSION OF main.py
#  ---------------------------------------------------------------------------
#  This is a stripped-down copy of main.py for environments where the
#  Solana and python-binance packages cannot be installed (e.g. ARM64 Windows
#  laptops where Rust tooling or MS Build Tools aren't available).
#
#  What's different from main.py:
#    - No solana / solders / spl-token imports (cannot be installed on ARM64)
#    - No python-binance import; uses plain `requests` to hit Binance's public
#      klines endpoint directly (no API key needed for market data)
#    - No live trading code path; only backtest mode
#    - No websocket, no email alerts, no Jupiter swaps
#
#  What's identical to main.py (must be identical — this is the whole point):
#    - Strategy parameters (EMA 35/15, RSI 53-65, ATR stops, 2.5 RR, etc.)
#    - Indicator calculations (compute_4h, compute_daily, attach_daily)
#    - Signal logic (buy_signal)
#    - Backtest loop (exit cascade, trailing stop, BE, timeout)
#    - Output CSV format (trades_v10.csv)
#
#  Expected backtest results (Apr 17, 2026 reference):
#    4yr (1460d): +1,889% | 103 trades | 54.4% WR
#    3yr (1095d): +900%   |  81 trades | 54.3% WR
#    2yr (730d):  +212%   |  53 trades | 52.8% WR
#    1yr (365d):  +30%    |  29 trades | 48.3% WR
#
#  If your numbers differ by more than a trade or two, something is wrong
#  with the environment (Python version, pandas_ta version, etc.) — stop
#  and investigate before trusting anything on live.
#
#  Run with:   python main_backtest.py --backtest --days 1460
# =============================================================================

import pandas as pd
import pandas_ta as ta
import time
import argparse
import os
import csv
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
#  STRATEGY PARAMS   (must match main.py exactly)
# =============================================================================

SYMBOL             = 'SOLUSDT'
INTERVAL_4H        = '4h'
INTERVAL_1D        = '1d'
EMA_TREND          = 35
EMA_VALUE          = 15
EMA_DAILY          = 20
RSI_LEN            = 14
ATR_LEN            = 14
REWARD_RATIO       = 2.5
ATR_BUFFER         = 0.45
MAX_HOLD_BARS      = 50
RSI_MIN            = 53
RSI_MAX            = 65
FEE_PCT            = 0.003
VOLUME_MULTIPLIER  = 1.1

POSITION_SIZE      = 1.0
MAX_RISK_PCT       = 0.10

CACHE_4H   = 'sol_4h_cache.csv'
CACHE_1D   = 'sol_1d_cache.csv'
TRADES_CSV = 'trades_v10.csv'

# =============================================================================
#  BINANCE PUBLIC API (no authentication needed for klines)
# =============================================================================

BINANCE_KLINES_URL = 'https://api.binance.com/api/v3/klines'

def fetch_klines_requests(symbol, interval, start_ms, end_ms, limit=1000):
    """
    Fetch klines from Binance's public REST API using plain requests.
    Paginates automatically to cover the full start-to-end range.
    Returns a list of klines in Binance's native format.
    """
    all_klines = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            'symbol':    symbol,
            'interval':  interval,
            'startTime': current_start,
            'endTime':   end_ms,
            'limit':     limit,
        }
        for attempt in range(3):
            try:
                r = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as e:
                logger.warning(f"Klines fetch attempt {attempt+1}/3 failed: {e}")
                if attempt == 2:
                    raise
                time.sleep(5)

        if not batch:
            break

        all_klines.extend(batch)

        # Advance to just after the last kline we got
        last_close_time = batch[-1][6]  # close_time in ms
        current_start = last_close_time + 1

        # Binance can be rate-limited; small sleep is polite
        time.sleep(0.2)

        # If we got fewer than `limit`, we've reached the end
        if len(batch) < limit:
            break

    return all_klines


# ── Fetch helpers ────────────────────────────────────────────────────────────

def _time_range(days):
    end   = int(time.time() * 1000)
    start = end - (days * 86400 * 1000)
    return end, start

def _klines_to_df(klines):
    df = pd.DataFrame(
        klines,
        columns=['timestamp','open','high','low','close','volume',
                 'close_time','quote_av','trades','tb_base_av','tb_quote_av','ignore']
    )
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    df = df.set_index(pd.to_datetime(df['timestamp'], unit='ms', utc=True))
    df.index.name = 'timestamp'
    return df[['open','high','low','close','volume']]

def _cache_is_stale(cache_path, max_age_hours):
    if not os.path.exists(cache_path):
        return True
    age_seconds = time.time() - os.path.getmtime(cache_path)
    return age_seconds > (max_age_hours * 3600)

def fetch_data(days=730):
    for attempt in range(3):
        try:
            end_ms, start_ms = _time_range(days)

            if not _cache_is_stale(CACHE_4H, max_age_hours=4):
                logger.info("Loading 4h from cache (fresh)...")
                df4_local = pd.read_csv(CACHE_4H, index_col='timestamp', parse_dates=True)
                df4_local = df4_local[['open','high','low','close','volume']]
                if df4_local.index.tz is None:
                    df4_local.index = df4_local.index.tz_localize('UTC')
            else:
                logger.info("Fetching fresh 4h data from Binance public API (requests)...")
                klines_4h = fetch_klines_requests(SYMBOL, INTERVAL_4H, start_ms, end_ms)
                df4_local = _klines_to_df(klines_4h)
                df4_local.to_csv(CACHE_4H)
                logger.info(f"4h data: {len(df4_local)} bars")

            logger.info("Fetching fresh daily data from Binance public API (requests)...")
            klines_1d  = fetch_klines_requests(SYMBOL, INTERVAL_1D, start_ms, end_ms)
            df1d_local = _klines_to_df(klines_1d)
            df1d_local.to_csv(CACHE_1D)
            logger.info(f"Daily data: {len(df1d_local)} bars")

            return df4_local, df1d_local

        except Exception as e:
            logger.warning(f"Fetch attempt {attempt+1}/3 failed: {e}")
            if attempt < 2:
                time.sleep(10)
    raise ValueError("Data fetch failed after 3 retries")


# ── Indicators (identical to main.py) ────────────────────────────────────────

def compute_4h(df, dropna=True):
    df = df.copy()
    df['ema_trend']  = ta.ema(df['close'], length=EMA_TREND)
    df['ema_value']  = ta.ema(df['close'], length=EMA_VALUE)
    df['rsi']        = ta.rsi(df['close'], length=RSI_LEN)
    df['atr']        = ta.atr(df['high'], df['low'], df['close'], length=ATR_LEN)
    df['vol_ma']     = df['volume'].rolling(20, min_periods=1).mean()
    df['bullish']    = (df['close'] > df['open']).astype(int)
    return df.dropna() if dropna else df

def compute_daily(df):
    df = df.copy()
    df['ema_daily'] = ta.ema(df['close'], length=EMA_DAILY)
    return df.dropna()

def attach_daily(df4_in, df1d_in):
    df1d_in = df1d_in.copy()
    df4_in  = df4_in.copy()
    if df1d_in.index.tz is None:
        df1d_in.index = df1d_in.index.tz_localize('UTC')
    if df4_in.index.tz is None:
        df4_in.index = df4_in.index.tz_localize('UTC')
    df1d_in['daily_bull'] = (df1d_in['close'] > df1d_in['ema_daily']).astype(int)
    daily_bull = df1d_in['daily_bull'].reindex(df4_in.index, method='ffill')
    df4_out = df4_in.copy()
    df4_out['daily_bull'] = daily_bull
    return df4_out.dropna()


# ── Signal (identical to main.py) ────────────────────────────────────────────

def buy_signal(df, i, debug=False):
    if i < 1:
        return False
    c  = df.iloc[i]
    p1 = df.iloc[i-1]
    recent_low    = df.iloc[max(0, i-8):i+1]['low'].min()
    in_value_zone = recent_low <= c['ema_value'] * 1.02

    conditions = [
        (c['daily_bull'] == 1,             "daily_bull"),
        (c['close'] > c['ema_trend'],      "close > ema_trend"),
        (in_value_zone,                     "in_value_zone"),
        (c['low'] > p1['low'],             "higher low"),
        (c['close'] > p1['high'],          "close > prev high"),
        (c['bullish'] == 1,                 "bullish candle"),
        (RSI_MIN < c['rsi'] < RSI_MAX,     "RSI range"),
        (c['volume'] >= c['vol_ma'] * VOLUME_MULTIPLIER, "volume threshold"),
        (c['close'] > c['ema_value'],      "close > ema_value"),
    ]

    for passed, name in conditions:
        if not passed:
            if debug:
                logger.info(f"  Signal FAILED: {name} | "
                            f"daily_bull={c['daily_bull']}, close={c['close']:.2f}, "
                            f"ema_trend={c['ema_trend']:.2f}, in_value_zone={in_value_zone}, "
                            f"low={c['low']:.2f}, p1_low={p1['low']:.2f}, "
                            f"close_vs_p1_high={c['close'] > p1['high']}, "
                            f"bullish={c['bullish']}, rsi={c['rsi']:.1f}, "
                            f"vol={c['volume']:.0f}, vol_ma*{VOLUME_MULTIPLIER}={c['vol_ma']*VOLUME_MULTIPLIER:.0f}, "
                            f"close_vs_ema_value={c['close'] > c['ema_value']}")
            return False

    if debug:
        logger.info("  Signal: ALL conditions passed!")
    return True


# ── Backtest (identical to main.py) ──────────────────────────────────────────

def backtest(df, initial_capital=10000):
    capital      = initial_capital
    position     = 0
    entry_price  = 0
    stop_price   = 0
    target_price = 0
    be_trigger   = 0
    at_be        = False
    trail_stop   = 0
    entry_bar    = 0
    trades       = []
    wins = losses = 0

    for i in range(8, len(df) - 1):
        if position == 0 and buy_signal(df, i):
            next_open   = df.iloc[i+1]['open']
            entry_price = next_open
            atr         = df.iloc[i]['atr']
            swing_low   = df.iloc[i-2:i+1]['low'].min()
            stop_price  = swing_low - (ATR_BUFFER * atr)
            risk        = entry_price - stop_price
            if risk <= 0 or (risk / entry_price) > MAX_RISK_PCT:
                continue

            target_price = entry_price + (REWARD_RATIO * risk)
            be_trigger   = entry_price + risk
            deploy       = capital * POSITION_SIZE
            shares       = deploy / entry_price
            position     = shares
            entry_bar    = i + 1
            trail_stop   = stop_price

            trades.append({
                'entry_date' : df.iloc[i+1].name,
                'price'      : round(entry_price, 4),
                'stop'       : round(stop_price, 4),
                'target'     : round(target_price, 4),
                'risk'       : round(risk, 4),
                'rr'         : round(REWARD_RATIO, 2),
                'shares'     : round(shares, 6),
                'profit'     : None,
                'capital'    : round(capital, 2),
                'exit_reason': None,
            })

        elif position > 0:
            cur       = df.iloc[i]
            bars_held = i - entry_bar

            if not at_be and cur['high'] >= be_trigger:
                stop_price = entry_price
                at_be      = True

            exit_price  = None
            exit_reason = None

            if cur['close'] > entry_price + (2 * risk):
                recent_high = df.iloc[max(0, i-2):i+1]['high'].max()
                new_trail   = recent_high - (2 * cur['atr'])
                trail_stop  = max(trail_stop, new_trail)

            effective_stop = max(stop_price, trail_stop)
            if cur['low'] <= effective_stop:
                exit_price  = effective_stop
                exit_reason = 'Trailing Stop' if trail_stop > stop_price and cur['low'] <= trail_stop else ('Breakeven Stop' if at_be else 'Stop Loss')
            elif cur['high'] >= target_price:
                exit_price  = target_price
                exit_reason = 'Take Profit'
            elif (cur['close'] < cur['ema_trend'] and df.iloc[i-1]['close'] < df.iloc[i-1]['ema_trend']):
                exit_price  = cur['close']
                exit_reason = 'Trend Exit'
            elif cur['daily_bull'] == 0:
                exit_price  = cur['close']
                exit_reason = 'Daily Trend Exit'
            elif bars_held >= MAX_HOLD_BARS:
                exit_price  = cur['close']
                exit_reason = 'Timeout'

            if exit_price is not None:
                gross_profit = (exit_price - entry_price) * position
                entry_fee    = (entry_price * position) * FEE_PCT
                exit_fee     = (exit_price * position) * FEE_PCT
                net_profit   = gross_profit - entry_fee - exit_fee
                capital     += net_profit
                position     = 0
                if net_profit > 0:
                    wins += 1
                else:
                    losses += 1
                trades[-1].update({
                    'exit_date'  : cur.name,
                    'exit_price' : round(exit_price, 4),
                    'profit'     : round(net_profit, 2),
                    'exit_reason': exit_reason,
                    'bars_held'  : bars_held,
                    'at_be'      : at_be,
                })

    if position > 0:
        exit_price = df.iloc[-1]['close']
        gross_profit = (exit_price - entry_price) * position
        entry_fee    = (entry_price * position) * FEE_PCT
        exit_fee     = (exit_price * position) * FEE_PCT
        net_profit   = gross_profit - entry_fee - exit_fee
        capital     += net_profit
        if net_profit > 0:
            wins += 1
        else:
            losses += 1
        trades[-1].update({
            'exit_date'  : df.iloc[-1].name,
            'exit_price' : round(exit_price, 4),
            'profit'     : round(net_profit, 2),
            'exit_reason': 'End of Data',
            'bars_held'  : len(df) - 1 - entry_bar,
            'at_be'      : at_be,
        })

    closed     = [t for t in trades if t.get('exit_reason')]
    profits    = [t['profit'] for t in closed if t['profit'] is not None]
    total      = len(closed)
    pct        = ((capital - initial_capital) / initial_capital) * 100
    wr         = (wins / total * 100) if total > 0 else 0
    avg_w      = sum(p for p in profits if p > 0) / wins if wins else 0
    avg_l      = sum(p for p in profits if p <= 0) / losses if losses else 0
    exp        = (wr/100 * avg_w) + ((1 - wr/100) * avg_l)

    logger.info("\n" + "═"*47)
    logger.info(f"BACKTEST — {SYMBOL} 4h + Daily (v10)  VOLUME x{VOLUME_MULTIPLIER}")
    logger.info(f"{df.index[0].date()} → {df.index[-1].date()}")
    logger.info("═"*47)
    logger.info(f"Starting Capital : ${initial_capital:,.2f}")
    logger.info(f"Final Capital    : ${capital:,.2f}")
    logger.info(f"Total Return     : {pct:+.2f}%")
    logger.info(f"Total Trades     : {total}")
    logger.info(f"Wins / Losses    : {wins} / {losses}")
    logger.info(f"Win Rate         : {wr:.1f}%")
    logger.info(f"Expectancy/trade : ${exp:,.2f}")
    logger.info("═"*47)

    fieldnames = ['entry_date','price','stop','target','risk','rr','shares','exit_date','exit_price','profit','capital','exit_reason','bars_held','at_be']
    with open(TRADES_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(closed)
    logger.info(f"Trade log → {TRADES_CSV}")
    return capital, trades


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Backtest-only version of main.py for ARM64 laptops'
    )
    parser.add_argument('--backtest', action='store_true',
                        help='Run backtest mode (this is the only mode)')
    parser.add_argument('--days', type=int, default=1095,
                        help='Number of days of historical data to test (default 1095 = 3yr)')
    parser.add_argument('--fresh', action='store_true',
                        help='Delete cache files before running')
    args = parser.parse_args()

    # This script only does backtests — if no flag given, still run backtest
    if args.fresh:
        for f in [CACHE_4H, CACHE_1D]:
            if os.path.exists(f):
                os.remove(f)
        logger.info("Cache cleared.")

    logger.info(f"BACKTEST-ONLY script — {args.days} days of SOLUSDT")
    logger.info("This version has NO live trading capability.")
    logger.info("For live trading, use main.py on the VPS.")

    df4, df1d = fetch_data(days=args.days)
    df1d = compute_daily(df1d)
    df4  = compute_4h(df4)
    df4  = attach_daily(df4, df1d)

    backtest(df4)