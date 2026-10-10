# MEAN REVERSION v3 — Tuned for Balance (More Signals than v2, Quality like IADSS)
# Standalone for signals/backtest; for IADSS-like confluence, combine with trend checks manually.

import pandas as pd
import numpy as np
import time
import requests

# ========================= DATA =========================
def fetch_binance_data(symbol, interval, days_back=365):  # Your backtest period
    end_time = int(time.time() * 1000)
    start_time = end_time - days_back * 86400 * 1000
    data = []
    cur = start_time
    while cur < end_time:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={cur}&limit=1000"
        resp = requests.get(url).json()
        if not resp or isinstance(resp, dict): break
        data.extend(resp)
        cur = resp[-1][0] + 1
        time.sleep(0.08)
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume',
                                     'close_time','quote_volume','trades','taker_base',
                                     'taker_quote','ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')[['open','high','low','close','volume','quote_volume']].astype(float)
    df.rename(columns={'volume':'baseVol','quote_volume':'usdtVol'}, inplace=True)
    return df

# ========================= INDICATORS =========================
def rsi(s, period=14):
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, min_periods=period).mean()
    roll_down = down.ewm(alpha=1/period, min_periods=period).mean()
    rs = roll_up / roll_down
    return 100 - 100/(1 + rs)

def atr(df, period=14):
    tr = pd.concat([df.high-df.low,
                    (df.high-df.close.shift()).abs(),
                    (df.low-df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ========================= MAIN =========================
symbol   = 'SOLUSDT'
interval = '4h'
days_back = 365
df = fetch_binance_data(symbol, interval, days_back=days_back)
df['volume'] = df['baseVol']

# PARAMETERS (Balanced for more signals with filters)
bb_period = 20
bb_std = 2.0       # Standard width
dev_thresh = 1.5   # Loosened for more deviations
rsi_buy = 40       # Less strict oversold
rsi_sell = 60      # Less strict overbought
atr_mult = 2.0     # For stops
vol_mult = 1.0     # Loosened volume
trail_mult = 4.0   # Hold longer
buy_size_pct = 1.0 # Full positions

# INDICATORS
df['ma200'] = df['close'].rolling(200).mean()
df['rsi'] = rsi(df['close'])
df['atr'] = atr(df)
df['vol_ma'] = df['volume'].rolling(20).mean()

# BOLLINGER BANDS
df['bb_mid'] = df['close'].rolling(bb_period).mean()
df['bb_std'] = df['close'].rolling(bb_period).std()
df['bb_lower'] = df['bb_mid'] - bb_std * df['bb_std']
df['bb_upper'] = df['bb_mid'] + bb_std * df['bb_std']

# DEVIATION (IADSS-style)
df['dev'] = (df['close'] - df['bb_mid']) / df['bb_std']

# SIGNALS (Loosened: No candle confirmation)
df['uptrend'] = df['close'] > df['ma200']  # Simplified trend filter

df['oversold_dev'] = df['dev'] < -dev_thresh
df['overbought_dev'] = df['dev'] > dev_thresh

df['buy'] = (
    df['oversold_dev'].shift(1) & 
    (df['rsi'].shift(1) < rsi_buy) & 
    df['uptrend'].shift(1) & 
    (df['volume'].shift(1) > vol_mult * df['vol_ma'].shift(1))
)

df['sell'] = (
    df['overbought_dev'].shift(1) & 
    (df['rsi'].shift(1) > rsi_sell)
)

print("\nDebug: Condition counts (full dataset)")
print(f"oversold_dev: {df['oversold_dev'].sum()}")
print(f"rsi < {rsi_buy}: {(df['rsi'] < rsi_buy).sum()}")
print(f"uptrend: {df['uptrend'].sum()}")
print(f"volume > {vol_mult}*vol_ma: {(df['volume'] > vol_mult * df['vol_ma']).sum()}")
print(f"overbought_dev: {df['overbought_dev'].sum()}")
print(f"rsi > {rsi_sell}: {(df['rsi'] > rsi_sell).sum()}")

print("\nRecent bars debug (check 'buy' and 'sell' columns for signals):")
print(df[['close', 'rsi', 'dev', 'bb_lower', 'bb_upper', 'uptrend', 'buy', 'sell']][-100:])

# BACKTEST
capital = 60_000.0
cash = capital
position = 0.0
equity = []
buy_count = 0
sell_count = 0
stop_count = 0
trail_high = 0.0

for i in range(300, len(df)):
    price = df['close'].iloc[i]
    equity.append(cash + position * price)

    if df['buy'].iloc[i] and cash > 2000:
        alloc_cash = cash * buy_size_pct
        coins = alloc_cash / price
        cash -= coins * price
        position += coins
        trail_high = price
        buy_count += 1

    if position > 0:
        trail_high = max(trail_high, price)
        if price < trail_high - df['atr'].iloc[i] * trail_mult:
            cash += position * price
            position = 0
            stop_count += 1

    if df['sell'].iloc[i] and position > 0:
        cash += position * price
        position = 0
        sell_count += 1

final_eq = cash + position * df['close'].iloc[-1]

# RESULTS
print("\nMEAN REVERSION v3 — 4h — TUNED FOR BALANCE")
print(f"Final equity   : ${final_eq:,.2f}")
print(f"Total return   : {(final_eq/capital-1)*100:+.2f}%")
print(f"Max drawdown   : {((pd.Series(equity).cummax() - equity)/pd.Series(equity).cummax()).max()*100:.2f}%")
print(f"Buy signals    : {int(df['buy'].sum())}")
print(f"Sell signals   : {int(df['sell'].sum())}")
print(f"Actual buys  : {buy_count}")
print(f"Actual sells : {sell_count}")
print(f"Stop losses  : {stop_count}")