# Stochastic Trend Dip Buyer — For SOLUSDT 4h
# New Strategy: Use Stochastic Oscillator for oversold/overbought in uptrends, with EMA filter and volume confirmation
# Focus on swing trades: Enter on stochastic cross-ups in dips, exit on cross-downs or trailing stop
# Aimed at capturing pullbacks in trends for better entries
import pandas as pd
import numpy as np
import time
import requests

# ========================= DATA =========================
def fetch_binance_data(symbol, interval, days_back=1000):  # Adjust days_back for period
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

def stochastic(df, k=14, d=3, smooth=3):
    low_min = df['low'].rolling(k).min()
    high_max = df['high'].rolling(k).max()
    df['%K'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['%K_smooth'] = df['%K'].rolling(smooth).mean()
    df['%D'] = df['%K_smooth'].rolling(d).mean()

# ========================= MAIN =========================
symbol   = 'SOLUSDT'
interval = '4h'
days_back = 365  # Adjust for backtest period
df = fetch_binance_data(symbol, interval, days_back=days_back)
df['volume'] = df['baseVol']

# PARAMETERS (tuned for SOL 4h swing trades)
stoch_k = 14
stoch_d = 3
stoch_smooth = 3
stoch_buy = 30  # Oversold threshold (loosened from 20 for more signals)
stoch_sell = 70  # Overbought threshold (tightened from 80 for quicker exits)
rsi_period = 14
rsi_filter = 45  # Additional RSI filter for buys
ma_period = 200
ema_short = 20
vol_ma_period = 20
atr_period = 14
atr_mult_stop = 2.0  # For trailing stop
buy_size_pct = 1.0  # All-in; adjust for risk

# INDICATORS
df['ma_long'] = df['close'].rolling(ma_period).mean()
df['ema_short'] = df['close'].ewm(span=ema_short, adjust=False).mean()
df['rsi'] = rsi(df['close'], rsi_period)
df['atr'] = atr(df, atr_period)
df['vol_ma'] = df['volume'].rolling(vol_ma_period).mean()
stochastic(df, stoch_k, stoch_d, stoch_smooth)

# SIGNALS
df['uptrend'] = df['close'] > df['ma_long']
df['above_ema'] = df['close'] > df['ema_short']  # Avoid buying deep dips

df['stoch_cross_up'] = (df['%K_smooth'].shift(1) < df['%D'].shift(1)) & (df['%K_smooth'] > df['%D']) & (df['%K_smooth'] < stoch_buy)
df['stoch_cross_down'] = (df['%K_smooth'].shift(1) > df['%D'].shift(1)) & (df['%K_smooth'] < df['%D']) & (df['%K_smooth'] > stoch_sell)

df['buy'] = (
    df['stoch_cross_up'].shift(1) &
    (df['rsi'].shift(1) < rsi_filter) &
    df['uptrend'].shift(1) &
    df['above_ema'].shift(1) &
    (df['volume'].shift(1) > df['vol_ma'].shift(1))
)

df['sell'] = (
    df['stoch_cross_down'].shift(1) |
    (df['rsi'].shift(1) > 70)  # Additional overbought exit
)

print("\nDebug: Condition counts (full dataset)")
print(f"stoch_cross_up: {df['stoch_cross_up'].sum()}")
print(f"rsi < {rsi_filter}: {(df['rsi'] < rsi_filter).sum()}")
print(f"uptrend: {df['uptrend'].sum()}")
print(f"above_ema: {df['above_ema'].sum()}")
print(f"volume > vol_ma: {(df['volume'] > df['vol_ma']).sum()}")
print(f"stoch_cross_down: {df['stoch_cross_down'].sum()}")
print(f"rsi > 70: {(df['rsi'] > 70).sum()}")

print("\nRecent bars debug:")
print(df[['close', 'rsi', '%K_smooth', '%D', 'uptrend', 'buy', 'sell']][-100:])

# BACKTEST
capital = 60_000.0
cash = capital
position = 0.0
trailing_stop = 0.0
equity = []
buy_count = 0
sell_count = 0
stop_count = 0

start_idx = max(ma_period, ema_short, vol_ma_period, rsi_period, stoch_k) + 1

for i in range(start_idx, len(df)):
    price = df['close'].iloc[i]
    equity.append(cash + position * price)

    if df['buy'].iloc[i] and cash > 2000:
        alloc_cash = cash * buy_size_pct
        coins = alloc_cash / price
        cash -= coins * price
        position += coins
        trailing_stop = price - atr_mult_stop * df['atr'].iloc[i]
        buy_count += 1

    if position > 0:
        new_stop = price - atr_mult_stop * df['atr'].iloc[i]
        trailing_stop = max(trailing_stop, new_stop)
        
        if price < trailing_stop:
            cash += position * price
            position = 0
            stop_count += 1
            continue

        if df['sell'].iloc[i]:
            cash += position * price
            position = 0
            sell_count += 1

final_eq = cash + position * df['close'].iloc[-1]

# RESULTS
print("\nStochastic Trend Dip Buyer — 4h — BETTER SWING STRATEGY")
print(f"Final equity   : ${final_eq:,.2f}")
print(f"Total return   : {(final_eq/capital-1)*100:+.2f}%")
print(f"Max drawdown   : {((pd.Series(equity).cummax() - equity)/pd.Series(equity).cummax()).max()*100:.2f}%")
print(f"Buy signals    : {int(df['buy'].sum())}")
print(f"Sell signals   : {int(df['sell'].sum())}")
print(f"Actual buys    : {buy_count}")
print(f"Actual sells   : {sell_count}")
print(f"Stop losses    : {stop_count}")