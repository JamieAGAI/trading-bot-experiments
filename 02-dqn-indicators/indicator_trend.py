import pandas as pd
import numpy as np
import requests
import time
import os

# Set the interval here: '1h' for 1 hour, '4h' for 4 hours
interval = '4h'  # Change to '4h' as needed

# Function to fetch and cache data from Binance API
def fetch_klines(symbol, interval, days):
    limit = 1000
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 86400 * 1000)
    data = []
    while start_time < end_time:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_time}&limit={limit}"
        resp = requests.get(url).json()
        if not resp:
            break
        data.extend(resp)
        start_time = resp[-1][0] + 1
        time.sleep(0.1)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'baseVol', 'close_time', 'usdtVol', 'trades', 'ignore1', 'ignore2', 'ignore3'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'baseVol', 'usdtVol']].astype(float)
    return df

# Load or fetch data
csv_file = f'sol_{interval}_data.csv'
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, parse_dates=['timestamp'], index_col='timestamp')
else:
    df = fetch_klines('SOLUSDT', interval, 1000)
    df.to_csv(csv_file)

# Set volume column (baseVol for base asset volume)
df['volume'] = df['baseVol']

# Function to calculate MA
def calculate_ma(series, length, ma_type):
    if ma_type == "SMA":
        return series.rolling(length).mean()
    elif ma_type == "EMA":
        return series.ewm(span=length, adjust=False).mean()
    elif ma_type == "HMA":
        def wma(s, period):
            weights = np.arange(1, period + 1)
            return s.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        half_len = int(length / 2)
        sqrt_len = int(np.sqrt(length))
        wma_half = wma(series, half_len)
        wma_full = wma(series, length)
        hma = wma(2 * wma_half - wma_full, sqrt_len)
        return hma
    return pd.Series(np.nan, index=series.index)

# Inputs (defaults from Pine Script)
fast_len = 10
slow_len = 22
ma_type = "SMA"

# Calculate MAs
df['fast_ma'] = calculate_ma(df['close'], fast_len, ma_type)
df['slow_ma'] = calculate_ma(df['close'], slow_len, ma_type)

# Signals
df['buy_signal'] = (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)) & (df['fast_ma'] > df['slow_ma'])
df['sell_signal'] = (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1)) & (df['fast_ma'] < df['slow_ma'])

# Simple backtest simulation (long only, with pyramiding; adapted from example)
initial_capital = 60000
capital = initial_capital
positions = []  # List of {'entry_price': float, 'qty': float}
trade_size = 30000
max_positions = 2
trades = []
profits = []
equity_curve = [initial_capital]

for i in range(1, len(df)):
    current_equity = capital + sum(p['qty'] * df['close'].iloc[i] for p in positions)
    equity_curve.append(current_equity)
    
    if df['buy_signal'].iloc[i] and capital >= trade_size and len(positions) < max_positions:
        entry_price = df['close'].iloc[i]
        qty = trade_size / entry_price
        positions.append({'entry_price': entry_price, 'qty': qty})
        capital -= trade_size
        trades.append({'date': df.index[i], 'action': 'buy', 'price': entry_price, 'qty': qty})
    
    elif df['sell_signal'].iloc[i] and len(positions) > 0:
        exit_price = df['close'].iloc[i]
        for pos in positions:
            trade_profit = pos['qty'] * (exit_price - pos['entry_price'])
            profits.append(trade_profit)
            trades.append({'date': df.index[i], 'action': 'sell', 'price': exit_price, 'qty': pos['qty'], 'profit': trade_profit})
        capital += sum(p['qty'] for p in positions) * exit_price
        positions = []

# Handle open positions at end
final_capital = capital + sum(p['qty'] * df['close'].iloc[-1] for p in positions)
if len(positions) > 0:
    exit_price = df['close'].iloc[-1]
    for pos in positions:
        final_profit = pos['qty'] * (exit_price - pos['entry_price'])
        trades.append({'date': df.index[-1], 'action': 'sell (final)', 'price': exit_price, 'qty': pos['qty'], 'profit': final_profit})
        profits.append(final_profit)

# Calculate metrics
total_trades = len(profits)
if total_trades > 0:
    win_rate = (sum(1 for p in profits if p > 0) / total_trades) * 100
    avg_profit = np.mean(profits) if profits else 0
    avg_win = np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0
    avg_loss = np.mean([p for p in profits if p < 0]) if any(p < 0 for p in profits) else 0
    equity_curve = np.array(equity_curve)
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peaks) / peaks
    max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0
else:
    win_rate = avg_profit = avg_win = avg_loss = max_drawdown = 0

# Print trades first
print("\nTrades:")
for trade in trades:
    print(trade)

# Then print the summary totals at the bottom
print(f"\nInitial Capital: ${initial_capital:.2f}")
print(f"Final Capital: ${final_capital:.2f}")
print(f"Profit/Loss: ${final_capital - initial_capital:.2f}")
print(f"Total Trades: {total_trades}")
print(f"Win Rate: {win_rate:.2f}%")
print(f"Average Profit per Trade: ${avg_profit:.2f}")
print(f"Average Win: ${avg_win:.2f}")
print(f"Average Loss: ${avg_loss:.2f}")
print(f"Max Drawdown: {max_drawdown * 100:.2f}%")

# Optional: Save signals to CSV
df.to_csv(f'sol_trend_strategy_signals_{interval}.csv')