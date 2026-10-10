import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
from binance.client import Client as BinanceClient
import ta
import os
import time

# Constants for backtest
TRADE_FEE = 0.003
BUY_PERCENTAGE = 0.5
SELL_PERCENTAGE = 0.5
MIN_TRADE_AMOUNT = 10.0

# Fetch historical data
def fetch_historical_data(days=365, force_fetch=False):
    data_cache = f'sol_usdt_{days}_days_15min.csv'
    if not force_fetch and os.path.exists(data_cache):
        print(f"Loading cached data for {days} days...")
        try:
            return pd.read_csv(data_cache, index_col='timestamp', parse_dates=True)
        except Exception as e:
            print(f"Cache load failed: {e}. Fetching new data...")
    
    print(f"Fetching new data for {days} days...")
    client = BinanceClient('', '')
    interval = BinanceClient.KLINE_INTERVAL_15MINUTE
    start_time = int((time.time() - days * 86400) * 1000)
    klines = []
    while True:
        new_klines = client.get_historical_klines('SOLUSDT', interval, start_str=start_time, limit=1000)
        if not new_klines:
            break
        klines += new_klines
        start_time = new_klines[-1][0] + 1
    
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df.to_csv(data_cache)
    return df

# Backtest function with indicators and logic
def backtest(df, params):
    # Add indicators if enabled
    if params['use_sma']:
        df['sma_short'] = df['close'].rolling(params['sma_short']).mean()
        df['sma_long'] = df['close'].rolling(params['sma_long']).mean()
    if params['use_ema']:
        df['ema_short'] = df['close'].ewm(span=params['ema_short']).mean()
        df['ema_long'] = df['close'].ewm(span=params['ema_long']).mean()
    if params['use_rsi']:
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], params['rsi_len']).rsi()
    if params['use_macd']:
        macd = ta.trend.MACD(df['close'], params['macd_fast'], params['macd_slow'], params['macd_signal'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
    if params['use_bb']:
        bb = ta.volatility.BollingerBands(df['close'], params['bb_len'], params['bb_std'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
    if params['use_stoch']:
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], params['stoch_k'], params['stoch_d'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
    if params['use_adx']:
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], params['adx_len']).adx()
    if params['use_obv']:
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    
    # Generate buy/sell conditions as Series
    buy_conditions = pd.Series(0, index=df.index)
    sell_conditions = pd.Series(0, index=df.index)
    
    if params['use_sma']:
        buy_conditions += (df['sma_short'] > df['sma_long']).astype(int)
        sell_conditions += (df['sma_short'] < df['sma_long']).astype(int)
    if params['use_ema']:
        buy_conditions += (df['ema_short'] > df['ema_long']).astype(int)
        sell_conditions += (df['ema_short'] < df['ema_long']).astype(int)
    if params['use_rsi']:
        buy_conditions += (df['rsi'] < params['rsi_buy']).astype(int)
        sell_conditions += (df['rsi'] > params['rsi_sell']).astype(int)
    if params['use_macd']:
        buy_conditions += (df['macd'] > df['macd_signal']).astype(int)
        sell_conditions += (df['macd'] < df['macd_signal']).astype(int)
    if params['use_bb']:
        buy_conditions += (df['close'] < df['bb_lower']).astype(int)
        sell_conditions += (df['close'] > df['bb_upper']).astype(int)
    if params['use_stoch']:
        buy_conditions += (df['stoch_k'] < params['stoch_buy']).astype(int)
        sell_conditions += (df['stoch_k'] > params['stoch_sell']).astype(int)
    if params['use_adx']:
        buy_conditions += (df['adx'] > params['adx_thresh']).astype(int)
        sell_conditions += (df['adx'] < params['adx_thresh']).astype(int)
    if params['use_obv']:
        buy_conditions += (df['obv'] > df['obv'].shift()).astype(int)
        sell_conditions += (df['obv'] < df['obv'].shift()).astype(int)
    
    # Buy/sell if enough conditions true
    df['buy_signal'] = (buy_conditions >= params['min_signals']) & (buy_conditions.shift() < params['min_signals'])  # New cross
    df['sell_signal'] = (sell_conditions >= params['min_signals']) & (sell_conditions.shift() < params['min_signals'])  # New cross
    
    print(f"Buy signals: {df['buy_signal'].sum()}")
    print(f"Sell signals: {df['sell_signal'].sum()}")

    # Backtest logic
    balance = 60000
    position = 0
    entry_price = 0
    trades = 0
    wins = 0
    net_worth_history = [balance]

    for i, row in df.iterrows():
        if position > 0:
            unrealized = (row['close'] - entry_price) / entry_price
            if unrealized <= params['sl_pct'] or row['sell_signal']:
                sell_value = position * row['close'] * (1 - TRADE_FEE)
                profit = sell_value - position * entry_price
                balance += sell_value
                position = 0
                trades += 1
                if profit > 0:
                    wins += 1
        elif row['buy_signal']:
            buy_value = balance * BUY_PERCENTAGE
            if buy_value >= MIN_TRADE_AMOUNT:
                position = buy_value / row['close']
                entry_price = row['close']
                balance -= buy_value + buy_value * TRADE_FEE
        net_worth = balance + position * row['close']
        net_worth_history.append(net_worth)

    profit = (net_worth_history[-1] - 60000) / 60000 * 100
    dd = max([0] + [1 - nw / max(net_worth_history[:i+1]) for i, nw in enumerate(net_worth_history[1:])]) * 100
    win_rate = (wins / trades * 100) if trades > 0 else 0
    score = profit - dd * 2  # Maximize profit, penalize DD

    # Print for visibility
    print(f"Profit: {profit:.2f}%")
    print(f"Max DD: {dd:.2f}%")
    print(f"Trades: {trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Score: {score:.2f}")

    return score

# Optuna objective
def objective(trial):
    params = {
        'use_sma': trial.suggest_categorical('use_sma', [True, False]),
        'sma_short': trial.suggest_int('sma_short', 18, 38),
        'sma_long': trial.suggest_int('sma_long', 74, 140),
        'use_ema': trial.suggest_categorical('use_ema', [True, False]),
        'ema_short': trial.suggest_int('ema_short', 7, 34),
        'ema_long': trial.suggest_int('ema_long', 67, 181),
        'use_rsi': trial.suggest_categorical('use_rsi', [True, False]),
        'rsi_len': trial.suggest_int('rsi_len', 5, 19),
        'rsi_buy': trial.suggest_int('rsi_buy', 21, 50),
        'rsi_sell': trial.suggest_int('rsi_sell', 59, 75),
        'use_macd': trial.suggest_categorical('use_macd', [True, False]),
        'macd_fast': trial.suggest_int('macd_fast', 9, 15),
        'macd_slow': trial.suggest_int('macd_slow', 19, 25),
        'macd_signal': trial.suggest_int('macd_signal', 13, 17),
        'use_bb': trial.suggest_categorical('use_bb', [True, False]),
        'bb_len': trial.suggest_int('bb_len', 10, 41),
        'bb_std': trial.suggest_float('bb_std', 1.7, 2.8),
        'use_stoch': trial.suggest_categorical('use_stoch', [True, False]),
        'stoch_k': trial.suggest_int('stoch_k', 5, 13),
        'stoch_d': trial.suggest_int('stoch_d', 3, 6),
        'stoch_buy': trial.suggest_int('stoch_buy', 20, 49),
        'stoch_sell': trial.suggest_int('stoch_sell', 53, 76),
        'use_adx': trial.suggest_categorical('use_adx', [True, False]),
        'adx_len': trial.suggest_int('adx_len', 11, 16),
        'adx_thresh': trial.suggest_int('adx_thresh', 23, 40),
        'use_obv': trial.suggest_categorical('use_obv', [True, False]),
        'min_signals': trial.suggest_int('min_signals', 1, 2),  # Lowered for signals
        'sl_pct': trial.suggest_float('sl_pct', -0.05, -0.03),  # Tighter SL
    }
    df = fetch_historical_data(days=365)
    score = backtest(df, params)
    return score

# Run Optuna
sampler = TPESampler(seed=42)
pruner = MedianPruner()
study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=100)  # Or 20 for quick

print("Best params:", study.best_params)
print("Best score:", study.best_value)