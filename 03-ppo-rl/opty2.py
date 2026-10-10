import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
from binance.client import Client as BinanceClient
import time
import os

# Constants
SLIPPAGE = 0.005
TRADE_FEE = 0.003
MIN_TRADE_AMOUNT = 10.0
RISK_PERCENT = 0.01  # Risk 1% per trade
DATA_DAYS = 730
DATA_CACHE = f'sol_usdt_{DATA_DAYS}_days_15min.csv'

# Fetch historical data (unchanged from your script)
def fetch_historical_data(days=DATA_DAYS, force_fetch=False):
    data_cache = f'sol_usdt_{days}_days_15min.csv'
    if not force_fetch and os.path.exists(data_cache):
        print(f"Loading cached data for {days} days...")
        try:
            return pd.read_csv(data_cache, index_col='timestamp', parse_dates=True)
        except Exception as e:
            print(f"Cache load failed: {e}. Fetching new data...")
    
    print(f"Fetching new data for {days} days...")
    client = BinanceClient('', '')
    interval = client.KLINE_INTERVAL_15MINUTE
    start_time = int((time.time() - days * 86400) * 1000)
    klines = client.get_historical_klines('SOLUSDT', interval, start_str=start_time)
    
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df.to_csv(data_cache)
    return df

# Compute indicators (parametric, with toggles) - Removed unused stoch/adx/vol/ma
def compute_indicators(df, rsi_len, macd_fast, macd_slow, macd_signal, bb_period, bb_std, atr_len):
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_len).mean()
    loss = -delta.where(delta < 0, 0).rolling(rsi_len).mean()
    rs = gain / loss.replace([np.inf, -np.inf, np.nan], 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_fast = df['close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=macd_slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(bb_period).mean()
    bb_std_val = df['close'].rolling(bb_period).std()
    df['bb_upper'] = df['bb_mid'] + bb_std_val * bb_std
    df['bb_lower'] = df['bb_mid'] - bb_std_val * bb_std
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = tr.rolling(atr_len).mean()
    
    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    return df

# Backtest function (rule-based signals) - Removed unused params/blocks for fixed False
def backtest(df, params):
    rsi_buy_thresh = params['rsi_buy_thresh']
    rsi_sell_thresh = params['rsi_sell_thresh']
    bb_squeeze_thresh = params['bb_squeeze_thresh']
    sl_mult = params['sl_mult']
    tp_mult = params['tp_mult']
    atr_len = params['atr_len']
    
    periods = [
        params['rsi_len'],
        params['macd_fast'], params['macd_slow'], params['macd_signal'],
        params['bb_period'],
        params['atr_len']
    ]
    min_start = max(periods) * 2  # Buffer for stabilization
    
    position = 0
    balance = 10000  # Start small for testing
    entry_price = 0
    sl_level = 0
    tp_level = 0
    trades = 0
    wins = 0
    losses = 0  # Added losses
    net_worth_history = [balance]

    for i in range(min_start, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Buy signal: RSI + MACD + BB (fixed True)
        buy_signal = (curr['rsi'] < rsi_buy_thresh) and \
                     (curr['macd'] > curr['macd_signal']) and (prev['macd'] <= prev['macd_signal']) and \
                     ((curr['bb_upper'] - curr['bb_lower']) / curr['bb_mid'] < bb_squeeze_thresh) and (curr['close'] > curr['bb_mid'])
        
        # Sell signal similar
        sell_signal = (curr['rsi'] > rsi_sell_thresh) and \
                      (curr['macd'] < curr['macd_signal']) and (prev['macd'] >= prev['macd_signal']) and \
                      (curr['close'] > curr['bb_upper'])
        
        if position == 0 and buy_signal:
            est_sl_dist = curr['atr'] * sl_mult / curr['close']
            buy_value = RISK_PERCENT * balance / est_sl_dist
            buy_value = min(buy_value, balance)
            if buy_value >= MIN_TRADE_AMOUNT:
                position = buy_value / curr['close']
                balance -= buy_value * (1 + TRADE_FEE + SLIPPAGE)
                entry_price = curr['close']
                sl_level = curr['close'] - curr['atr'] * sl_mult
                tp_level = curr['close'] + curr['atr'] * tp_mult
                trades += 1
        
        elif position > 0:
            profit = 0
            if curr['low'] <= sl_level:
                sell_value = position * sl_level
                profit = sell_value - (position * entry_price)
                balance += sell_value * (1 - TRADE_FEE - SLIPPAGE)
                position = 0
            elif curr['high'] >= tp_level:
                sell_value = position * tp_level
                profit = sell_value - (position * entry_price)
                balance += sell_value * (1 - TRADE_FEE - SLIPPAGE)
                position = 0
            elif sell_signal:
                sell_value = position * curr['close']
                profit = sell_value - (position * entry_price)
                balance += sell_value * (1 - TRADE_FEE - SLIPPAGE)
                position = 0
            if profit > 0:
                wins += 1
            elif profit < 0:
                losses += 1
        
        net_worth = balance + position * curr['close']
        net_worth_history.append(net_worth)
    
    profit_pct = (net_worth_history[-1] - 10000) / 10000 * 100 if net_worth_history else 0
    max_dd = min(0, min((nw / max_nw - 1) for nw, max_nw in zip(net_worth_history, np.maximum.accumulate(net_worth_history)))) * 100 if net_worth_history else 0
    win_rate = (wins / trades * 100) if trades > 0 else 0
    trade_penalty = abs(trades - 30) * 0.1  # Adjusted aim for more trades
    score = profit_pct - abs(max_dd) + win_rate - trade_penalty
    if trades < 10:
        score -= 100  # Penalize low activity
    
    # Return metrics as dict for averaging/printing
    return {
        'score': score,
        'profit_pct': profit_pct,
        'max_dd': max_dd,
        'trades': trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate
    }

# Objective for Optuna (walk-forward) - Average metrics and print
def objective(trial):
    rsi_len = trial.suggest_int('rsi_len', 17, 18)
    rsi_buy_thresh = trial.suggest_int('rsi_buy_thresh', 46, 47)
    rsi_sell_thresh = trial.suggest_int('rsi_sell_thresh', 74, 75)
    macd_fast = trial.suggest_int('macd_fast', 9, 10)
    macd_slow = trial.suggest_int('macd_slow', 24, 29)
    bb_period = trial.suggest_int('bb_period', 23, 25)
    bb_std = trial.suggest_float('bb_std', 2.10, 2.18)
    bb_squeeze_thresh = trial.suggest_float('bb_squeeze_thresh', 0.04, 0.05)
    sl_mult = trial.suggest_float('sl_mult', 2.81, 2.92)
    tp_mult = trial.suggest_float('tp_mult', 4.57, 4.79)
    
    params = trial.params
    params['ma_type'] = 'EMA'
    params['use_ma'] = False
    params['use_rsi'] = True
    params['macd_signal'] = 7
    params['use_macd'] = True
    params['use_bb'] = True
    params['use_stoch'] = False
    params['atr_len'] = 17
    params['use_adx'] = False
    params['use_vol'] = False
    
    # Full data
    df_full = fetch_historical_data()
    df_full = compute_indicators(df_full, rsi_len, macd_fast, macd_slow, 7, bb_period, bb_std, 17)
    
    # Walk-forward: 3 folds
    fold_metrics = []
    for cv_i in range(3):
        holdout_days = 100
        train_days = 200
        holdout_rows = holdout_days * 96  # 15min bars/day
        train_rows = train_days * 96
        shift = cv_i * (holdout_rows // 2)
        train_start = len(df_full) - holdout_rows - train_rows - shift
        train_end = train_start + train_rows
        if train_start < 0:
            continue
        train_df = df_full.iloc[train_start:train_end]
        holdout_df = df_full.iloc[train_end:train_end + holdout_rows]
        
        metrics = backtest(holdout_df, params)  # Eval on holdout
        fold_metrics.append(metrics)
    
    if not fold_metrics:
        return -np.inf
    
    # Average metrics across folds
    avg_score = np.mean([m['score'] for m in fold_metrics])
    avg_profit_pct = np.mean([m['profit_pct'] for m in fold_metrics])
    avg_max_dd = np.mean([m['max_dd'] for m in fold_metrics])
    avg_trades = np.mean([m['trades'] for m in fold_metrics])
    avg_wins = np.mean([m['wins'] for m in fold_metrics])
    avg_losses = np.mean([m['losses'] for m in fold_metrics])
    avg_win_rate = np.mean([m['win_rate'] for m in fold_metrics])
    
    # Print details for this trial
    print(f"Trial {trial.number}:")
    print(f"  Avg Score: {avg_score:.2f}")
    print(f"  Avg Profit %: {avg_profit_pct:.2f}")
    print(f"  Avg Max DD: {avg_max_dd:.2f}")
    print(f"  Avg Trades: {avg_trades:.2f}")
    print(f"  Avg Wins: {avg_wins:.2f}")
    print(f"  Avg Losses: {avg_losses:.2f}")
    print(f"  Avg Win Rate: {avg_win_rate:.2f}%")
    print(f"  Params: {params}")
    
    return avg_score

# Run
storage = "sqlite:///strategy_opt_refined2.db"  # New DB for further refined
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42), pruner=MedianPruner(), storage=storage, study_name="strategy_opt_refined2", load_if_exists=True)
study.optimize(objective, n_trials=25)

print("Best params:", study.best_params)
print("Best score:", study.best_value)