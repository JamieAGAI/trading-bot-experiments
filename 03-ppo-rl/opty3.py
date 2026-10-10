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

# Fetch historical data (unchanged)
def fetch_historical_data(days=DATA_DAYS, force_fetch=False):
    data_cache = f'sol_usdt_{days}_days_15min.csv'
    if not force_fetch and os.path.exists(data_cache):
        print(f"Loading cached data for {days} days...")
        try:
            return pd.read_csv(data_cache, index_col='timestamp', parse_dates=True)
        except Exception as e:
            print(f"Cache load failed: {e}. Fetching new data...")
    
    print(f"Fetching new data for {days} days...")
    client = BinanceClient('', '')  # Fill in your API keys here
    interval = client.KLINE_INTERVAL_15MINUTE
    start_time = int((time.time() - days * 86400) * 1000)
    klines = client.get_historical_klines('SOLUSDT', interval, start_str=start_time)
    
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df.to_csv(data_cache)
    return df

# Compute indicators: Supertrend (corrected flip logic)
def compute_indicators(df, atr_len, mult):
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = tr.rolling(atr_len).mean().values
    
    hl2 = ((df['high'] + df['low']) / 2).values
    upper_basic = hl2 + (mult * atr)
    lower_basic = hl2 - (mult * atr)
    upper = upper_basic.copy()
    lower = lower_basic.copy()
    close = df['close'].values
    in_uptrend = np.full(len(df), np.nan)
    
    first_valid = atr_len * 2
    if len(df) > first_valid:
        in_uptrend[first_valid] = 1 if close[first_valid] > lower_basic[first_valid-1] else 0  # Adjusted initial to assume up if above lower
    
    for i in range(first_valid + 1, len(df)):
        if close[i-1] > upper[i-1]:
            in_uptrend[i] = 1  # Flip to uptrend
        elif close[i-1] < lower[i-1]:
            in_uptrend[i] = 0  # Flip to downtrend
        else:
            in_uptrend[i] = in_uptrend[i-1]
        
        if in_uptrend[i] == 1:
            lower[i] = max(lower[i], lower[i-1])
        else:
            upper[i] = min(upper[i], upper[i-1])
    
    df['atr'] = atr
    df['upper'] = upper
    df['lower'] = lower
    df['in_uptrend'] = (in_uptrend == 1)
    
    df = df.fillna(False).replace([np.inf, -np.inf], 0)
    return df

# Backtest function (Supertrend flips) - Added debug prints
def backtest(df, params):
    atr_len = params['atr_len']
    mult = params['mult']
    sl_mult = params['sl_mult']
    tp_mult = params['tp_mult']
    
    min_start = atr_len * 3  # Larger buffer for Supertrend stabilization
    
    position = 0
    balance = 10000  # Start small for testing
    entry_price = 0
    sl_level = 0
    tp_level = 0
    trades = 0
    wins = 0
    losses = 0
    net_worth_history = [balance]
    buy_signals = 0
    sell_signals = 0

    for i in range(min_start, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Buy signal: Flip to uptrend
        buy_signal = curr['in_uptrend'] and not prev['in_uptrend']
        if buy_signal:
            buy_signals += 1
        
        # Sell signal: Flip to downtrend
        sell_signal = not curr['in_uptrend'] and prev['in_uptrend']
        if sell_signal:
            sell_signals += 1
        
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
    trade_penalty = abs(trades - 30) * 0.1  # Aim for moderate trades
    score = profit_pct - abs(max_dd) + win_rate - trade_penalty
    if trades < 10:
        score -= 100  # Penalize low activity
    
    # Compute buy-and-hold for comparison (over the backtested period)
    if len(df) > min_start:
        bh_profit = ((df['close'].iloc[-1] / df['close'].iloc[min_start]) - 1) * 100
    else:
        bh_profit = 0
    
    print(f"Backtest summary for params {params}: trades={trades}, buy_signals={buy_signals}, sell_signals={sell_signals}, bh_profit={bh_profit:.2f}%")
    
    return {
        'score': score,
        'profit_pct': profit_pct,
        'max_dd': max_dd,
        'trades': trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'bh_profit': bh_profit
    }

# Objective for Optuna (walk-forward)
def objective(trial):
    atr_len = trial.suggest_int('atr_len', 5, 20)
    mult = trial.suggest_float('mult', 1.0, 5.0)
    sl_mult = trial.suggest_float('sl_mult', 1.5, 3.0)
    tp_mult = trial.suggest_float('tp_mult', 2.0, 5.0)
    
    params = trial.params
    
    # Full data
    df_full = fetch_historical_data()
    
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
        df_slice = df_full.iloc[train_start:train_end + holdout_rows].copy()
        
        df_slice = compute_indicators(df_slice, atr_len, mult)
        
        holdout_df = df_slice.iloc[train_rows:]
        
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
    avg_bh_profit = np.mean([m['bh_profit'] for m in fold_metrics])
    
    # Print details for this trial
    print(f"Trial {trial.number}:")
    print(f"  Avg Score: {avg_score:.2f}")
    print(f"  Avg Profit %: {avg_profit_pct:.2f}")
    print(f"  Avg Buy-Hold Profit %: {avg_bh_profit:.2f}")
    print(f"  Avg Max DD: {avg_max_dd:.2f}")
    print(f"  Avg Trades: {avg_trades:.2f}")
    print(f"  Avg Wins: {avg_wins:.2f}")
    print(f"  Avg Losses: {avg_losses:.2f}")
    print(f"  Avg Win Rate: {avg_win_rate:.2f}%")
    print(f"  Params: {params}")
    
    return avg_score

# Run optimization
storage = "sqlite:///supertrend_strategy_fixed.db"  # New DB for fixed strategy
study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42), pruner=MedianPruner(), storage=storage, study_name="supertrend_strategy_fixed", load_if_exists=True)
study.optimize(objective, n_trials=50)  # More trials for broader search

print("Best params:", study.best_params)
print("Best score:", study.best_value)