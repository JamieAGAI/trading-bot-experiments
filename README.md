# trading-bot-experiments

> **Personal learning experiments. NOT financial advice. Nothing here is recommended for live trading.**
> Several of these scripts *could* sign real Solana transactions or call exchange APIs if given keys. Don't. All credentials were read from `.env` files that are not in this repo; one hard-coded wallet address was redacted (see "Scrubbing").

This repo is the history of how I went from pasting AI-generated trading bots to building a data collector with pre-registered hypotheses. It's organized as stages, in the order things happened. The code is kept close to how it was, bugs included, because the bugs are part of the point.

## Stages

### 01 — pasted prototypes (`01-pasted-prototypes/`)
Node.js bots pasted in from chat, saved as `.txt` (converted to `.js`/`.py` here). They pulled 1-minute SOL prices from Birdeye, traded with RSI(6)/SMA(4/10) rules, and some added a JSON **Q-table** (tabular Q-learning, learning rate 0.12) to "learn" when to buy. The file names record what I thought I had: "70% WR scalper", "61% WR day trader", "50% win with qtable".
- **What went wrong:** those win rates came from backtests over about 7 days of 1-minute data (`DAYS_BACK = 7`). A sample that small is noise, and the Q-table was tuned on the same window it was scored on. A high win rate also says nothing on its own about profit once fees and losing-trade size are counted.
- The live JS bots log the first 5 characters of the wallet private key at startup ("Debug"). That's a habit to drop: never log key material.
- I tried Raydium and Orca direct-swap versions (`sol_raydium.py`, `sol_orca.py`) before settling on the Jupiter aggregator.

### 02 — DQN + indicators (`02-dqn-indicators/`)
Moved to Python + TensorFlow/Keras: a **Deep Q-Network** (then Double-DQN, `double_dql.py`) on SOL/USDC with a replay buffer, epsilon-greedy exploration and model/epsilon checkpoints in `dql_model/` (see the README there; the binaries are excluded). The reward was realized profit minus fees, plus a small fraction of *unrealized* PnL while holding (`0.05`–`0.1 × unrealized`). The separate indicator scripts (stochastic dip-buyer, mean reversion, trend) were standalone rule backtests on Binance 4h klines, used to look for confluence.
- **What went wrong:** rewarding unrealized PnL lets the agent score points just by holding through a rally, so it learns the training period's drift rather than an edge. Per-regime models (bull/bear/sideways) were a patch for that. The file names `after issues` / `profit loss fixes` record a run of normalization, `iloc` and API bugs fixed one at a time.

### 03 — PPO reinforcement learning (`03-ppo-rl/`)
Switched to **PPO** (stable-baselines3) in a custom Gymnasium `TradingEnv` on 15-minute SOL/USDT, about 730 days from Binance, `RobustScaler` features, `SEQ_LEN = 40`, a 0.3% fee. **Optuna** (`opty*.py`, TPE sampler, 25–100 trials, SQLite storage) searched hyperparameters and strategy thresholds. The best was "trial 32", which `v3.py` trains with and saves as `ppo_trial32`. `v3.py` holds out the last 200 days for evaluation, and `evalmodel.py`/`evalv3.py` replay saved models on that holdout. `v2.py`/`v5.py`/`ppo_transformer_sol.py` wrap the model in a live loop that swaps via Jupiter (the transformer variant also pulled news and Twitter sentiment). The public repo `Solana-PPO-Trading-Bot-V1` comes from this stage.
- **What went wrong:** the reward is heavily shaped. It adds an `activity_incentive` for every buy and sell, a bonus for buying when RSI is low and ADX > 25, a hold penalty, rolling Sharpe/Sortino bonuses, and penalties for drawdown and unrealized loss. With that many terms the agent can collect reward by trading often and matching indicator conditions without making money (**reward hacking**). Picking "trial 32" out of dozens of Optuna trials and then checking it on a holdout that was also used to compare versions leads to **overfitting through selection**. Fees and slippage were partly modeled: `SLIPPAGE = 0.005` is defined but marked "unused in training".

### 04 — rule-based live SOL trader (`04-live-sol-trader/`)
I dropped RL for an explicit swing strategy. `main.py` ("v10") buys pullbacks in confirmed uptrends on 4h Binance data (pandas_ta), moves the stop to breakeven at +1R, trails after +2R, takes profit at +2.5R, and exits on trend, daily or timeout signals. It has a websocket feed, a `--shadow` paper mode (`shadow_trades.csv`, with state that survives restarts) and swaps through Jupiter. `backtest.py` replays the same rules; `perp.py` ports them to leveraged perps with funding cost and a liquidation guardrail; `trendfilter.py` adds a higher-timeframe filter; `swaptest.py` tests swap plumbing. `main_early.py`/`sol.py`/`oldmain.py` are earlier versions.
- **What improved:** the rules are readable, there's a shadow mode before going live, and backtest and live use the same rules.
- **⚠ Note:** `main.py`, `oldmain.py` and `backtest.py` carry a header `# Copyright (c) 2026 Laura Staples. All rights reserved.` It is kept unchanged and **flagged**: confirm the attribution and that you're fine publishing it, even privately. (`agent5.py` and `oldadvisor.py` say "James Staples".)

### 05 — LLM agent → deterministic rules advisor (`05-llm-agent-to-rules-advisor/`)
- `agent_v1.py`: an LLM (Grok via the x.ai OpenAI-compatible API) as the trader. It got multi-timeframe indicators and news RSS feeds and a prompt as "Jamie's autonomous SOL trading agent", plus a second "critic" prompt answering SAFE/RISKY. It had `LIVE_TRADING = True`.
- `agent5.py`: a much more structured version, with `LIVE_TRADING = False`. Each cycle it computes a **system conviction score** with a breakdown (e.g. `-1.5 below both daily EMAs | -1.0 CHOP regime`) and a signal "fingerprint" (`regime+div+macd+ema`). It asks the model for JSON (`thought`, `action`, `reason`), prints `🤖 THOUGHT / ACTION / System conviction / Reason`, and logs every decision. It later tags each one with the actual 12/24/36/48h high, low and close so decisions can be graded, and uses an LLM to write a one-sentence "lesson" after each trade. `analyzeagent5.py` / `agent5trades.py` grade the log.
- `oldadvisor.py` → `advisor.py`: an LLM perp *advisor* producing a bias, confidence, leverage, stop and targets, run in shadow mode with outcomes tagged (`advisor_shadow_perps.jsonl`; `perptrades.py`/`analyzeperps.py` compare against real Jupiter perps fills).
- `advisor_v2.py`: **"Replaces the LLM-as-trader with a pure-Python rules engine and a walk-forward backtest harness that replays history bar-by-bar with no lookahead."** It uses fixed thresholds, ATR stops and a minimum R:R of 1.5.

Excerpts from `sample_logs/` (first lines only):
- agent5 backfill: `action STAY, conviction 2.5, regime chop … conviction_breakdown "-1.5 below both daily EMAs | -1.0 CHOP regime"`, and the price then rose about 2.7% over 48h, so staying out missed the move.
- LLM advisor, 2026-03-30, SOL ≈ $84, bear regime: *"Exact setup shows 89% historical long win rate"* → bias long, 3x. Outcome: 24h close −4.0%, low −5.1%.
- Shadow perps, 2026-04-07: three consecutive 3–4x shorts at confidence 6.8–7.0, **all stopped out** within 12h (net −7.2%, −10.3%, −6.8% leveraged).
- **What went wrong:** the LLM sounded confident and cited win rates that were really small-sample or made up, so confidence didn't track outcomes. Leverage multiplied the losses on small adverse moves. Outcome tagging exposed this, which is why the decision moved into deterministic, backtestable rules.

### 06 — regime monitors → `solpy-monitor`
`solregime.py`, `btcregime.py` and `coldfront.py` are **alert-only** monitors (email, no wallet, no keys) built on weekly EMA50/RSI and daily EMA50/200 regimes using completed bars only. That led to the current project, **[solpy-monitor](https://github.com/JamieAGAI/solpy-monitor)** (private): a data collector that places no trades, plus hypotheses that are written down and locked before any outcome is examined, tested once with a minimum event count, net of costs, against shuffled-timestamp baselines.

## Lessons
1. A win rate from a few days of data is noise, and a high win rate alone says nothing about profit.
2. Every knob tuned on the data you grade on (Optuna trials, Q-table, thresholds) inflates the result. Hold data out, and don't reuse the holdout for selection.
3. RL reward shaping gets gamed: activity bonuses and unrealized-PnL rewards teach behavior, not edge.
4. Model fees, slippage and funding *in training*, not just at the end.
5. An LLM explains decisions well but is not calibrated. Log every call, tag it with real outcomes, and believe the tags.
6. Leverage turns an ordinary 2% adverse move into a −7% to −10% result.
7. Shadow mode and alert-only before anything live, and drop the live parts if the edge isn't proven.
8. Define the test before seeing the data (see solpy-monitor's `HYPOTHESES.md`).

## Scrubbing
- No API keys, private keys, RPC keys or passwords appear in any file; all are read via `os.getenv` / `process.env` from `.env` files that are not included. Placeholders such as `your_key`, `your_base58_key` and `you@gmail.com` remain.
- **Redacted:** a personal Solana wallet address hard-coded in `agent5trades.py` and as the default in `perptrades.py` (now `os.getenv("WALLET_ADDRESS", "")`).
- Decision logs are reduced to 3 sample lines each; the full logs, CSV data and model binaries are excluded.
- Public program, mint and pool addresses (SOL/USDC/USDT mints, Raydium/Orca/Jupiter programs) are left in; they're public.
