require('dotenv').config();
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const math = require('mathjs');

// Constants
const SOL_MINT = 'So11111111111111111111111111111111111111112';
const RSI_PERIOD = 10;
const SMA_PERIOD = 20;
const MAX_SOL_POSITION = 20;
const INITIAL_USDC_BALANCE = 5000;
const SIMULATED_SOL_BALANCE = 0;
const TIMEFRAME = '15m';
const DAYS_BACK = 30;
const CHUNK_DAYS = 14;
const MAX_HOLD_INTERVALS = 48;
const TRANSACTION_FEE = 0.001;
const QTABLE_FILE = path.join(__dirname, 'qtable.json');

// Q-learning variables
let qTable = loadQTable();
let learningRate = 0.1;
let discountFactor = 0.9;
let explorationRate = 0.2;
let recentPerformance = 0; // Track recent profitability for adaptability

// Condition failure counters
let buyConditionFailures = { rsi: 0, trend: 0, smaSlope: 0, fib: 0 };
let sellConditionFailures = { rsi: 0, trend: 0, smaSlope: 0, fib: 0 };

// Fibonacci levels
const FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786];

// Actions array with dynamic Fib consideration
const actions = [
  { rsiBuy: 25, rsiSell: 75, tpMultiplier: 15, slMultiplier: 1.5, fibBuyLevel: 0.618, fibSellLevel: 0.382, type: 'full', trendStrength: 'strong' },
  { rsiBuy: 30, rsiSell: 70, tpMultiplier: 14, slMultiplier: 1.5, fibBuyLevel: 0.5, fibSellLevel: 0.5, type: 'full', trendStrength: 'moderate' },
  { rsiBuy: 35, rsiSell: 65, tpMultiplier: 13, slMultiplier: 1.5, fibBuyLevel: 0.382, fibSellLevel: 0.618, type: 'hold', trendStrength: 'weak' },
  { rsiBuy: 20, rsiSell: 80, tpMultiplier: 18, slMultiplier: 1.2, fibBuyLevel: 0.786, fibSellLevel: 0.236, type: 'full', trendStrength: 'bull' }, // Bull run action
];

// Delay function to avoid API rate limits
function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Fetch historical price data from Birdeye in chunks
async function fetchHistoricalData() {
  try {
    const endTime = Math.floor(Date.now() / 1000);
    const startTime = endTime - (DAYS_BACK * 24 * 60 * 60);
    let allPrices = [];
    let currentStartTime = startTime;

    while (currentStartTime < endTime) {
      const chunkEndTime = Math.min(currentStartTime + (CHUNK_DAYS * 24 * 60 * 60), endTime);
      console.log(`[Backtest] Fetching ${TIMEFRAME} chunk from ${new Date(currentStartTime * 1000).toISOString()} to ${new Date(chunkEndTime * 1000).toISOString()}`);

      const response = await axios.get(
        `https://public-api.birdeye.so/defi/history_price?address=${SOL_MINT}&type=${TIMEFRAME}&time_from=${currentStartTime}&time_to=${chunkEndTime}`,
        { headers: { 'X-API-KEY': process.env.BIRDEYE_API_KEY } }
      );

      if (response.data.success) {
        const items = response.data.data.items;
        const prices = items.map(item => ({ timestamp: item.unixTime, price: item.value }));
        allPrices = allPrices.concat(prices);
      } else {
        console.error('[Backtest] API response unsuccessful:', response.data);
      }

      currentStartTime = chunkEndTime;
      await delay(2000);
    }

    allPrices.sort((a, b) => a.timestamp - b.timestamp);
    console.log(`[Backtest] Fetched ${allPrices.length} ${TIMEFRAME} price points`);
    return allPrices;
  } catch (error) {
    console.error('[Backtest] Error fetching historical data:', error.message);
    return [];
  }
}

// Calculate Fibonacci levels
function calculateFibonacciLevels(prices) {
  if (prices.length < 2) return { levels: {}, trend: 'none', strength: 'weak' };
  
  const high = Math.max(...prices.slice(-20));
  const low = Math.min(...prices.slice(-20));
  const diff = high - low;
  
  const levels = {};
  FIB_LEVELS.forEach(level => {
    levels[level] = low + diff * level;
  });
  
  const trend = prices[prices.length - 1] > prices[prices.length - 2] ? 'up' : 'down';
  const priceChange = (prices[prices.length - 1] - prices[prices.length - 5]) / prices[prices.length - 5];
  const strength = priceChange > 0.05 ? 'bull' : priceChange > 0.01 ? 'strong' : priceChange > -0.01 ? 'moderate' : 'weak';
  return { levels, trend, strength, high, low };
}

// Load Q-table
function loadQTable() {
  try {
    if (fs.existsSync(QTABLE_FILE)) {
      const data = fs.readFileSync(QTABLE_FILE, 'utf8');
      return JSON.parse(data);
    }
    return {};
  } catch (error) {
    console.error('[QTable] Error loading Q-table:', error.message);
    return {};
  }
}

// Save Q-table
function saveQTable() {
  try {
    fs.writeFileSync(QTABLE_FILE, JSON.stringify(qTable, null, 2));
    console.log('[QTable] Q-table saved successfully');
  } catch (error) {
    console.error('[QTable] Error saving Q-table:', error.message);
  }
}

// Calculate RSI
function calculateRSI(prices) {
  if (prices.length < RSI_PERIOD + 1) return null;
  let gains = 0, losses = 0;
  for (let i = prices.length - RSI_PERIOD - 1; i < prices.length - 1; i++) {
    const change = prices[i + 1] - prices[i];
    if (change > 0) gains += change;
    else losses -= change;
  }
  const avgGain = gains / RSI_PERIOD;
  const avgLoss = losses / RSI_PERIOD;
  const rs = avgLoss === 0 ? (avgGain === 0 ? 0 : Infinity) : avgGain / avgLoss;
  return rs === Infinity ? 100 : 100 - (100 / (1 + rs));
}

// Calculate SMA
function calculateSMA(prices, period) {
  if (prices.length < period) return null;
  return math.mean(prices.slice(-period));
}

// Calculate SMA slope
function calculateSMASlope(smaHistory) {
  if (smaHistory.length < 5) return 0;
  const recentSMAs = smaHistory.slice(-5);
  return (recentSMAs[4] - recentSMAs[0]) / 4;
}

// Calculate ATR
function calculateATR(prices, period = 14) {
  if (prices.length < period + 1) return 0;
  const tr = prices.slice(-period - 1).map((p, i) => {
    if (i === 0) return 0;
    return Math.max(p - prices[i - 1], Math.abs(p - prices[i - 1]));
  }).slice(1);
  return math.mean(tr);
}

// Q-learning: Get state
function getState(rsi, smaSlope, volatility, hasPosition, fibProximity, trendStrength) {
  const rsiLevel = rsi < 35 ? 'low' : rsi < 65 ? 'mid' : 'high';
  const trend = smaSlope > 0.2 ? 'strong_up' : smaSlope > 0 ? 'up' : smaSlope < -0.2 ? 'strong_down' : 'down';
  const volatilityLevel = volatility < 0.08 ? 'low' : volatility < 0.09 ? 'mid' : 'high';
  const positionStatus = hasPosition ? 'holding' : 'flat';
  const fibLevel = fibProximity < 0.02 ? 'near' : fibProximity < 0.05 ? 'mid' : 'far';
  return `${rsiLevel}_${trend}_${volatilityLevel}_${positionStatus}_${fibLevel}_${trendStrength}`;
}

// Q-learning: Get action
function getAction(state) {
  if (!qTable[state]) qTable[state] = actions.map(() => 0);

  if (Math.random() < explorationRate) {
    return actions[Math.floor(Math.random() * actions.length)];
  }

  const qValues = qTable[state];
  const maxQ = Math.max(...qValues);
  const bestActionIndex = qValues.indexOf(maxQ);
  return actions[bestActionIndex];
}

// Q-learning: Update Q-table
function updateQTable(state, actionIndex, reward, nextState, holdIntervals, smaSlope) {
  if (!qTable[nextState]) qTable[nextState] = Array(actions.length).fill(0);
  const currentQ = qTable[state][actionIndex];
  const maxFutureQ = Math.max(...qTable[nextState]);
  const trendBonus = smaSlope > 0.2 ? holdIntervals * 0.15 : smaSlope < -0.2 ? -holdIntervals * 0.1 : 0;
  const adjustedReward = reward + trendBonus;
  const newQ = currentQ + learningRate * (adjustedReward + discountFactor * maxFutureQ - currentQ);
  qTable[state][actionIndex] = newQ;

  // Adaptive learning adjustments
  explorationRate = Math.max(0.05, explorationRate * (recentPerformance > 0 ? 0.99 : 0.995));
  learningRate = Math.min(0.2, Math.max(0.05, learningRate * (recentPerformance > 0 ? 1.01 : 0.99)));
  saveQTable();
}

// Log trades to CSV
function logBacktestTrade(timestamp, tradeType, price, solAmount, usdcAmount, profit, holdIntervals) {
  const logEntry = `${new Date(timestamp * 1000).toISOString()},${timestamp},${tradeType},${price},${solAmount},${usdcAmount},${profit},${holdIntervals}\n`;
  const logFilePath = path.join(__dirname, 'backtest_trades.csv');
  if (!fs.existsSync(logFilePath)) {
    fs.writeFileSync(logFilePath, 'timestamp,unixTimestamp,tradeType,price(SOL/USDC),solAmount,usdcAmount,profit(USDC),holdIntervals\n');
  }
  fs.appendFileSync(logFilePath, logEntry);
}

// Backtest function
async function backtest() {
  console.log('[Backtest] Starting backtest...');

  const historicalData = await fetchHistoricalData();
  if (historicalData.length === 0) {
    console.error(`[Backtest] No ${TIMEFRAME} data available. Exiting.`);
    return;
  }

  let priceHistory = [], smaHistory = [];
  let simulatedPositions = [];
  let usdcBalance = INITIAL_USDC_BALANCE;
  let solBalance = SIMULATED_SOL_BALANCE;
  let totalProfit = 0, wins = 0, trades = 0;
  let indicatorsAvailable = false;
  let lastPrice = null;
  let volatilityValues = [];
  let winProfits = [], lossProfits = [];
  let winHoldIntervals = [], lossHoldIntervals = [];

  // Calculate volatility stats for adaptive sizing
  for (const { price } of historicalData) {
    priceHistory.push(price);
    if (priceHistory.length > Math.max(RSI_PERIOD + 1, SMA_PERIOD + 5, 20)) priceHistory.shift();
    if (priceHistory.length >= 20) {
      const volatility = math.std(priceHistory.slice(-20)) / math.mean(priceHistory.slice(-20));
      volatilityValues.push(volatility);
    }
  }
  const meanVolatility = math.mean(volatilityValues);
  const stdVolatility = math.std(volatilityValues);
  const volatilityThreshold = meanVolatility + 1.5 * stdVolatility;
  priceHistory = [];

  for (const { timestamp, price } of historicalData) {
    lastPrice = price;
    priceHistory.push(price);
    if (priceHistory.length > Math.max(RSI_PERIOD + 1, SMA_PERIOD + 5, 20)) priceHistory.shift();

    const rsi = calculateRSI(priceHistory);
    const smaTrend = calculateSMA(priceHistory, SMA_PERIOD);
    smaHistory.push(smaTrend);
    if (smaHistory.length > 5) smaHistory.shift();
    const atr = calculateATR(priceHistory);
    const volatility = math.std(priceHistory.slice(-20)) / math.mean(priceHistory.slice(-20));
    const smaSlope = calculateSMASlope(smaHistory);
    const fib = calculateFibonacciLevels(priceHistory);
    
    const fibProximity = Math.min(...FIB_LEVELS.map(level => 
      Math.abs(price - fib.levels[level]) / price
    ));

    if (!rsi || !smaTrend) {
      if (!indicatorsAvailable) console.log(`[Backtest] Waiting for indicators at ${new Date(timestamp * 1000).toISOString()}`);
      continue;
    }
    if (!indicatorsAvailable) {
      console.log(`[Backtest] Indicators available at ${new Date(timestamp * 1000).toISOString()}`);
      indicatorsAvailable = true;
    }

    if (volatility < volatilityThreshold) {
      if (historicalData.indexOf({ timestamp, price }) % 48 === 0) {
        console.log(`[Backtest] Skipping trade at ${new Date(timestamp * 1000).toISOString()} due to low volatility: ${volatility.toFixed(4)}`);
      }
      continue;
    }

    const state = getState(rsi, smaSlope, volatility, simulatedPositions.length > 0, fibProximity, fib.strength);
    const action = getAction(state);
    const { rsiBuy, rsiSell, tpMultiplier, slMultiplier, fibBuyLevel, fibSellLevel, type } = action;

    // Manage open positions
    for (let i = simulatedPositions.length - 1; i >= 0; i--) {
      const position = simulatedPositions[i];
      position.holdIntervals = (position.holdIntervals || 0) + 1;

      const unrealizedProfit = (price - position.entryPrice) * position.solAmount;
      const minTrailProfit = position.entryPrice * 0.03;
      const trailAmount = unrealizedProfit > minTrailProfit ? Math.max(atr, unrealizedProfit * 0.5) : atr;
      if (price > position.entryPrice && price - trailAmount > position.trailingStop) {
        position.trailingStop = price - trailAmount;
      }

      if (price >= position.takeProfit || price <= position.stopLoss || price <= position.trailingStop || position.holdIntervals >= MAX_HOLD_INTERVALS) {
        const profit = (price - position.entryPrice) * position.solAmount;
        const fee = position.tradeAmount * TRANSACTION_FEE;
        usdcBalance += position.tradeAmount + profit - fee;
        solBalance -= position.solAmount;
        totalProfit += profit - fee;
        trades++;
        recentPerformance = profit - fee; // Update recent performance
        if (profit > 0) {
          wins++;
          winProfits.push(profit - fee);
          winHoldIntervals.push(position.holdIntervals);
        } else {
          lossProfits.push(profit - fee);
          lossHoldIntervals.push(position.holdIntervals);
        }
        const tradeType = position.holdIntervals >= MAX_HOLD_INTERVALS ? 'SELL (Time Exit)' : 'SELL (TP/SL)';
        logBacktestTrade(timestamp, tradeType, price, position.solAmount, position.tradeAmount + profit - fee, profit - fee, position.holdIntervals);

        const profitPercentage = (profit - fee) / position.tradeAmount;
        const reward = (profit - fee) > 0 ? (profitPercentage * 10 + position.holdIntervals * 0.05) : -5;
        const nextState = getState(rsi, smaSlope, volatility, false, fibProximity, fib.strength);
        const actionIndex = actions.findIndex(a => a.rsiBuy === rsiBuy && a.type === type);
        updateQTable(state, actionIndex, reward, nextState, position.holdIntervals, smaSlope);

        simulatedPositions.splice(i, 1);
      }
    }

    let tradeAmount;
    const absSlope = Math.abs(smaSlope);
    if (fib.strength === 'bull' || volatility > meanVolatility + stdVolatility || absSlope > 0.2) {
      tradeAmount = usdcBalance * 0.15; // Aggressive sizing in bull runs
    } else if (volatility > meanVolatility - stdVolatility || absSlope > 0.1) {
      tradeAmount = usdcBalance * 0.09;
    } else {
      tradeAmount = usdcBalance * 0.07;
    }
    tradeAmount = Math.max(50, tradeAmount);

    const buyConditions = {
      rsi: rsi < rsiBuy,
      trend: price > smaTrend * 0.95,
      smaSlope: smaSlope > -0.3,
      fib: Math.abs(price - fib.levels[fibBuyLevel]) / price < 0.03,
    };
    const sellConditions = {
      rsi: rsi > rsiSell,
      trend: price < smaTrend * 1.05,
      smaSlope: smaSlope < 0.3,
      fib: Math.abs(price - fib.levels[fibSellLevel]) / price < 0.03,
    };

    for (const [key, value] of Object.entries(buyConditions)) if (!value) buyConditionFailures[key]++;
    for (const [key, value] of Object.entries(sellConditions)) if (!value) sellConditionFailures[key]++;

    if (simulatedPositions.length === 0) {
      const stateNoPosition = getState(rsi, smaSlope, volatility, false, fibProximity, fib.strength);
      const actionIndex = actions.findIndex(a => a.rsiBuy === rsiBuy && a.type === type);
      updateQTable(stateNoPosition, actionIndex, -2.0, stateNoPosition, 0, smaSlope);
    }

    if (simulatedPositions.length === 0) {
      if (buyConditions.rsi && buyConditions.trend && buyConditions.smaSlope && buyConditions.fib) {
        if (usdcBalance < tradeAmount || solBalance + (tradeAmount / price) > MAX_SOL_POSITION) continue;

        const solAmount = tradeAmount / price;
        const fee = tradeAmount * TRANSACTION_FEE;
        const atrScaled = atr * (tradeAmount / 500);
        const takeProfit = price + (tpMultiplier * atrScaled * (fib.strength === 'bull' ? 1.2 : 1));
        const stopLoss = price - (slMultiplier * atrScaled * (fib.strength === 'bull' ? 0.8 : 1));
        const trailingStop = stopLoss;

        usdcBalance -= (tradeAmount + fee);
        solBalance += solAmount;
        totalProfit -= fee;
        simulatedPositions.push({ entryPrice: price, takeProfit, stopLoss, solAmount, trailingStop, holdIntervals: 0, tradeAmount });
        logBacktestTrade(timestamp, 'BUY', price, solAmount, tradeAmount, -fee, 0);
      } else if (sellConditions.rsi && sellConditions.trend && sellConditions.smaSlope && sellConditions.fib && simulatedPositions.length > 0) {
        const position = simulatedPositions.pop();
        const profit = (price - position.entryPrice) * position.solAmount;
        const fee = position.tradeAmount * TRANSACTION_FEE;
        usdcBalance += position.tradeAmount + profit - fee;
        solBalance -= position.solAmount;
        totalProfit += profit - fee;
        trades++;
        recentPerformance = profit - fee;
        if (profit > 0) {
          wins++;
          winProfits.push(profit - fee);
          winHoldIntervals.push(position.holdIntervals);
        } else {
          lossProfits.push(profit - fee);
          lossHoldIntervals.push(position.holdIntervals);
        }
        logBacktestTrade(timestamp, 'SELL', price, position.solAmount, position.tradeAmount + profit - fee, profit - fee, position.holdIntervals);

        const profitPercentage = (profit - fee) / position.tradeAmount;
        const reward = (profit - fee) > 0 ? (profitPercentage * 10 + position.holdIntervals * 0.05) : -5;
        const nextState = getState(rsi, smaSlope, volatility, false, fibProximity, fib.strength);
        const actionIndex = actions.findIndex(a => a.rsiBuy === rsiBuy && a.type === type);
        updateQTable(state, actionIndex, reward, nextState, position.holdIntervals, smaSlope);
      }
    }
  }

  if (simulatedPositions.length > 0 && lastPrice) {
    for (const position of simulatedPositions) {
      const profit = (lastPrice - position.entryPrice) * position.solAmount;
      const fee = position.tradeAmount * TRANSACTION_FEE;
      usdcBalance += position.tradeAmount + profit - fee;
      solBalance -= position.solAmount;
      totalProfit += profit - fee;
      trades++;
      if (profit > 0) {
        wins++;
        winProfits.push(profit - fee);
        winHoldIntervals.push(position.holdIntervals);
      } else {
        lossProfits.push(profit - fee);
        lossHoldIntervals.push(position.holdIntervals);
      }
      logBacktestTrade(historicalData[historicalData.length - 1].timestamp, 'SELL (Liquidation)', lastPrice, position.solAmount, position.tradeAmount + profit - fee, profit - fee, position.holdIntervals);
    }
  }

  const winRate = trades > 0 ? (wins / trades) * 100 : 0;
  const avgWin = winProfits.length > 0 ? math.mean(winProfits) : 0;
  const avgLoss = lossProfits.length > 0 ? math.mean(lossProfits) : 0;
  const avgWinHold = winHoldIntervals.length > 0 ? math.mean(winHoldIntervals) : 0;
  const avgLossHold = lossHoldIntervals.length > 0 ? math.mean(lossHoldIntervals) : 0;
  console.log(`[Backtest] Results:`);
  console.log(`- Total Profit: ${totalProfit.toFixed(2)} USDC`);
  console.log(`- Win Rate: ${winRate.toFixed(2)}%`);
  console.log(`- Total Trades: ${trades}`);
  console.log(`- Average Win: ${avgWin.toFixed(2)} USDC`);
  console.log(`- Average Loss: ${avgLoss.toFixed(2)} USDC`);
  console.log(`- Average Win Hold Time: ${avgWinHold.toFixed(2)} intervals`);
  console.log(`- Average Loss Hold Time: ${avgLossHold.toFixed(2)} intervals`);
  console.log(`- Final USDC Balance: ${usdcBalance.toFixed(2)}`);
  console.log(`- Final SOL Balance: ${solBalance.toFixed(2)}`);
  console.log(`[Backtest] Buy Condition Failures:`, buyConditionFailures);
  console.log(`[Backtest] Sell Condition Failures:`, sellConditionFailures);
  console.log(`[Backtest] Volatility Stats: Min=${Math.min(...volatilityValues).toFixed(4)}, Max=${Math.max(...volatilityValues).toFixed(4)}, Mean=${math.mean(volatilityValues).toFixed(4)}, Std=${stdVolatility.toFixed(4)}`);
}

// Run backtest
backtest().catch(console.error);