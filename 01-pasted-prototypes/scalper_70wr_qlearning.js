require('dotenv').config();
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const math = require('mathjs');

// Constants
const SOL_MINT = 'So11111111111111111111111111111111111111112';
const RSI_PERIOD = 6;
const SMA_SHORT = 4;
const SMA_LONG = 10;
const MAX_SOL_POSITION = 5;
const INITIAL_USDC_BALANCE = 5000;
const SIMULATED_SOL_BALANCE = 0;
const TIMEFRAME = '1m';
const DAYS_BACK = 7;
const CHUNK_DAYS = 3;
const MAX_HOLD_INTERVALS = 12;
const TRANSACTION_FEE = 0.0005;
const MAX_DRAWDOWN = 0.20;
const QTABLE_FILE = path.join(__dirname, 'qtable_scalping_final_profit_push.json');

// Q-learning variables
let qTable = loadQTable();
let learningRate = 0.12;
let discountFactor = 0.88;
let explorationRate = 0.2;
let recentPerformance = 0;

// Adjusted actions with higher TP for better profit
const actions = [
  { rsiBuy: 30, rsiSell: 70, baseTp: 2.0, baseSl: 0.6, type: 'aggressive' }, // Increased TP from 1.8 to 2.0
  { rsiBuy: 35, rsiSell: 65, baseTp: 1.8, baseSl: 0.4, type: 'moderate' },     // Increased TP from 1.6 to 1.8
  { rsiBuy: 27, rsiSell: 73, baseTp: 2.2, baseSl: 0.7, type: 'conservative' }, // Increased TP from 2.0 to 2.2
];

// Delay function
function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Fetch historical data (unchanged)
async function fetchHistoricalData() {
  try {
    const endTime = Math.floor(Date.now() / 1000);
    const startTime = endTime - (DAYS_BACK * 24 * 60 * 60);
    let allPrices = [];
    let currentStartTime = startTime;

    while (currentStartTime < endTime) {
      const chunkEndTime = Math.min(currentStartTime + (CHUNK_DAYS * 24 * 60 * 60), endTime);
      const response = await axios.get(
        `https://public-api.birdeye.so/defi/history_price?address=${SOL_MINT}&type=${TIMEFRAME}&time_from=${currentStartTime}&time_to=${chunkEndTime}`,
        { headers: { 'X-API-KEY': process.env.BIRDEYE_API_KEY } }
      );

      if (response.data.success) {
        const prices = response.data.data.items.map(item => ({ timestamp: item.unixTime, price: item.value }));
        allPrices = allPrices.concat(prices);
      }
      currentStartTime = chunkEndTime;
      await delay(1000);
    }

    allPrices.sort((a, b) => a.timestamp - b.timestamp);
    return allPrices;
  } catch (error) {
    console.error('[Scalping] Error fetching data:', error.message);
    return [];
  }
}

// Q-table functions (unchanged)
function loadQTable() {
  try {
    if (fs.existsSync(QTABLE_FILE)) return JSON.parse(fs.readFileSync(QTABLE_FILE, 'utf8'));
    return {};
  } catch (error) {
    console.error('[QTable] Error loading:', error.message);
    return {};
  }
}

function saveQTable() {
  try {
    fs.writeFileSync(QTABLE_FILE, JSON.stringify(qTable, null, 2));
  } catch (error) {
    console.error('[QTable] Error saving:', error.message);
  }
}

// Technical indicators
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
  const rs = avgLoss === 0 ? Infinity : avgGain / avgLoss;
  return rs === Infinity ? 100 : 100 - (100 / (1 + rs));
}

function calculateSMA(prices, period) {
  if (prices.length < period) return null;
  return math.mean(prices.slice(-period));
}

function calculateATR(prices, period = 6) {
  if (prices.length < period + 1) return 0;
  const tr = prices.slice(-period - 1).map((p, i) => {
    if (i === 0) return 0;
    return Math.max(p - prices[i - 1], Math.abs(p - prices[i - 1]));
  }).slice(1);
  return math.mean(tr);
}

function calculateMomentum(prices, period = 5) {
  if (prices.length < period) return 0;
  return prices[prices.length - 1] - prices[prices.length - period];
}

// State definition (unchanged)
function getState(rsi, volatility, hasPosition, trend, momentum) {
  const rsiLevel = rsi < 33 ? 'low' : rsi < 67 ? 'mid' : 'high';
  const volatilityLevel = volatility < 0.03 ? 'low' : volatility < 0.06 ? 'mid' : 'high';
  const positionStatus = hasPosition ? 'holding' : 'flat';
  const trendStatus = trend > 0 ? 'up' : trend < 0 ? 'down' : 'flat';
  const momentumStatus = momentum > 0 ? 'positive' : 'negative';
  return `${rsiLevel}_${volatilityLevel}_${positionStatus}_${trendStatus}_${momentumStatus}`;
}

function getAction(state) {
  if (!qTable[state]) qTable[state] = actions.map(() => 0);
  return Math.random() < explorationRate 
    ? actions[Math.floor(Math.random() * actions.length)]
    : actions[qTable[state].indexOf(Math.max(...qTable[state]))];
}

function updateQTable(state, actionIndex, reward, nextState) {
  if (!qTable[nextState]) qTable[nextState] = Array(actions.length).fill(0);
  const currentQ = qTable[state][actionIndex];
  const maxFutureQ = Math.max(...qTable[nextState]);
  const newQ = currentQ + learningRate * (reward + discountFactor * maxFutureQ - currentQ);
  qTable[state][actionIndex] = newQ;

  explorationRate = Math.max(0.05, Math.min(0.25, explorationRate * (recentPerformance > 0 ? 0.99 : 0.97)));
  saveQTable();
}

// Logging (unchanged)
function logBacktestTrade(timestamp, tradeType, price, solAmount, usdcAmount, profit, holdIntervals) {
  try {
    const logEntry = `${new Date(timestamp * 1000).toISOString()},${timestamp},${tradeType},${price},${solAmount},${usdcAmount},${profit},${holdIntervals}\n`;
    const logFilePath = path.join(__dirname, 'scalping_trades_final_profit_push.csv');
    if (!fs.existsSync(logFilePath)) {
      fs.writeFileSync(logFilePath, 'timestamp,unixTimestamp,tradeType,price(SOL/USDC),solAmount,usdcAmount,profit(USDC),holdIntervals\n');
    }
    fs.appendFileSync(logFilePath, logEntry);
  } catch (error) {
    console.error('[Logging] Error writing to log file:', error.message);
  }
}

async function backtest() {
  try {
    console.log('[Scalping] Starting final-profit-push backtest...');
    const historicalData = await fetchHistoricalData();
    if (!historicalData.length) {
      console.error('[Scalping] No data available.');
      return;
    }

    let priceHistory = [];
    let positions = [];
    let usdcBalance = INITIAL_USDC_BALANCE;
    let solBalance = SIMULATED_SOL_BALANCE;
    let totalProfit = 0, wins = 0, trades = 0;
    let lastPrice = null;
    let peakBalance = INITIAL_USDC_BALANCE;

    // Volatility threshold (unchanged)
    let volatilityValues = [];
    for (const { price } of historicalData) {
      priceHistory.push(price);
      if (priceHistory.length > 10) priceHistory.shift();
      if (priceHistory.length >= 10) {
        const volatility = math.std(priceHistory) / math.mean(priceHistory);
        volatilityValues.push(volatility);
      }
    }
    const volatilityThreshold = math.mean(volatilityValues) * 0.5;
    priceHistory = [];

    for (const { timestamp, price } of historicalData) {
      lastPrice = price;
      priceHistory.push(price);
      if (priceHistory.length > Math.max(RSI_PERIOD + 1, SMA_LONG)) priceHistory.shift();

      const rsi = calculateRSI(priceHistory);
      const smaShort = calculateSMA(priceHistory, SMA_SHORT);
      const smaLong = calculateSMA(priceHistory, SMA_LONG);
      const atr = calculateATR(priceHistory);
      const volatility = math.std(priceHistory.slice(-10)) / math.mean(priceHistory.slice(-10));
      const trend = smaShort - smaLong;
      const momentum = calculateMomentum(priceHistory);

      if (!rsi || !smaShort || !smaLong) continue;

      if (volatility < volatilityThreshold * 0.8) continue;
      const currentBalance = usdcBalance + (solBalance * price);
      peakBalance = Math.max(peakBalance, currentBalance);
      if ((peakBalance - currentBalance) / peakBalance > MAX_DRAWDOWN) continue;

      const state = getState(rsi, volatility, positions.length > 0, trend, momentum);
      const action = getAction(state);

      // Manage positions
      for (let i = positions.length - 1; i >= 0; i--) {
        const pos = positions[i];
        pos.holdIntervals = (pos.holdIntervals || 0) + 1;

        const unrealizedProfit = (price - pos.entryPrice) / pos.entryPrice;
        if (unrealizedProfit > 0.003) { // Tightened trailing stop trigger from 0.005 to 0.003
          pos.stopLoss = Math.max(pos.stopLoss, price - (atr * action.baseSl * 0.5)); // Tightened multiplier from 0.6 to 0.5
        }

        if (price >= pos.takeProfit || price <= pos.stopLoss || pos.holdIntervals >= MAX_HOLD_INTERVALS) {
          const profit = (price - pos.entryPrice) * pos.solAmount;
          const fee = pos.tradeAmount * TRANSACTION_FEE;
          usdcBalance += pos.tradeAmount + profit - fee;
          solBalance -= pos.solAmount;
          totalProfit += profit - fee;
          trades++;
          recentPerformance = profit - fee;
          if (profit > 0) wins++;
          logBacktestTrade(timestamp, 'SELL', price, pos.solAmount, pos.tradeAmount + profit - fee, profit - fee, pos.holdIntervals);

          const reward = profit > 0 ? (profit / pos.tradeAmount) * 25 : -5;
          const nextState = getState(rsi, volatility, false, trend, momentum);
          const actionIndex = actions.findIndex(a => a.rsiBuy === action.rsiBuy);
          updateQTable(state, actionIndex, reward, nextState);

          positions.splice(i, 1);
        }
      }

      // New position with adjusted momentum filter
      if (positions.length < 3) {
        const volatilityFactor = Math.min(1.8, Math.max(0.7, volatility / volatilityThreshold));
        const baseTradeAmount = usdcBalance * 0.02 * volatilityFactor; // Increased from 0.018 to 0.02
        const tradeAmount = Math.min(baseTradeAmount, 175, usdcBalance * 0.04); // Increased max from 150 to 175
        
        // Tightened momentum filter
        if (((rsi < action.rsiBuy && momentum > 0.08) || (rsi < action.rsiBuy - 5 && trend > 0)) && 
            usdcBalance >= tradeAmount && solBalance + (tradeAmount / price) <= MAX_SOL_POSITION) {
          const solAmount = tradeAmount / price;
          const fee = tradeAmount * TRANSACTION_FEE;
          const tpMultiplier = action.baseTp * (1 + (volatility - volatilityThreshold) * 0.6);
          const slMultiplier = action.baseSl * (1 - (volatility - volatilityThreshold) * 0.4);
          const takeProfit = price + (tpMultiplier * atr);
          const stopLoss = price - (slMultiplier * atr);

          usdcBalance -= (tradeAmount + fee);
          solBalance += solAmount;
          totalProfit -= fee;
          positions.push({ entryPrice: price, takeProfit, stopLoss, solAmount, tradeAmount, holdIntervals: 0 });
          logBacktestTrade(timestamp, 'BUY', price, solAmount, tradeAmount, -fee, 0);
        } else if (((rsi > action.rsiSell && momentum < -0.08) || (rsi > action.rsiSell + 5 && trend < 0)) && 
                   positions.length > 0) {
          const pos = positions.pop();
          const profit = (price - pos.entryPrice) * pos.solAmount;
          const fee = pos.tradeAmount * TRANSACTION_FEE;
          usdcBalance += pos.tradeAmount + profit - fee;
          solBalance -= pos.solAmount;
          totalProfit += profit - fee;
          trades++;
          recentPerformance = profit - fee;
          if (profit > 0) wins++;
          logBacktestTrade(timestamp, 'SELL', price, pos.solAmount, pos.tradeAmount + profit - fee, profit - fee, pos.holdIntervals);

          const reward = profit > 0 ? (profit / pos.tradeAmount) * 25 : -5;
          const nextState = getState(rsi, volatility, false, trend, momentum);
          const actionIndex = actions.findIndex(a => a.rsiBuy === action.rsiBuy);
          updateQTable(state, actionIndex, reward, nextState);
        }
      }
    }

    // Liquidate remaining positions
    if (positions.length > 0 && lastPrice) {
      for (const pos of positions) {
        const profit = (lastPrice - pos.entryPrice) * pos.solAmount;
        const fee = pos.tradeAmount * TRANSACTION_FEE;
        usdcBalance += pos.tradeAmount + profit - fee;
        solBalance -= pos.solAmount;
        totalProfit += profit - fee;
        trades++;
        if (profit > 0) wins++;
        logBacktestTrade(historicalData[historicalData.length - 1].timestamp, 'SELL (Liquidation)', lastPrice, pos.solAmount, pos.tradeAmount + profit - fee, profit - fee, pos.holdIntervals);
      }
    }

    const winRate = trades > 0 ? (wins / trades) * 100 : 0;
    console.log(`[Scalping] Final-Profit-Push Results:`);
    console.log(`- Total Profit: ${totalProfit.toFixed(2)} USDC`);
    console.log(`- Win Rate: ${winRate.toFixed(2)}%`);
    console.log(`- Total Trades: ${trades}`);
    console.log(`- Final USDC Balance: ${usdcBalance.toFixed(2)}`);
    console.log(`- Final SOL Balance: ${solBalance.toFixed(2)}`);
  } catch (error) {
    console.error('[Scalping] Backtest failed:', error.message);
  }
}

backtest().catch(error => console.error('[Scalping] Fatal error:', error.message));