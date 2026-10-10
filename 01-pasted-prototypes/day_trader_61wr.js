require('dotenv').config();
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const math = require('mathjs');

// Constants
const SOL_MINT = 'So11111111111111111111111111111111111111112';
const INITIAL_USDC_BALANCE = 5000;
const SIMULATED_SOL_BALANCE = 0;
const MAX_SOL_POSITION = 20;
const TIMEFRAME = '15m';
const DAYS_BACK = 1000;
const CHUNK_DAYS = 14;
const TRANSACTION_FEE = 0.001;
const QTABLE_FILE = path.join(__dirname, 'qtable_trend_volume_v2.json');
const CACHE_FILE = path.join(__dirname, 'historical_data_cache.json');
const EMA_LONG_PERIOD = 50;
const EMA_SHORT_PERIOD = 20;
const VOLUME_MA_PERIOD = 20;
const ATR_PERIOD = 14;
const ADX_PERIOD = 14;
const MIN_ADX = 35;

// Q-learning variables
let qTable = loadQTable();
let learningRate = 0.05; // Lowered for smoother updates
let discountFactor = 0.9; // Slightly reduced for shorter-term focus
let explorationRate = 0.3; // Starting higher for more initial exploration
let tradeCount = 0; // To track performance for adaptive exploration

// Actions: Adjusted TP/SL multipliers and position sizes
const actions = [
  { type: 'buy', sizeMultiplier: 0.04, tpMultiplier: 5, slMultiplier: 0.7 }, // Conservative
  { type: 'buy', sizeMultiplier: 0.06, tpMultiplier: 7, slMultiplier: 0.9 }, // Moderate
  { type: 'buy', sizeMultiplier: 0.12, tpMultiplier: 10, slMultiplier: 1.7 }, // Aggressive
  { type: 'sell', sizeMultiplier: 0, tpMultiplier: 0, slMultiplier: 0 }, // Exit
];

// Delay function to avoid API rate limits
function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Load cached historical data
function loadCachedData() {
  try {
    if (fs.existsSync(CACHE_FILE)) {
      const data = fs.readFileSync(CACHE_FILE, 'utf8');
      console.log('[Cache] Loaded historical data from cache');
      return JSON.parse(data);
    }
    return null;
  } catch (error) {
    console.error('[Cache] Error loading cached data:', error.message);
    return null;
  }
}

// Save historical data to cache
function saveCachedData(data) {
  try {
    fs.writeFileSync(CACHE_FILE, JSON.stringify(data, null, 2));
    console.log('[Cache] Historical data saved to cache');
  } catch (error) {
    console.error('[Cache] Error saving cached data:', error.message);
  }
}

// Fetch historical price and volume data from Birdeye
async function fetchHistoricalData() {
  const cachedData = loadCachedData();
  if (cachedData) {
    return cachedData;
  }

  try {
    const endTime = Math.floor(Date.now() / 1000);
    const startTime = endTime - (DAYS_BACK * 24 * 60 * 60);
    let allData = [];
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
        const dataPoints = items.map(item => ({
          timestamp: item.unixTime,
          price: item.value,
          volume: item.value * (Math.random() * 1000 + 500), // Simulated volume
        }));
        allData = allData.concat(dataPoints);
      } else {
        console.error('[Backtest] API response unsuccessful:', response.data);
      }

      currentStartTime = chunkEndTime;
      await delay(2000);
    }

    allData.sort((a, b) => a.timestamp - b.timestamp);
    console.log(`[Backtest] Fetched ${allData.length} ${TIMEFRAME} data points`);
    
    saveCachedData(allData);
    return allData;
  } catch (error) {
    console.error('[Backtest] Error fetching historical data:', error.message);
    return [];
  }
}

// Calculate EMA
function calculateEMA(prices, period) {
  if (prices.length < period) return null;
  const k = 2 / (period + 1);
  let ema = prices[0];
  for (let i = 1; i < prices.length; i++) {
    ema = prices[i] * k + ema * (1 - k);
  }
  return ema;
}

// Calculate Volume MA
function calculateVolumeMA(volumes) {
  if (volumes.length < VOLUME_MA_PERIOD) return null;
  return math.mean(volumes.slice(-VOLUME_MA_PERIOD));
}

// Calculate ATR
function calculateATR(prices, period = ATR_PERIOD) {
  if (prices.length < period + 1) return 0;
  const tr = prices.slice(-period - 1).map((p, i) => {
    if (i === 0) return 0;
    return Math.max(p - prices[i - 1], Math.abs(p - prices[i - 1]));
  }).slice(1);
  return math.mean(tr);
}

// Calculate ADX (simplified)
function calculateADX(prices, period = ADX_PERIOD) {
  if (prices.length < period + 1) return 0;
  let plusDM = 0, minusDM = 0, trSum = 0;
  for (let i = 1; i < prices.length; i++) {
    const high = prices[i] > prices[i - 1] ? prices[i] - prices[i - 1] : 0;
    const low = prices[i - 1] > prices[i] ? prices[i - 1] - prices[i] : 0;
    plusDM += high > low ? high : 0;
    minusDM += low > high ? low : 0;
    trSum += Math.max(high, low);
  }
  const plusDI = (plusDM / trSum) * 100;
  const minusDI = (minusDM / trSum) * 100;
  const dx = trSum > 0 ? Math.abs(plusDI - minusDI) / (plusDI + minusDI) * 100 : 0;
  return dx;
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

// Q-learning: Get state (expanded with volatility)
function getState(trendStrength, volumeRatio, adx, hasPosition, atr, price) {
  const trendLevel = trendStrength > 0.02 ? 'strong' : trendStrength > 0 ? 'moderate' : 'weak';
  const volumeLevel = volumeRatio > 1.5 ? 'high' : volumeRatio > 1 ? 'normal' : 'low';
  const adxLevel = adx > 30 ? 'very_strong' : adx > 20 ? 'strong' : 'weak';
  const positionStatus = hasPosition ? 'holding' : 'flat';
  const volatilityLevel = (atr / price) > 0.02 ? 'high' : (atr / price) > 0.01 ? 'medium' : 'low';
  return `${trendLevel}_${volumeLevel}_${adxLevel}_${positionStatus}_${volatilityLevel}`;
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

// Q-learning: Update Q-table with balanced reward
function updateQTable(state, actionIndex, reward, nextState, holdingTime) {
  if (!qTable[nextState]) qTable[nextState] = Array(actions.length).fill(0);

  const currentQ = qTable[state][actionIndex];
  const maxFutureQ = Math.max(...qTable[nextState]);
  const newQ = currentQ + learningRate * (reward + discountFactor * maxFutureQ - currentQ);
  qTable[state][actionIndex] = newQ;

  // Adaptive exploration: Decay exponentially, increase slightly after losses
  explorationRate = Math.max(0.05, explorationRate * 0.995);
  if (reward < 0) explorationRate = Math.min(0.5, explorationRate + 0.01);
  tradeCount++;

  saveQTable();
}

// Log trades to CSV
function logBacktestTrade(timestamp, tradeType, price, solAmount, usdcAmount, profit) {
  const logEntry = `${new Date(timestamp * 1000).toISOString()},${timestamp},${tradeType},${price},${solAmount},${usdcAmount},${profit}\n`;
  const logFilePath = path.join(__dirname, 'backtest_trades_trend_volume_v2.csv');
  if (!fs.existsSync(logFilePath)) {
    fs.writeFileSync(logFilePath, 'timestamp,unixTimestamp,tradeType,price(SOL/USDC),solAmount,usdcAmount,profit(USDC)\n');
  }
  fs.appendFileSync(logFilePath, logEntry);
}

// Backtest function
async function backtest() {
  console.log('[Backtest] Starting trend-following with volume confirmation and ADX filter backtest...');

  const historicalData = await fetchHistoricalData();
  if (historicalData.length === 0) {
    console.error(`[Backtest] No ${TIMEFRAME} data available. Exiting.`);
    return;
  }

  let priceHistory = [];
  let volumeHistory = [];
  let usdcBalance = INITIAL_USDC_BALANCE;
  let solBalance = SIMULATED_SOL_BALANCE;
  let totalProfit = 0, wins = 0, trades = 0;
  let positions = [];
  let winProfits = [], lossProfits = [];

  for (const { timestamp, price, volume } of historicalData) {
    priceHistory.push(price);
    volumeHistory.push(volume);
    if (priceHistory.length > Math.max(EMA_LONG_PERIOD, EMA_SHORT_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD + 1, ADX_PERIOD + 1)) {
      priceHistory.shift();
      volumeHistory.shift();
    }

    const emaLong = calculateEMA(priceHistory, EMA_LONG_PERIOD);
    const emaShort = calculateEMA(priceHistory, EMA_SHORT_PERIOD);
    const volumeMA = calculateVolumeMA(volumeHistory);
    const atr = calculateATR(priceHistory);
    const adx = calculateADX(priceHistory);
    const trendStrength = emaShort && emaLong ? (emaShort - emaLong) / emaLong : 0;
    const volumeRatio = volumeMA ? volume / volumeMA : 0;

    if (!emaLong || !emaShort || !volumeMA) continue;

    const state = getState(trendStrength, volumeRatio, adx, positions.length > 0, atr, price);
    const action = getAction(state);

    // Manage open positions with trailing stop
    for (let i = positions.length - 1; i >= 0; i--) {
      const position = positions[i];
      const profit = (price - position.entryPrice) * position.solAmount;
      const holdingTime = (timestamp - position.entryTime) / (15 * 60); // In 15m intervals

      // Update trailing stop
      const trailingStop = position.highestPrice - 2 * position.atr;
      position.highestPrice = Math.max(position.highestPrice, price);

      if (price >= position.takeProfit || price <= position.stopLoss || price <= trailingStop || (emaShort < emaLong && profit > 0)) {
        const fee = position.tradeAmount * TRANSACTION_FEE;
        usdcBalance += position.tradeAmount + profit - fee;
        solBalance -= position.solAmount;
        totalProfit += profit - fee;
        trades++;
        if (profit > 0) {
          wins++;
          winProfits.push(profit - fee);
        } else {
          lossProfits.push(profit - fee);
        }
        logBacktestTrade(timestamp, 'SELL', price, position.solAmount, position.tradeAmount + profit - fee, profit - fee);

        // Balanced reward: Normalized by ATR, with time penalty
        const reward = (profit / (position.atr * position.solAmount)) - (holdingTime * 0.005); // ATR-normalized, small time penalty
        const nextState = getState(trendStrength, volumeRatio, adx, false, atr, price);
        const actionIndex = actions.findIndex(a => a.type === action.type && a.sizeMultiplier === action.sizeMultiplier);
        updateQTable(state, actionIndex, reward, nextState, holdingTime);

        positions.splice(i, 1);
      }
    }

    // Entry logic
    if (positions.length === 0 && action.type === 'buy' && emaShort > emaLong && volumeRatio > 1.2 && adx > MIN_ADX) {
      const tradeAmount = usdcBalance * action.sizeMultiplier;
      if (usdcBalance < tradeAmount || solBalance + (tradeAmount / price) > MAX_SOL_POSITION) continue;

      const solAmount = tradeAmount / price;
      const fee = tradeAmount * TRANSACTION_FEE;
      const takeProfit = price + action.tpMultiplier * atr;
      const stopLoss = price - action.slMultiplier * atr;

      usdcBalance -= (tradeAmount + fee);
      solBalance += solAmount;
      totalProfit -= fee;
      positions.push({
        entryPrice: price,
        takeProfit,
        stopLoss,
        solAmount,
        tradeAmount,
        atr,
        highestPrice: price,
        entryTime: timestamp, // Added for holding time calculation
      });
      logBacktestTrade(timestamp, 'BUY', price, solAmount, tradeAmount, -fee);
    }
  }

  // Liquidate remaining positions
  if (positions.length > 0) {
    const lastPrice = historicalData[historicalData.length - 1].price;
    for (const position of positions) {
      const profit = (lastPrice - position.entryPrice) * position.solAmount;
      const fee = position.tradeAmount * TRANSACTION_FEE;
      usdcBalance += position.tradeAmount + profit - fee;
      solBalance -= position.solAmount;
      totalProfit += profit - fee;
      trades++;
      if (profit > 0) {
        wins++;
        winProfits.push(profit - fee);
      } else {
        lossProfits.push(profit - fee);
      }
      logBacktestTrade(historicalData[historicalData.length - 1].timestamp, 'SELL (Liquidation)', lastPrice, position.solAmount, position.tradeAmount + profit - fee, profit - fee);
    }
  }

  const winRate = trades > 0 ? (wins / trades) * 100 : 0;
  const avgWin = winProfits.length > 0 ? math.mean(winProfits) : 0;
  const avgLoss = lossProfits.length > 0 ? math.mean(lossProfits) : 0;
  console.log(`[Backtest] Results:`);
  console.log(`- Total Profit: ${totalProfit.toFixed(2)} USDC`);
  console.log(`- Win Rate: ${winRate.toFixed(2)}%`);
  console.log(`- Total Trades: ${trades}`);
  console.log(`- Average Win: ${avgWin.toFixed(2)} USDC`);
  console.log(`- Average Loss: ${avgLoss.toFixed(2)} USDC`);
  console.log(`- Final USDC Balance: ${usdcBalance.toFixed(2)}`);
  console.log(`- Final SOL Balance: ${solBalance.toFixed(2)}`);
}

// Run backtest
backtest().catch(console.error);