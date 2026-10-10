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
const CACHE_FILE = path.join(__dirname, 'historical_data_cache.json');
const ADX_PERIOD = 14;
const ADX_THRESHOLD = 30;
const POSITION_SIZE_MULTIPLIER = 0.05;
const FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786];

// Delay function
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
    console.log('[Cache] No cache file found');
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

// Fetch historical data with cache priority
async function fetchHistoricalData() {
  const cachedData = loadCachedData();
  if (cachedData && cachedData.length > 0) {
    console.log(`[Backtest] Using cached data with ${cachedData.length} ${TIMEFRAME} data points`);
    return cachedData;
  }

  console.log('[Backtest] No valid cached data found, fetching from Birdeye...');
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

// Calculate ADX
function calculateADX(highs, lows, closes, period = ADX_PERIOD) {
  if (closes.length < period * 2) return 0;
  
  const dmPlus = [], dmMinus = [], tr = [];
  for (let i = 1; i < highs.length; i++) {
    const highDiff = highs[i] - highs[i-1];
    const lowDiff = lows[i-1] - lows[i];
    dmPlus.push(Math.max(highDiff, 0));
    dmMinus.push(Math.max(lowDiff, 0));
    tr.push(Math.max(highs[i] - lows[i], 
           Math.abs(highs[i] - closes[i-1]), 
           Math.abs(lows[i] - closes[i-1])));
  }

  const smoothPlus = math.mean(dmPlus.slice(-period));
  const smoothMinus = math.mean(dmMinus.slice(-period));
  const smoothTR = math.mean(tr.slice(-period));
  
  const diPlus = (smoothPlus / smoothTR) * 100;
  const diMinus = (smoothMinus / smoothTR) * 100;
  const dx = Math.abs(diPlus - diMinus) / (diPlus + diMinus) * 100;
  
  return dx;
}

// Calculate Fibonacci levels
function calculateFibLevels(high, low) {
  const diff = high - low;
  return FIB_LEVELS.map(level => ({
    level,
    price: high - (diff * level)
  }));
}

// Log trades to CSV
function logBacktestTrade(timestamp, tradeType, price, solAmount, usdcAmount, profit) {
  const logEntry = `${new Date(timestamp * 1000).toISOString()},${timestamp},${tradeType},${price},${solAmount},${usdcAmount},${profit}\n`;
  const logFilePath = path.join(__dirname, 'backtest_trades_swing.csv');
  if (!fs.existsSync(logFilePath)) {
    fs.writeFileSync(logFilePath, 'timestamp,unixTimestamp,tradeType,price(SOL/USDC),solAmount,usdcAmount,profit(USDC)\n');
  }
  fs.appendFileSync(logFilePath, logEntry);
}

async function backtest() {
  console.log('[Backtest] Starting trend-following swing trading backtest with cached data...');

  const historicalData = await fetchHistoricalData();
  if (historicalData.length === 0) {
    console.error(`[Backtest] No ${TIMEFRAME} data available. Exiting.`);
    return;
  }

  let priceHistory = [];
  let highHistory = [];
  let lowHistory = [];
  let usdcBalance = INITIAL_USDC_BALANCE;
  let solBalance = SIMULATED_SOL_BALANCE;
  let totalProfit = 0, wins = 0, trades = 0;
  let positions = [];
  let winProfits = [], lossProfits = [];
  let trendHigh = 0, trendLow = Infinity;

  for (const { timestamp, price, volume } of historicalData) {
    const high = price * (1 + Math.random() * 0.005);
    const low = price * (1 - Math.random() * 0.005);
    
    priceHistory.push(price);
    highHistory.push(high);
    lowHistory.push(low);
    
    if (priceHistory.length > ADX_PERIOD * 2) {
      priceHistory.shift();
      highHistory.shift();
      lowHistory.shift();
    }

    trendHigh = Math.max(trendHigh, high);
    trendLow = Math.min(trendLow, low);

    const adx = calculateADX(highHistory, lowHistory, priceHistory);
    const fibLevels = calculateFibLevels(trendHigh, trendLow);

    for (let i = positions.length - 1; i >= 0; i--) {
      const position = positions[i];
      const profit = (price - position.entryPrice) * position.solAmount;

      const exitFib = fibLevels.find(f => f.level === 0.618);
      if ((position.type === 'LONG' && price >= exitFib.price) || 
          (position.type === 'SHORT' && price <= exitFib.price) || 
          adx < ADX_THRESHOLD) {
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
        logBacktestTrade(timestamp, `${position.type === 'LONG' ? 'SELL' : 'BUY'}`, price, position.solAmount, position.tradeAmount + profit - fee, profit - fee);
        positions.splice(i, 1);
        if (position.type === 'LONG') trendHigh = price;
        else trendLow = price;
      }
    }

    if (positions.length === 0 && adx > ADX_THRESHOLD) {
      const tradeAmount = usdcBalance * POSITION_SIZE_MULTIPLIER;
      if (usdcBalance < tradeAmount || solBalance + (tradeAmount / price) > MAX_SOL_POSITION) continue;

      const solAmount = tradeAmount / price;
      const fee = tradeAmount * TRANSACTION_FEE;
      const entryFib = fibLevels.find(f => f.level === 0.382);

      const isUptrend = highHistory[highHistory.length-1] > highHistory[0];
      
      if (isUptrend && price <= entryFib.price) {
        usdcBalance -= (tradeAmount + fee);
        solBalance += solAmount;
        totalProfit -= fee;
        positions.push({
          entryPrice: price,
          tradeAmount,
          solAmount,
          type: 'LONG'
        });
        logBacktestTrade(timestamp, 'BUY', price, solAmount, tradeAmount, -fee);
      } else if (!isUptrend && price >= entryFib.price) {
        usdcBalance -= (tradeAmount + fee);
        solBalance += solAmount;
        totalProfit -= fee;
        positions.push({
          entryPrice: price,
          tradeAmount,
          solAmount,
          type: 'SHORT'
        });
        logBacktestTrade(timestamp, 'SHORT', price, solAmount, tradeAmount, -fee);
      }
    }
  }

  if (positions.length > 0) {
    const lastPrice = historicalData[historicalData.length - 1].price;
    for (const position of positions) {
      const profit = position.type === 'LONG' ? 
        (lastPrice - position.entryPrice) * position.solAmount :
        (position.entryPrice - lastPrice) * position.solAmount;
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
      logBacktestTrade(historicalData[historicalData.length - 1].timestamp, 
        `${position.type === 'LONG' ? 'SELL' : 'BUY'} (Liquidation)`, 
        lastPrice, position.solAmount, position.tradeAmount + profit - fee, profit - fee);
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

backtest().catch(console.error);