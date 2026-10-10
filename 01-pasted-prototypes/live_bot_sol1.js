require('dotenv').config();
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const math = require('mathjs');
const { Connection, Keypair, VersionedTransaction, PublicKey } = require('@solana/web3.js');
const { createJupiterApiClient } = require('@jup-ag/api');
const { getAssociatedTokenAddress, getAccount, TOKEN_PROGRAM_ID } = require('@solana/spl-token');
const bs58 = require('bs58');

// Constants
const SOL_MINT = 'So11111111111111111111111111111111111111112';
const USDC_MINT = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v';
const INITIAL_USDC_BALANCE = 500;
const MAX_SOL_POSITION = 5;
const TRANSACTION_FEE = 0.001;
const QTABLE_FILE = path.join(__dirname, 'qtable_trend_volume_v2.json');
const EMA_LONG_PERIOD = 50;
const EMA_SHORT_PERIOD = 20;
const VOLUME_MA_PERIOD = 20;
const ATR_PERIOD = 14;
const ADX_PERIOD = 14;
const MIN_ADX = 35;
const TRADE_COOLDOWN = 10 * 1000; // 10 seconds in milliseconds

// Q-learning variables
let qTable = loadQTable();
let learningRate = 0.05;
let discountFactor = 0.9;
let explorationRate = 0.3; // Reset for live market adaptation
let tradeCount = 0;

// Actions
const actions = [
  { type: 'buy', sizeMultiplier: 0.04, tpMultiplier: 5, slMultiplier: 0.7 },
  { type: 'buy', sizeMultiplier: 0.06, tpMultiplier: 7, slMultiplier: 0.9 },
  { type: 'buy', sizeMultiplier: 0.12, tpMultiplier: 10, slMultiplier: 1.7 },
  { type: 'sell', sizeMultiplier: 0, tpMultiplier: 0, slMultiplier: 0 },
];

// Solana connection and wallet setup
const connection = new Connection(process.env.RPC_ENDPOINT, 'confirmed');
const wallet = (() => {
  try {
    const privateKey = process.env.PHANTOM_PRIVATE_KEY;
    if (!privateKey) throw new Error('PHANTOM_PRIVATE_KEY not set');
    console.log('Private key (first 5 chars):', privateKey.substring(0, 5)); // Debug
    const decodedKey = bs58.decode(privateKey);
    const keypair = Keypair.fromSecretKey(decodedKey);
    console.log('[Live Bot] Wallet initialized. Public Key:', keypair.publicKey.toBase58());
    return keypair;
  } catch (error) {
    console.error('[Live Bot] Error initializing wallet:', error.message);
    process.exit(1);
  }
})();

// Jupiter client
const jupiter = createJupiterApiClient({ basePath: 'https://quote-api.jup.ag/v6' });

// State variables
let priceHistory = [];
let volumeHistory = [];
let usdcBalance = INITIAL_USDC_BALANCE;
let solBalance = 0;
let positions = [];
let lastTradeTime = 0;

// Fetch token balance (Updated to handle native SOL)
async function getTokenBalance(mintAddress) {
  try {
    if (mintAddress === SOL_MINT) {
      // Fetch native SOL balance
      const nativeBalance = await connection.getBalance(wallet.publicKey);
      const solBalance = nativeBalance / 1e9; // Convert lamports to SOL
      return solBalance;
    } else {
      // For SPL tokens like USDC
      const mintPublicKey = new PublicKey(mintAddress);
      const associatedTokenAddress = await getAssociatedTokenAddress(
        mintPublicKey,
        wallet.publicKey,
        false,
        TOKEN_PROGRAM_ID
      );
      const accountInfo = await getAccount(connection, associatedTokenAddress);
      return Number(accountInfo.amount) / 1e6; // USDC uses 6 decimals
    }
  } catch (error) {
    if (error.name === 'TokenAccountNotFoundError' || error.message.includes('Account not found')) {
      console.log(`[Live Bot] No token account found for ${mintAddress}, returning 0`);
      return 0;
    }
    console.error(`[Live Bot] Error fetching balance for ${mintAddress}:`, error.message);
    return 0;
  }
}

// Fetch live price and volume from Birdeye
async function fetchLiveData() {
  try {
    console.log('[Live Bot] Fetching live data from Birdeye...');
    const response = await axios.get(
      `https://public-api.birdeye.so/defi/price?address=${SOL_MINT}`,
      { headers: { 'X-API-KEY': process.env.BIRDEYE_API_KEY } }
    );
    if (response.data.success) {
      const price = response.data.data.value;
      const volumeResponse = await axios.get(
        `https://public-api.birdeye.so/defi/token_overview?address=${SOL_MINT}`,
        { headers: { 'X-API-KEY': process.env.BIRDEYE_API_KEY } }
      );
      const volume = volumeResponse.data.success ? (volumeResponse.data.data.v24hUSD || 0) / 144 : 0; // 15-min estimate
      console.log(`[Live Bot] Fetched data - Price: ${price}, Volume (15m est.): ${volume}`);
      return { timestamp: Math.floor(Date.now() / 1000), price, volume };
    }
    console.error('[Live Bot] Invalid price response:', response.data);
    return null;
  } catch (error) {
    console.error('[Live Bot] Error fetching live data:', error.message);
    return null;
  }
}

// Calculate EMA
function calculateEMA(prices, period) {
  if (prices.length < period) {
    console.log(`[Live Bot] Not enough data for EMA (${period}): ${prices.length}/${period} data points`);
    return null;
  }
  const k = 2 / (period + 1);
  let ema = prices[0];
  for (let i = 1; i < prices.length; i++) {
    ema = prices[i] * k + ema * (1 - k);
  }
  return ema;
}

// Calculate Volume MA
function calculateVolumeMA(volumes) {
  if (volumes.length < VOLUME_MA_PERIOD) {
    console.log(`[Live Bot] Not enough data for Volume MA: ${volumes.length}/${VOLUME_MA_PERIOD} data points`);
    return null;
  }
  return math.mean(volumes.slice(-VOLUME_MA_PERIOD));
}

// Calculate ATR
function calculateATR(prices) {
  if (prices.length < ATR_PERIOD + 1) {
    console.log(`[Live Bot] Not enough data for ATR: ${prices.length}/${ATR_PERIOD + 1} data points`);
    return 0;
  }
  const tr = prices.slice(-ATR_PERIOD - 1).map((p, i) => {
    if (i === 0) return 0;
    return Math.max(p - prices[i - 1], Math.abs(p - prices[i - 1]));
  }).slice(1);
  return math.mean(tr);
}

// Calculate ADX (simplified)
function calculateADX(prices) {
  if (prices.length < ADX_PERIOD + 1) {
    console.log(`[Live Bot] Not enough data for ADX: ${prices.length}/${ADX_PERIOD + 1} data points`);
    return 0;
  }
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
      console.log('[Live Bot] Q-table loaded');
      return JSON.parse(data);
    }
    console.log('[Live Bot] Q-table not found, starting fresh');
    return {};
  } catch (error) {
    console.error('[Live Bot] Error loading Q-table:', error.message);
    return {};
  }
}

// Save Q-table
function saveQTable() {
  try {
    fs.writeFileSync(QTABLE_FILE, JSON.stringify(qTable, null, 2));
  } catch (error) {
    console.error('[Live Bot] Error saving Q-table:', error.message);
  }
}

// Get state
function getState(trendStrength, volumeRatio, adx, hasPosition, atr, price) {
  const trendLevel = trendStrength > 0.02 ? 'strong' : trendStrength > 0 ? 'moderate' : 'weak';
  const volumeLevel = volumeRatio > 1.5 ? 'high' : volumeRatio > 1 ? 'normal' : 'low';
  const adxLevel = adx > 30 ? 'very_strong' : adx > 20 ? 'strong' : 'weak';
  const positionStatus = hasPosition ? 'holding' : 'flat';
  const volatilityLevel = (atr / price) > 0.02 ? 'high' : (atr / price) > 0.01 ? 'medium' : 'low';
  return `${trendLevel}_${volumeLevel}_${adxLevel}_${positionStatus}_${volatilityLevel}`;
}

// Get action
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

// Update Q-table
function updateQTable(state, actionIndex, reward, nextState, holdingTime) {
  if (!qTable[nextState]) qTable[nextState] = Array(actions.length).fill(0);
  const currentQ = qTable[state][actionIndex];
  const maxFutureQ = Math.max(...qTable[nextState]);
  const newQ = currentQ + learningRate * (reward + discountFactor * maxFutureQ - currentQ);
  qTable[state][actionIndex] = newQ;
  explorationRate = Math.max(0.05, explorationRate * 0.995);
  if (reward < 0) explorationRate = Math.min(0.5, explorationRate + 0.01);
  tradeCount++;
  saveQTable();
}

// Log trades to CSV
function logTrade(timestamp, tradeType, price, solAmount, usdcAmount, profit) {
  const logEntry = `${new Date(timestamp * 1000).toISOString()},${timestamp},${tradeType},${price},${solAmount},${usdcAmount},${profit}\n`;
  const logFilePath = path.join(__dirname, 'live_trades.csv');
  if (!fs.existsSync(logFilePath)) {
    fs.writeFileSync(logFilePath, 'timestamp,unixTimestamp,tradeType,price(SOL/USDC),solAmount,usdcAmount,profit(USDC)\n');
  }
  fs.appendFileSync(logFilePath, logEntry);
  console.log(`[Live Bot] Logged trade: ${tradeType}, Price: ${price}, SOL: ${solAmount}, USDC: ${usdcAmount}, Profit: ${profit}`);
}

// Execute swap via Jupiter
async function executeSwap(inputMint, outputMint, amount, tradeType, price) {
  try {
    console.log(`[Live Bot] Executing swap: ${tradeType}, Amount: ${amount}`);
    const amountInLamports = Math.floor(amount * (inputMint === SOL_MINT ? 1e9 : 1e6));
    const quoteResponse = await jupiter.quoteGet({
      inputMint,
      outputMint,
      amount: amountInLamports,
      slippageBps: 50,
    });
    const swapResponse = await jupiter.swapPost({
      swapRequest: {
        quoteResponse,
        userPublicKey: wallet.publicKey.toBase58(),
        wrapAndUnwrapSol: true,
      },
    });
    const transactionBuf = Buffer.from(swapResponse.swapTransaction, 'base64');
    const transaction = VersionedTransaction.deserialize(transactionBuf);
    transaction.sign([wallet]);
    const signature = await connection.sendTransaction(transaction, {
      skipPreflight: false,
      preflightCommitment: 'confirmed',
    });
    await connection.confirmTransaction({
      signature,
      blockhash: transaction.message.recentBlockhash,
      lastValidBlockHeight: swapResponse.lastValidBlockHeight,
    }, 'confirmed');
    console.log(`[Live Bot] Swap executed: ${signature}`);
    const solAmount = inputMint === SOL_MINT ? amount : Number(quoteResponse.outAmount) / 1e9;
    const usdcAmount = inputMint === USDC_MINT ? amount : Number(quoteResponse.outAmount) / 1e6;
    logTrade(Math.floor(Date.now() / 1000), tradeType, price, solAmount, usdcAmount, tradeType === 'BUY' ? -amount : profit);
    return { success: true, solAmount };
  } catch (error) {
    console.error('[Live Bot] Swap error:', error.message);
    return { success: false, solAmount: 0 };
  }
}

// Main trading logic
async function trade() {
  try {
    console.log('[Live Bot] Starting trade cycle...');
    const now = Date.now();
    if (now - lastTradeTime < TRADE_COOLDOWN) {
      console.log(`[Live Bot] In cooldown period, skipping trade. Time since last trade: ${(now - lastTradeTime) / 1000} seconds`);
      return;
    }

    const data = await fetchLiveData();
    if (!data) {
      console.log('[Live Bot] No data fetched, skipping trade cycle');
      return;
    }

    const { timestamp, price, volume } = data;
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
    if (!emaLong || !emaShort || !volumeMA) {
      console.log('[Live Bot] Insufficient data for indicators, skipping trade');
      return;
    }

    usdcBalance = await getTokenBalance(USDC_MINT);
    solBalance = await getTokenBalance(SOL_MINT);
    console.log(`[Live Bot] Balances - USDC: ${usdcBalance.toFixed(2)}, SOL: ${solBalance.toFixed(6)}`);

    const trendStrength = (emaShort - emaLong) / emaLong;
    const volumeRatio = volume / volumeMA;
    const state = getState(trendStrength, volumeRatio, adx, positions.length > 0, atr, price);
    const action = getAction(state);
    console.log(`[Live Bot] State: ${state}, Action: ${action.type}, Trend Strength: ${trendStrength}, Volume Ratio: ${volumeRatio}, ADX: ${adx}`);

    // Manage positions
    for (let i = positions.length - 1; i >= 0; i--) {
      const position = positions[i];
      const profit = (price - position.entryPrice) * position.solAmount;
      const holdingTime = (timestamp - position.entryTime) / (15 * 60);
      const trailingStop = position.highestPrice - 2 * position.atr;
      position.highestPrice = Math.max(position.highestPrice, price);

      if (price >= position.takeProfit || price <= position.stopLoss || price <= trailingStop || (emaShort < emaLong && profit > 0)) {
        const fee = position.tradeAmount * TRANSACTION_FEE;
        const { success } = await executeSwap(SOL_MINT, USDC_MINT, position.solAmount, 'SELL', price);
        if (success) {
          usdcBalance += position.tradeAmount + profit - fee;
          solBalance -= position.solAmount;
          const reward = (profit / (position.atr * position.solAmount)) - (holdingTime * 0.005);
          const nextState = getState(trendStrength, volumeRatio, adx, false, atr, price);
          const actionIndex = actions.findIndex(a => a.type === action.type && a.sizeMultiplier === action.sizeMultiplier);
          updateQTable(state, actionIndex, reward, nextState, holdingTime);
          positions.splice(i, 1);
          console.log('[Live Bot] Position closed');
        }
      }
    }

    // Entry logic
    if (positions.length === 0 && action.type === 'buy' && emaShort > emaLong && volumeRatio > 1.2 && adx > MIN_ADX) {
      const tradeAmount = usdcBalance * action.sizeMultiplier;
      if (usdcBalance < tradeAmount || solBalance + (tradeAmount / price) > MAX_SOL_POSITION) {
        console.log('[Live Bot] Insufficient balance or max position reached, skipping buy');
        return;
      }

      const solAmount = tradeAmount / price;
      const fee = tradeAmount * TRANSACTION_FEE;
      const takeProfit = price + action.tpMultiplier * atr;
      const stopLoss = price - action.slMultiplier * atr;

      const { success, solAmount: executedSol } = await executeSwap(USDC_MINT, SOL_MINT, tradeAmount, 'BUY', price);
      if (success) {
        usdcBalance -= (tradeAmount + fee);
        solBalance += executedSol;
        positions.push({
          entryPrice: price,
          takeProfit,
          stopLoss,
          solAmount: executedSol,
          tradeAmount,
          atr,
          highestPrice: price,
          entryTime: timestamp,
        });
        lastTradeTime = now;
        console.log('[Live Bot] New position opened');
      }
    } else {
      console.log('[Live Bot] No trade executed - Conditions not met');
    }
  } catch (error) {
    console.error('[Live Bot] Error in trade cycle:', error.message);
  }
}

// Run bot
async function startBot() {
  console.log('[Live Bot] Starting SOL/USDC trading bot...');
  while (true) {
    await trade();
    console.log('[Live Bot] Waiting 60 seconds for next cycle...');
    await new Promise(resolve => setTimeout(resolve, 60000)); 
  }
}

startBot().catch(error => {
  console.error('[Live Bot] Fatal error in startBot:', error.message);
  process.exit(1);
});