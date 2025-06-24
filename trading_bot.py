#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Cryptocurrency Trading Bot
- Uses OKX API for market data and trading
- Implements Whalemap indicator to detect large buy/sell orders
- Sends notifications via Telegram
- Provides entry/exit points and take profit levels
"""

import os
import time
import json
import logging
import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union

import ccxt
import pandas as pd
import numpy as np
import pytz
from dotenv import load_dotenv
import requests
import telegram
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configure logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# OKX API credentials
OKX_API_KEY = os.getenv('OKX_API_KEY')
OKX_SECRET_KEY = os.getenv('OKX_SECRET_KEY')
OKX_PASSPHRASE = os.getenv('OKX_PASSPHRASE')

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Trading configuration
TRADING_MODE = os.getenv('TRADING_MODE', 'spot')
RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', 1))
TAKE_PROFIT_PERCENTAGE = float(os.getenv('TAKE_PROFIT_PERCENTAGE', 3))
STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', 2))

# Whalemap indicator settings
WHALE_ORDER_THRESHOLD = float(os.getenv('WHALE_ORDER_THRESHOLD', 100000))
WHALE_CONFIRMATION_COUNT = int(os.getenv('WHALE_CONFIRMATION_COUNT', 3))

# List of coins to monitor
COINS = [
    'ARB', 'TRB', 'EIGEN', 'MERL', 'COMP', 'TIA', 'API3', 'TRUMP',
    'KAIA', 'HBAR', 'MOVR', 'DOGE', 'RPL', 'LINK',
    'BNB', 'XRP', 'DOT',
    'GALA', 'CAT', 'LDO', 'BCH', 'AWE', 'WAL', 'DOGS',
    'ATOM', 'GOAT', 'GRIFFAIN', 'GRASS', 'FLOCK', 'ANIME', 'PENGU', 'TRX', 'MASK', 'LA',
    'PEPE', 'NEROETH', 'TON', 'FORTH', 'ARPA', 'PHA', 'MÉTIS', 'BAL', 'INJ', 'YFI',
    'NEAR', 'VIRTUAL', 'PINUT', 'SHIB', 'AVAX', 'OP', 'WIF', 'ONDO', 'LTC', 'SNT',
    'A8', 'MOODENG', 'MILK', 'OBOL', 'NKN', 'FB', 'FWOG', 'ZBCN', 'STRK', 'ENA',
    'PENDLE', 'CRV', 'ADA', 'REX', 'FLM', 'AVAAI', 'WLD', 'ORDI', 'CATI', 'OL',
    'BADGER', 'BOBA', 'SOLAYER', 'LISTA', 'PUNDIX', 'DOOD', 'UMA', 'UNI', 'S', 'SUI',
    'APE', 'PYTH', 'JELLYJELLY', 'ETC', 'CTV', 'GLM', 'RENDER', 'NOT', 'ENS', 'RATS',
    'MEME', 'SAND', 'XCN', 'PEOPLE', '1INCH', 'RVN', 'MAVIA', 'DEEP', 'XMR', 'MDT',
    'QNT', 'VINE', 'JASMY', 'BOME', 'BANK', 'DOG'
]

# Technical analysis parameters
EMA_SHORT = 9
EMA_MEDIUM = 21
EMA_LONG = 50
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2


class TradingBot:
    def __init__(self):
        """
        Initialize the trading bot with OKX exchange connection and Telegram bot
        """
        # Initialize OKX exchange
        self.exchange = ccxt.okx({
            'apiKey': OKX_API_KEY,
            'secret': OKX_SECRET_KEY,
            'password': OKX_PASSPHRASE,
            'enableRateLimit': True,
        })
        
        # Initialize Telegram bot
        self.telegram_bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        # Initialize data structures
        self.market_data = {}
        self.whale_orders = {}
        self.active_trades = {}
        self.trade_history = []
        
        # Initialize timeframes to analyze
        self.timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']
        
        logger.info("Trading bot initialized successfully")
    
    async def send_telegram_message(self, message: str) -> None:
        """
        Send a message to the configured Telegram chat
        
        Args:
            message: The message text to send
        """
        try:
            await self.telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')
            logger.info(f"Telegram message sent: {message[:50]}...")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    def send_telegram_chart(self, symbol: str, timeframe: str, indicators: List[str] = None) -> None:
        """
        Generate and send a chart image to Telegram
        
        Args:
            symbol: The trading pair symbol
            timeframe: The timeframe for the chart
            indicators: List of indicators to include on the chart
        """
        # This would require matplotlib to generate charts
        # For now, we'll just send a message that this feature is not implemented
        message = f"📊 Chart generation for {symbol} ({timeframe}) is not implemented in this version."
        self.send_telegram_message(message)
    
    def fetch_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol and timeframe
        
        Args:
            symbol: The trading pair symbol
            timeframe: The timeframe for the data
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Format symbol for OKX (e.g., 'BTC/USDT')
            formatted_symbol = f"{symbol}/USDT"
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(formatted_symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Store in market_data
            if symbol not in self.market_data:
                self.market_data[symbol] = {}
            self.market_data[symbol][timeframe] = df
            
            logger.info(f"Fetched market data for {formatted_symbol} ({timeframe})")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate EMAs
        df['ema_short'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
        df['ema_medium'] = df['close'].ewm(span=EMA_MEDIUM, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=RSI_PERIOD).mean()
        avg_loss = loss.rolling(window=RSI_PERIOD).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        df['ema12'] = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate Bollinger Bands
        df['sma'] = df['close'].rolling(window=BOLLINGER_PERIOD).mean()
        df['std'] = df['close'].rolling(window=BOLLINGER_PERIOD).std()
        df['upper_band'] = df['sma'] + (df['std'] * BOLLINGER_STD)
        df['lower_band'] = df['sma'] - (df['std'] * BOLLINGER_STD)
        
        # Calculate volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def detect_whale_orders(self, symbol: str) -> List[Dict]:
        """
        Detect large (whale) orders for a symbol using order book data
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            List of detected whale orders
        """
        try:
            formatted_symbol = f"{symbol}/USDT"
            order_book = self.exchange.fetch_order_book(formatted_symbol, limit=100)
            
            whale_orders = []
            
            # Check for large buy orders
            for price, volume in order_book['bids']:
                order_value = price * volume
                if order_value >= WHALE_ORDER_THRESHOLD:
                    whale_orders.append({
                        'symbol': symbol,
                        'type': 'buy',
                        'price': price,
                        'volume': volume,
                        'value': order_value,
                        'timestamp': datetime.datetime.now(pytz.UTC)
                    })
            
            # Check for large sell orders
            for price, volume in order_book['asks']:
                order_value = price * volume
                if order_value >= WHALE_ORDER_THRESHOLD:
                    whale_orders.append({
                        'symbol': symbol,
                        'type': 'sell',
                        'price': price,
                        'volume': volume,
                        'value': order_value,
                        'timestamp': datetime.datetime.now(pytz.UTC)
                    })
            
            # Store whale orders
            if symbol not in self.whale_orders:
                self.whale_orders[symbol] = []
            
            self.whale_orders[symbol].extend(whale_orders)
            
            # Keep only recent orders (last 24 hours)
            cutoff_time = datetime.datetime.now(pytz.UTC) - datetime.timedelta(hours=24)
            self.whale_orders[symbol] = [order for order in self.whale_orders[symbol] 
                                        if order['timestamp'] > cutoff_time]
            
            if whale_orders:
                logger.info(f"Detected {len(whale_orders)} whale orders for {symbol}")
            
            return whale_orders
        
        except Exception as e:
            logger.error(f"Error detecting whale orders for {symbol}: {e}")
            return []
    
    def analyze_whale_activity(self, symbol: str) -> Dict:
        """
        Analyze whale activity to determine if there's a potential bottom or top formation
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Dictionary with analysis results
        """
        if symbol not in self.whale_orders or not self.whale_orders[symbol]:
            return {'symbol': symbol, 'whale_signal': None}
        
        # Count recent buy and sell whale orders
        recent_orders = self.whale_orders[symbol]
        buy_orders = [order for order in recent_orders if order['type'] == 'buy']
        sell_orders = [order for order in recent_orders if order['type'] == 'sell']
        
        buy_value = sum(order['value'] for order in buy_orders)
        sell_value = sum(order['value'] for order in sell_orders)
        
        # Determine if there's a whale signal
        whale_signal = None
        signal_strength = 0
        
        if len(buy_orders) >= WHALE_CONFIRMATION_COUNT and buy_value > sell_value * 2:
            whale_signal = 'buy'
            signal_strength = min(10, len(buy_orders) / WHALE_CONFIRMATION_COUNT * 5)
        elif len(sell_orders) >= WHALE_CONFIRMATION_COUNT and sell_value > buy_value * 2:
            whale_signal = 'sell'
            signal_strength = min(10, len(sell_orders) / WHALE_CONFIRMATION_COUNT * 5)
        
        return {
            'symbol': symbol,
            'whale_signal': whale_signal,
            'signal_strength': signal_strength,
            'buy_orders_count': len(buy_orders),
            'sell_orders_count': len(sell_orders),
            'buy_value': buy_value,
            'sell_value': sell_value
        }
    
    def analyze_market(self, symbol: str) -> Dict:
        """
        Perform comprehensive market analysis for a symbol
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.datetime.now(pytz.UTC),
            'signals': {},
            'entry_points': [],
            'exit_points': [],
            'take_profit_levels': [],
            'stop_loss_levels': []
        }
        
        # Fetch and analyze data for different timeframes
        for timeframe in self.timeframes:
            df = self.fetch_market_data(symbol, timeframe)
            if df.empty:
                continue
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Store the latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None
            
            # Determine trend
            trend = 'neutral'
            if latest['ema_short'] > latest['ema_long'] and latest['close'] > latest['ema_medium']:
                trend = 'bullish'
            elif latest['ema_short'] < latest['ema_long'] and latest['close'] < latest['ema_medium']:
                trend = 'bearish'
            
            # Determine signals
            signals = []
            
            # EMA crossover signals
            if prev is not None:
                if prev['ema_short'] <= prev['ema_medium'] and latest['ema_short'] > latest['ema_medium']:
                    signals.append('ema_golden_cross')
                elif prev['ema_short'] >= prev['ema_medium'] and latest['ema_short'] < latest['ema_medium']:
                    signals.append('ema_death_cross')
            
            # RSI signals
            if latest['rsi'] < RSI_OVERSOLD:
                signals.append('rsi_oversold')
            elif latest['rsi'] > RSI_OVERBOUGHT:
                signals.append('rsi_overbought')
            
            # MACD signals
            if prev is not None:
                if prev['macd_hist'] <= 0 and latest['macd_hist'] > 0:
                    signals.append('macd_bullish_crossover')
                elif prev['macd_hist'] >= 0 and latest['macd_hist'] < 0:
                    signals.append('macd_bearish_crossover')
            
            # Bollinger Band signals
            if latest['close'] < latest['lower_band']:
                signals.append('bb_oversold')
            elif latest['close'] > latest['upper_band']:
                signals.append('bb_overbought')
            
            # Volume signals
            if latest['volume_ratio'] > 2.0:
                signals.append('high_volume')
            
            # Calculate signal strength for this timeframe
            signal_strength = self._calculate_signal_confidence(signals, trend)
            
            # Store signals for this timeframe
            analysis['signals'][timeframe] = {
                'trend': trend,
                'signals': signals,
                'strength': signal_strength,  # Add strength indicator
                'close': latest['close'],
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'macd_signal': latest['macd_signal'],
                'macd_hist': latest['macd_hist'],
                'upper_band': latest['upper_band'],
                'lower_band': latest['lower_band'],
                'volume_ratio': latest['volume_ratio']
            }
            
            # Determine potential entry points
            if trend == 'bullish' and ('rsi_oversold' in signals or 'bb_oversold' in signals or 'macd_bullish_crossover' in signals):
                analysis['entry_points'].append({
                    'timeframe': timeframe,
                    'price': latest['close'],
                    'type': 'buy',
                    'confidence': self._calculate_signal_confidence(signals, trend)
                })
            elif trend == 'bearish' and ('rsi_overbought' in signals or 'bb_overbought' in signals or 'macd_bearish_crossover' in signals):
                analysis['entry_points'].append({
                    'timeframe': timeframe,
                    'price': latest['close'],
                    'type': 'sell',
                    'confidence': self._calculate_signal_confidence(signals, trend)
                })
            
            # Determine potential exit points
            if trend == 'bullish' and ('rsi_overbought' in signals or 'bb_overbought' in signals):
                analysis['exit_points'].append({
                    'timeframe': timeframe,
                    'price': latest['close'],
                    'type': 'take_profit',
                    'confidence': self._calculate_signal_confidence(signals, trend)
                })
            elif trend == 'bearish' and ('rsi_oversold' in signals or 'bb_oversold' in signals):
                analysis['exit_points'].append({
                    'timeframe': timeframe,
                    'price': latest['close'],
                    'type': 'stop_loss',
                    'confidence': self._calculate_signal_confidence(signals, trend)
                })
            
            # Calculate take profit levels
            if trend == 'bullish':
                analysis['take_profit_levels'].append({
                    'timeframe': timeframe,
                    'price': latest['close'] * (1 + TAKE_PROFIT_PERCENTAGE / 100),
                    'percentage': TAKE_PROFIT_PERCENTAGE
                })
            
            # Calculate stop loss levels
            if trend == 'bullish':
                analysis['stop_loss_levels'].append({
                    'timeframe': timeframe,
                    'price': latest['close'] * (1 - STOP_LOSS_PERCENTAGE / 100),
                    'percentage': STOP_LOSS_PERCENTAGE
                })
        
        # Detect and analyze whale activity
        whale_analysis = self.analyze_whale_activity(symbol)
        analysis['whale_analysis'] = whale_analysis
        
        # Combine signals from different timeframes to get overall recommendation
        analysis['recommendation'] = self._generate_recommendation(analysis)
        
        return analysis
    
    def _calculate_signal_confidence(self, signals: List[str], trend: str) -> float:
        """
        Calculate confidence score for a set of signals
        
        Args:
            signals: List of detected signals
            trend: Current market trend
            
        Returns:
            Confidence score (0-10)
        """
        confidence = 5.0  # Start with neutral confidence
        
        # Adjust based on number of signals
        confidence += min(len(signals), 5)
        
        # Adjust based on signal types
        if 'ema_golden_cross' in signals:
            confidence += 1
        if 'ema_death_cross' in signals:
            confidence -= 1
        if 'macd_bullish_crossover' in signals:
            confidence += 1
        if 'macd_bearish_crossover' in signals:
            confidence -= 1
        if 'rsi_oversold' in signals and trend == 'bullish':
            confidence += 1
        if 'rsi_overbought' in signals and trend == 'bearish':
            confidence += 1
        if 'high_volume' in signals:
            confidence += 0.5
        
        # Ensure confidence is within range
        return max(0, min(10, confidence))
    
    def _generate_recommendation(self, analysis: Dict) -> Dict:
        """
        Generate overall trading recommendation based on analysis
        
        Args:
            analysis: Market analysis results
            
        Returns:
            Dictionary with recommendation details
        """
        # Count bullish and bearish signals across timeframes with strength consideration
        bullish_signals = 0
        bearish_signals = 0
        bullish_timeframes = []
        bearish_timeframes = []
        weighted_bullish_strength = 0
        weighted_bearish_strength = 0
        
        # Timeframe weights - higher weight for longer timeframes
        timeframe_weights = {
            '1m': 0.2,
            '3m': 0.3,
            '5m': 0.4,
            '15m': 0.6,
            '30m': 0.8,
            '1h': 1.0,
            '2h': 1.2,
            '4h': 1.5,
            '1d': 2.0
        }
        
        for timeframe, data in analysis['signals'].items():
            weight = timeframe_weights.get(timeframe, 1.0)
            strength = data.get('strength', 5.0)
            
            if data['trend'] == 'bullish':
                bullish_signals += 1
                bullish_timeframes.append(timeframe)
                weighted_bullish_strength += strength * weight
            elif data['trend'] == 'bearish':
                bearish_signals += 1
                bearish_timeframes.append(timeframe)
                weighted_bearish_strength += strength * weight
        
        # Consider whale analysis
        whale_signal = analysis['whale_analysis'].get('whale_signal')
        whale_strength = analysis['whale_analysis'].get('signal_strength', 0)
        
        if whale_signal == 'buy':
            bullish_signals += 1
            weighted_bullish_strength += whale_strength
        elif whale_signal == 'sell':
            bearish_signals += 1
            weighted_bearish_strength += whale_strength
        
        # Determine action
        action = 'hold'
        confidence = 5.0
        
        # Calculate total weighted strength
        total_bullish = weighted_bullish_strength / max(1, len(bullish_timeframes))
        total_bearish = weighted_bearish_strength / max(1, len(bearish_timeframes))
        
        if bullish_signals > bearish_signals and total_bullish > 0:
            action = 'buy'
            confidence = min(10, total_bullish)
        elif bearish_signals > bullish_signals and total_bearish > 0:
            action = 'sell'
            confidence = min(10, total_bearish)
        
        # Adjust confidence based on timeframe alignment
        if '1d' in bullish_timeframes and '4h' in bullish_timeframes and '1h' in bullish_timeframes and action == 'buy':
            confidence = min(10, confidence + 1)
        if '1d' in bearish_timeframes and '4h' in bearish_timeframes and '1h' in bearish_timeframes and action == 'sell':
            confidence = min(10, confidence + 1)
        
        # Determine entry point
        entry_point = None
        if action != 'hold' and analysis['entry_points']:
            # Sort by confidence and choose the highest
            entry_points = sorted(analysis['entry_points'], key=lambda x: x['confidence'], reverse=True)
            entry_point = entry_points[0]['price']
        
        # Determine exit strategy
        exit_strategy = {}
        if action == 'buy' and analysis['take_profit_levels']:
            # Use the 1h timeframe take profit level if available
            for tp in analysis['take_profit_levels']:
                if tp['timeframe'] == '1h':
                    exit_strategy['take_profit'] = tp['price']
                    break
            else:
                exit_strategy['take_profit'] = analysis['take_profit_levels'][0]['price']
        
        if action == 'buy' and analysis['stop_loss_levels']:
            # Use the 1h timeframe stop loss level if available
            for sl in analysis['stop_loss_levels']:
                if sl['timeframe'] == '1h':
                    exit_strategy['stop_loss'] = sl['price']
                    break
            else:
                exit_strategy['stop_loss'] = analysis['stop_loss_levels'][0]['price']
        
        return {
            'action': action,
            'confidence': confidence,
            'entry_point': entry_point,
            'exit_strategy': exit_strategy,
            'bullish_timeframes': bullish_timeframes,
            'bearish_timeframes': bearish_timeframes,
            'whale_signal': whale_signal
        }
    
    def generate_signal_message(self, symbol: str, analysis: Dict) -> str:
        """
        Generate a formatted message for Telegram notifications
        
        Args:
            symbol: The trading pair symbol
            analysis: Market analysis results
            
        Returns:
            Formatted message string
        """
        recommendation = analysis['recommendation']
        action = recommendation['action'].upper()
        confidence = recommendation['confidence']
        
        # Determine emoji based on action
        emoji = "🔄"
        if action == "BUY":
            emoji = "🟢"
        elif action == "SELL":
            emoji = "🔴"
        
        # Format message
        message = f"*{emoji} {symbol}/USDT Signal: {action}*\n\n"
        message += f"*Confidence:* {confidence:.1f}/10\n"
        
        # Add price information
        current_price = analysis['signals']['1h']['close'] if '1h' in analysis['signals'] else "Unknown"
        message += f"*Current Price:* ${current_price:.4f}\n\n"
        
        # Add entry point if available
        if recommendation['entry_point']:
            message += f"*Entry Point:* ${recommendation['entry_point']:.4f}\n"
        
        # Add exit strategy if available
        if 'take_profit' in recommendation['exit_strategy']:
            message += f"*Take Profit:* ${recommendation['exit_strategy']['take_profit']:.4f}\n"
        if 'stop_loss' in recommendation['exit_strategy']:
            message += f"*Stop Loss:* ${recommendation['exit_strategy']['stop_loss']:.4f}\n\n"
        
        # Add timeframe analysis
        message += "*Timeframe Analysis:*\n"
        for timeframe in self.timeframes:
            if timeframe in analysis['signals']:
                trend = analysis['signals'][timeframe]['trend']
                trend_emoji = "🟢" if trend == "bullish" else "🔴" if trend == "bearish" else "⚪"
                message += f"{trend_emoji} {timeframe}: {trend.capitalize()}\n"
        
        # Add whale analysis if available
        whale_analysis = analysis['whale_analysis']
        if whale_analysis['whale_signal']:
            whale_emoji = "🐋" if whale_analysis['whale_signal'] == "buy" else "🐳"
            message += f"\n*Whale Activity:* {whale_emoji} {whale_analysis['whale_signal'].capitalize()} pressure\n"
            message += f"Buy orders: {whale_analysis['buy_orders_count']} | Sell orders: {whale_analysis['sell_orders_count']}\n"
        
        # Add timestamp
        message += f"\n*Generated:* {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC"
        
        return message
    
    def monitor_symbol(self, symbol: str) -> None:
        """
        Monitor a single symbol and generate signals
        
        Args:
            symbol: The trading pair symbol
        """
        try:
            logger.info(f"Monitoring {symbol}...")
            
            # Detect whale orders
            self.detect_whale_orders(symbol)
            
            # Analyze market
            analysis = self.analyze_market(symbol)
            
            # Generate and send signal if confidence is high enough
            recommendation = analysis['recommendation']
            if recommendation['action'] != 'hold' and recommendation['confidence'] >= 7.0:
                message = self.generate_signal_message(symbol, analysis)
                self.send_telegram_message(message)
                
                # Also send chart for high confidence signals
                if recommendation['confidence'] >= 8.0:
                    self.send_telegram_chart(symbol, '1h', ['ema', 'macd', 'rsi'])
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error monitoring {symbol}: {e}")
            return None
    
    async def run(self, interval: int = 3600) -> None:
        """
        Run the trading bot in a continuous loop
        
        Args:
            interval: Time between iterations in seconds (default: 1 hour)
        """
        logger.info(f"Starting trading bot with {len(COINS)} coins")
        
        # Send detailed startup message
        import datetime
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get system info
        import platform
        system_info = f"{platform.system()} {platform.release()}"
        
        await self.send_telegram_message(
            f"⚙️ *Trading Bot Engine Initialized*\n\n"
            f"🔍 Scanning markets for trading opportunities\n"
            f"🪙 Coins monitored: {len(COINS)}\n"
            f"⏱️ Scan interval: {interval//60} minutes\n"
            f"💻 System: {system_info}\n"
            f"📅 Engine start time: {current_time}\n\n"
            f"📊 Trading signals will be sent automatically when detected"
        )
        
        try:
            while True:
                start_time = time.time()
                
                # Process each coin
                for symbol in COINS:
                    try:
                        self.monitor_symbol(symbol)
                        # Add a small delay between coins to avoid rate limits
                        time.sleep(2)
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                
                # Calculate time taken and sleep for the remaining interval
                elapsed = time.time() - start_time
                sleep_time = max(1, interval - elapsed)
                
                logger.info(f"Completed monitoring cycle in {elapsed:.2f}s. Sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
            self.send_telegram_message("🛑 *Trading Bot Stopped*\n\nThe bot was manually stopped.")
        except Exception as e:
            logger.error(f"Trading bot crashed: {e}")
            self.send_telegram_message(f"⚠️ *Trading Bot Error*\n\nThe bot encountered an error and stopped:\n`{str(e)}`")


def main():
    """
    Main function to start the trading bot
    """
    # Check if API keys are configured
    if not OKX_API_KEY or not OKX_SECRET_KEY or not OKX_PASSPHRASE:
        logger.error("OKX API credentials not configured. Please set them in the .env file.")
        return
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram configuration not set. Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the .env file.")
        return
    
    # Create and run the trading bot
    bot = TradingBot()
    
    # Run the bot with 15-minute interval
    bot.run(interval=900)  # 900 seconds = 15 minutes


if __name__ == "__main__":
    main()