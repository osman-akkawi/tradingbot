#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration Module

This module contains all configuration settings and parameters for the trading bot.
It loads values from environment variables and provides default values when needed.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# Technical analysis parameters
EMA_SHORT = 9
EMA_MEDIUM = 21
EMA_LONG = 50
EMA_EXTRA_LONG = 200
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Timeframes to analyze
TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']

# Monitoring interval in seconds
MONITORING_INTERVAL = 900  # 15 minutes

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

# Signal confidence threshold for notifications
SIGNAL_CONFIDENCE_THRESHOLD = 7.0

# Rate limiting settings
API_CALLS_PER_SECOND = 5
API_CALLS_PER_MINUTE = 100

# Check if required configuration is set
def check_config():
    """
    Check if all required configuration is set
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not OKX_API_KEY or not OKX_SECRET_KEY or not OKX_PASSPHRASE:
        return False, "OKX API credentials not configured. Please set them in the .env file."
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "Telegram configuration not set. Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the .env file."
    
    return True, "Configuration is valid."


# Print configuration summary
def print_config_summary():
    """
    Print a summary of the current configuration
    """
    logger.info("=== Trading Bot Configuration ===")
    logger.info(f"Trading Mode: {TRADING_MODE}")
    logger.info(f"Risk Percentage: {RISK_PERCENTAGE}%")
    logger.info(f"Take Profit: {TAKE_PROFIT_PERCENTAGE}%")
    logger.info(f"Stop Loss: {STOP_LOSS_PERCENTAGE}%")
    logger.info(f"Whale Order Threshold: ${WHALE_ORDER_THRESHOLD}")
    logger.info(f"Monitoring Interval: {MONITORING_INTERVAL} seconds")
    logger.info(f"Number of Coins: {len(COINS)}")
    logger.info(f"Timeframes: {', '.join(TIMEFRAMES)}")
    logger.info("===============================")