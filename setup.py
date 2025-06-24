#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup and Configuration Script

This script helps users set up their environment and test their API connections
before running the full trading bot.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt
import telegram
from dotenv import load_dotenv

# Configure logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def create_env_file() -> None:
    """
    Create a .env file with template values if it doesn't exist
    """
    env_path = Path('.env')
    
    if env_path.exists():
        logger.info(".env file already exists. Skipping creation.")
        return
    
    env_example_path = Path('.env.example')
    if env_example_path.exists():
        # Copy from example file
        with open(env_example_path, 'r') as example_file:
            env_content = example_file.read()
    else:
        # Create default template
        env_content = """# OKX API credentials
OKX_API_KEY=your_okx_api_key_here
OKX_SECRET_KEY=your_okx_secret_key_here
OKX_PASSPHRASE=your_okx_passphrase_here

# Telegram configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Trading configuration
TRADING_MODE=spot
RISK_PERCENTAGE=1
TAKE_PROFIT_PERCENTAGE=3
STOP_LOSS_PERCENTAGE=2

# Whalemap indicator settings
WHALE_ORDER_THRESHOLD=100000
WHALE_CONFIRMATION_COUNT=3
"""
    
    with open(env_path, 'w') as env_file:
        env_file.write(env_content)
    
    logger.info(f".env file created at {env_path.absolute()}")
    logger.info("Please edit this file with your API credentials before running the bot.")


def test_okx_connection() -> bool:
    """
    Test connection to OKX API
    
    Returns:
        True if successful, False otherwise
    """
    load_dotenv()
    
    api_key = os.getenv('OKX_API_KEY')
    secret_key = os.getenv('OKX_SECRET_KEY')
    passphrase = os.getenv('OKX_PASSPHRASE')
    
    if not api_key or not secret_key or not passphrase:
        logger.error("OKX API credentials not found in .env file")
        return False
    
    if api_key == 'your_okx_api_key_here':
        logger.error("OKX API credentials not configured. Please edit the .env file.")
        return False
    
    try:
        logger.info("Testing connection to OKX API...")
        exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': secret_key,
            'password': passphrase,
            'enableRateLimit': True,
        })
        
        # Test public endpoint
        markets = exchange.load_markets()
        logger.info(f"Successfully connected to OKX API. Found {len(markets)} markets.")
        
        # Test private endpoint
        balance = exchange.fetch_balance()
        logger.info("Successfully authenticated with OKX API.")
        
        return True
    
    except ccxt.AuthenticationError as e:
        logger.error(f"Authentication error: {e}")
        logger.error("Please check your OKX API credentials in the .env file.")
        return False
    
    except Exception as e:
        logger.error(f"Error connecting to OKX API: {e}")
        return False


def test_telegram_connection() -> bool:
    """
    Test connection to Telegram API
    
    Returns:
        True if successful, False otherwise
    """
    load_dotenv()
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        logger.error("Telegram credentials not found in .env file")
        return False
    
    if bot_token == 'your_telegram_bot_token_here':
        logger.error("Telegram credentials not configured. Please edit the .env file.")
        return False
    
    try:
        logger.info("Testing connection to Telegram API...")
        bot = telegram.Bot(token=bot_token)
        bot_info = bot.get_me()
        logger.info(f"Successfully connected to Telegram API. Bot name: {bot_info.first_name}")
        
        # Send test message
        message = "🤖 *Trading Bot Setup Test*\n\nThis is a test message to verify your Telegram configuration."
        bot.send_message(chat_id=chat_id, text=message, parse_mode=telegram.ParseMode.MARKDOWN)
        logger.info("Test message sent successfully.")
        
        return True
    
    except telegram.error.Unauthorized as e:
        logger.error(f"Telegram authentication error: {e}")
        logger.error("Please check your TELEGRAM_BOT_TOKEN in the .env file.")
        return False
    
    except telegram.error.BadRequest as e:
        logger.error(f"Telegram bad request error: {e}")
        logger.error("Please check your TELEGRAM_CHAT_ID in the .env file.")
        return False
    
    except Exception as e:
        logger.error(f"Error connecting to Telegram API: {e}")
        return False


def check_dependencies() -> bool:
    """
    Check if all required dependencies are installed
    
    Returns:
        True if all dependencies are installed, False otherwise
    """
    required_packages = [
        'ccxt',
        'python-telegram-bot',
        'pandas',
        'numpy',
        'matplotlib',
        'python-dotenv',
        'requests',
        'websocket-client'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install them using: pip install -r requirements.txt")
        return False
    
    logger.info("All required dependencies are installed.")
    return True


def check_files() -> bool:
    """
    Check if all required files exist
    
    Returns:
        True if all files exist, False otherwise
    """
    required_files = [
        'main.py',
        'trading_bot.py',
        'whalemap_indicator.py',
        'technical_analysis.py',
        'telegram_handler.py',
        'config.py',
        'api_utils.py',
        'requirements.txt'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {', '.join(missing_files)}")
        return False
    
    logger.info("All required files are present.")
    return True


def test_okx_market_data() -> bool:
    """
    Test fetching market data from OKX
    
    Returns:
        True if successful, False otherwise
    """
    load_dotenv()
    
    api_key = os.getenv('OKX_API_KEY')
    secret_key = os.getenv('OKX_SECRET_KEY')
    passphrase = os.getenv('OKX_PASSPHRASE')
    
    if not api_key or api_key == 'your_okx_api_key_here':
        logger.error("OKX API credentials not configured. Skipping market data test.")
        return False
    
    try:
        exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': secret_key,
            'password': passphrase,
            'enableRateLimit': True,
        })
        
        # Test symbols from our list
        test_symbols = ['BTC/USDT', 'ETH/USDT', 'LINK/USDT', 'DOGE/USDT', 'XRP/USDT']
        
        for symbol in test_symbols:
            try:
                logger.info(f"Testing market data for {symbol}...")
                
                # Fetch ticker
                ticker = exchange.fetch_ticker(symbol)
                logger.info(f"{symbol} ticker: Last price = {ticker['last']}")
                
                # Fetch OHLCV
                ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=5)
                logger.info(f"{symbol} OHLCV: Fetched {len(ohlcv)} candles")
                
                # Fetch order book
                order_book = exchange.fetch_order_book(symbol, limit=5)
                logger.info(f"{symbol} order book: {len(order_book['bids'])} bids, {len(order_book['asks'])} asks")
                
                time.sleep(1)  # Avoid rate limits
            
            except Exception as e:
                logger.warning(f"Could not fetch data for {symbol}: {e}")
        
        logger.info("Market data test completed successfully.")
        return True
    
    except Exception as e:
        logger.error(f"Error testing market data: {e}")
        return False


def main() -> None:
    """
    Main function to run the setup script
    """
    logger.info("=== Trading Bot Setup Script ===")
    
    # Create .env file if it doesn't exist
    create_env_file()
    
    # Check dependencies
    dependencies_ok = check_dependencies()
    
    # Check files
    files_ok = check_files()
    
    # Test OKX connection
    okx_ok = test_okx_connection()
    
    # Test Telegram connection
    telegram_ok = test_telegram_connection()
    
    # Test market data if OKX connection is OK
    market_data_ok = False
    if okx_ok:
        market_data_ok = test_okx_market_data()
    
    # Print summary
    logger.info("\n=== Setup Summary ===")
    logger.info(f"Dependencies check: {'✅ PASSED' if dependencies_ok else '❌ FAILED'}")
    logger.info(f"Files check: {'✅ PASSED' if files_ok else '❌ FAILED'}")
    logger.info(f"OKX API connection: {'✅ PASSED' if okx_ok else '❌ FAILED'}")
    logger.info(f"Telegram API connection: {'✅ PASSED' if telegram_ok else '❌ FAILED'}")
    logger.info(f"Market data test: {'✅ PASSED' if market_data_ok else '❌ FAILED' if okx_ok else '⏭️ SKIPPED'}")
    
    if dependencies_ok and files_ok and okx_ok and telegram_ok and market_data_ok:
        logger.info("\n✅ All tests passed! You can now run the trading bot using: python main.py")
    else:
        logger.warning("\n⚠️ Some tests failed. Please fix the issues before running the trading bot.")


if __name__ == "__main__":
    main()