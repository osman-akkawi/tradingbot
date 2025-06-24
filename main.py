#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Cryptocurrency Trading Bot

This is the main entry point for the trading bot that:
- Uses OKX API for market data and trading
- Implements Whalemap indicator to detect large buy/sell orders
- Sends notifications via Telegram
- Provides entry/exit points and take profit levels
"""

import time
import logging
import sys

# Import configuration
from config import (
    check_config, print_config_summary, MONITORING_INTERVAL, COINS,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
)

# Import components
from trading_bot import TradingBot
from telegram_handler import TelegramHandler

# Configure logging
logger = logging.getLogger(__name__)


async def main():
    """
    Main function to start the trading bot
    """
    # Check if configuration is valid
    is_valid, error_message = check_config()
    if not is_valid:
        logger.error(error_message)
        sys.exit(1)
    
    # Print configuration summary
    print_config_summary()
    
    # Initialize Telegram handler
    telegram_handler = TelegramHandler(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    
    # Send startup message with timestamp
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await telegram_handler.send_message(
        f"🚀 *Advanced Trading Bot Started*\n\n"
        f"✅ Status: Bot is now running\n"
        f"📊 Monitoring {len(COINS)} coins\n"
        f"⏱️ Interval: {MONITORING_INTERVAL//60} minutes\n"
        f"📅 Start time: {current_time}\n\n"
        f"💡 Type /help for available commands\n\n"
        f"📱 This message confirms your bot is working correctly!"
    )
    
    # Start Telegram command handler in a separate thread
    import threading
    telegram_thread = threading.Thread(target=telegram_handler.start_command_handler)
    telegram_thread.daemon = True
    telegram_thread.start()
    
    # Create and run the trading bot
    bot = TradingBot()
    
    try:
        # Run the bot with the configured interval
        await bot.run(interval=MONITORING_INTERVAL)
    
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
        await telegram_handler.send_message("🛑 *Trading Bot Stopped*\n\nThe bot was manually stopped.")
    
    except Exception as e:
        logger.error(f"Trading bot crashed: {e}")
        await telegram_handler.send_message(
            f"⚠️ *Trading Bot Error*\n\n"
            f"The bot encountered an error and stopped:\n"
            f"`{str(e)}`"
        )
    
    finally:
        # Stop Telegram command handler
        telegram_handler.stop_command_handler()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())