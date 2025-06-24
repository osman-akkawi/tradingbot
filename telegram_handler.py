#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Telegram Notification Handler

This module handles all Telegram-related functionality for the trading bot,
including sending notifications, charts, and handling commands from the user.
"""

import os
import logging
import datetime
from typing import Dict, List, Optional, Union
import io

import telegram
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackContext, filters, ContextTypes
from telegram import Update
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class TelegramHandler:
    def __init__(self, token: str, chat_id: str):
        """
        Initialize the Telegram handler
        
        Args:
            token: Telegram bot token
            chat_id: Telegram chat ID to send messages to
        """
        self.token = token
        self.chat_id = chat_id
        self.bot = telegram.Bot(token=token)
        
        # Initialize application for handling commands
        self.application = None
        
        logger.info("Telegram handler initialized")
    
    async def send_message(self, message: str) -> bool:
        """
        Send a message to the configured Telegram chat
        
        Args:
            message: The message text to send
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode="Markdown"
            )
            logger.info(f"Telegram message sent: {message[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_chart(self, symbol: str, timeframe: str, df: pd.DataFrame, indicators: List[str] = None) -> bool:
        """
        Generate and send a chart image
        
        Args:
            symbol: The trading pair symbol
            timeframe: The timeframe for the chart
            df: DataFrame with OHLCV and indicator data
            indicators: List of indicators to include on the chart
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
            fig.suptitle(f"{symbol}/USDT - {timeframe} Chart", fontsize=16)
            
            # Plot price and indicators on main chart
            ax1 = axes[0]
            
            # Plot candlesticks
            width = 0.6
            width2 = 0.05
            up = df[df.close >= df.open]
            down = df[df.close < df.open]
            
            # Plot up candles
            ax1.bar(up.index, up.close-up.open, width, bottom=up.open, color='green')
            ax1.bar(up.index, up.high-up.close, width2, bottom=up.close, color='green')
            ax1.bar(up.index, up.low-up.open, width2, bottom=up.open, color='green')
            
            # Plot down candles
            ax1.bar(down.index, down.close-down.open, width, bottom=down.open, color='red')
            ax1.bar(down.index, down.high-down.open, width2, bottom=down.open, color='red')
            ax1.bar(down.index, down.low-down.close, width2, bottom=down.close, color='red')
            
            # Plot indicators if available
            if indicators:
                if 'ema' in indicators and 'ema_9' in df.columns and 'ema_21' in df.columns:
                    ax1.plot(df.index, df['ema_9'], color='blue', linewidth=1, label='EMA 9')
                    ax1.plot(df.index, df['ema_21'], color='orange', linewidth=1, label='EMA 21')
                
                if 'bollinger' in indicators and 'upper_band' in df.columns and 'lower_band' in df.columns:
                    ax1.plot(df.index, df['upper_band'], color='gray', linewidth=1, linestyle='--', label='Upper BB')
                    ax1.plot(df.index, df['middle_band'], color='gray', linewidth=1, linestyle='-', label='Middle BB')
                    ax1.plot(df.index, df['lower_band'], color='gray', linewidth=1, linestyle='--', label='Lower BB')
            
            # Add legend
            ax1.legend(loc='upper left')
            
            # Format x-axis
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # Add grid
            ax1.grid(True, alpha=0.3)
            
            # Plot volume on bottom subplot
            ax2 = axes[1]
            ax2.bar(up.index, up.volume, width, color='green', alpha=0.8)
            ax2.bar(down.index, down.volume, width, color='red', alpha=0.8)
            ax2.set_ylabel('Volume')
            
            # Format x-axis for volume subplot
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # Add grid to volume subplot
            ax2.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save chart to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Send chart to Telegram
            self.bot.send_photo(
                chat_id=self.chat_id,
                photo=buf,
                caption=f"{symbol}/USDT {timeframe} Chart - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # Close the figure to free memory
            plt.close(fig)
            
            logger.info(f"Telegram chart sent for {symbol} ({timeframe})")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send Telegram chart: {e}")
            return False
    
    def send_signal_notification(self, symbol: str, analysis: Dict) -> bool:
        """
        Send a formatted trading signal notification
        
        Args:
            symbol: The trading pair symbol
            analysis: Market analysis results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            recommendation = analysis['recommendation']
            action = recommendation['action'].upper()
            confidence = recommendation['confidence']
            
            # Determine emoji based on action and confidence
            emoji = "🔄"
            signal_strength = ""
            
            if action == "BUY":
                if confidence >= 7.0:
                    emoji = "🟢"
                    signal_strength = "STRONG"
                else:
                    emoji = "🟡"
                    signal_strength = f"WEAK ({confidence:.1f}/10)"
            elif action == "SELL":
                if confidence >= 7.0:
                    emoji = "🔴"
                    signal_strength = "STRONG"
                else:
                    emoji = "🟠"
                    signal_strength = f"WEAK ({confidence:.1f}/10)"
            
            # Format message
            message = f"*{emoji} {symbol}/USDT Signal: {action} - {signal_strength}*\n\n"
            message += f"*Confidence:* {confidence:.1f}/10\n"
            
            # Add price information
            current_price = analysis['signals']['1h']['close'] if '1h' in analysis['signals'] else "Unknown"
            message += f"*Current Price:* ${current_price:.4f}\n\n"
            
            # Add entry point if available - make it more prominent
            if recommendation['entry_point']:
                message += f"*⏩ ENTRY POINT:* ${recommendation['entry_point']:.4f}\n"
            
            # Add exit strategy if available - make it more prominent
            message += "*📊 EXIT STRATEGY:*\n"
            if 'take_profit' in recommendation['exit_strategy']:
                message += f"*✅ TAKE PROFIT:* ${recommendation['exit_strategy']['take_profit']:.4f}\n"
            if 'stop_loss' in recommendation['exit_strategy']:
                message += f"*⛔ STOP LOSS:* ${recommendation['exit_strategy']['stop_loss']:.4f}\n\n"
            
            # Add timeframe analysis
            message += "*Timeframe Analysis:*\n"
            timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d']
            for timeframe in timeframes:
                if timeframe in analysis['signals']:
                    trend = analysis['signals'][timeframe]['trend']
                    trend_strength = analysis['signals'][timeframe].get('strength', 0)
                    
                    # Determine emoji based on trend and strength
                    if trend == "bullish":
                        if trend_strength >= 7.0:
                            trend_emoji = "🟢"
                        else:
                            trend_emoji = "🟡"
                            trend = f"bullish (weak {trend_strength:.1f}/10)"
                    elif trend == "bearish":
                        if trend_strength >= 7.0:
                            trend_emoji = "🔴"
                        else:
                            trend_emoji = "🟠"
                            trend = f"bearish (weak {trend_strength:.1f}/10)"
                    else:
                        trend_emoji = "⚪"
                    
                    message += f"{trend_emoji} {timeframe}: {trend.capitalize()}\n"
            
            # Add whale analysis if available
            whale_analysis = analysis['whale_analysis']
            if whale_analysis['whale_signal']:
                whale_emoji = "🐋" if whale_analysis['whale_signal'] == "buy" else "🐳"
                message += f"\n*Whale Activity:* {whale_emoji} {whale_analysis['whale_signal'].capitalize()} pressure\n"
                message += f"Buy orders: {whale_analysis['buy_orders_count']} | Sell orders: {whale_analysis['sell_orders_count']}\n"
            
            # Add timestamp
            message += f"\n*Generated:* {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC"
            
            # Send the message
            return self.send_message(message)
        
        except Exception as e:
            logger.error(f"Failed to send signal notification: {e}")
            return False
    
    def start_command_handler(self) -> None:
        """
        Start the command handler to listen for user commands
        """
        try:
            import asyncio
            import pytz  # Import pytz at the beginning of the function
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Create application
            self.application = Application.builder().token(self.token).build()
            
            # Register command handlers
            self.application.add_handler(CommandHandler("start", self._cmd_start))
            self.application.add_handler(CommandHandler("help", self._cmd_help))
            self.application.add_handler(CommandHandler("status", self._cmd_status))
            self.application.add_handler(CommandHandler("coins", self._cmd_coins))
            
            # Start the bot with explicit timezone from pytz
            self.application.run_polling(
                allowed_updates=["message"],
                # Use UTC timezone from pytz library
                tzinfo=pytz.UTC
            )
            
            logger.info("Telegram command handler started")
        
        except Exception as e:
            logger.error(f"Failed to start Telegram command handler: {e}")
    
    def stop_command_handler(self) -> None:
        """
        Stop the command handler
        """
        if self.application:
            self.application.stop()
            logger.info("Telegram command handler stopped")
    
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /start command
        """
        await update.message.reply_text(
            "🤖 *Trading Bot Started*\n\n"
            "Welcome to the Advanced Crypto Trading Bot!\n\n"
            "Use /help to see available commands.",
            parse_mode="Markdown"
        )
    
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /help command
        """
        await update.message.reply_text(
            "🤖 *Trading Bot Commands*\n\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/status - Show bot status\n"
            "/coins - List monitored coins\n",
            parse_mode="Markdown"
        )
    
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /status command
        """
        # This would be updated with actual bot status
        await update.message.reply_text(
            "🤖 *Trading Bot Status*\n\n"
            "✅ Bot is running\n"
            "✅ API connection: OK\n"
            "✅ Monitoring active\n\n"
            f"Last update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            parse_mode="Markdown"
        )
    
    async def _cmd_coins(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /coins command
        """
        # This would be updated with actual monitored coins
        from trading_bot import COINS
        
        # Split the coins into chunks for better readability
        chunks = [COINS[i:i+10] for i in range(0, len(COINS), 10)]
        
        message = "🤖 *Monitored Coins*\n\n"
        for chunk in chunks:
            message += ", ".join(chunk) + "\n"
        
        await update.message.reply_text(message, parse_mode="Markdown")