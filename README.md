# Advanced Cryptocurrency Trading Bot

An advanced trading bot that analyzes cryptocurrency markets, detects whale activity, and sends trading signals via Telegram. The bot uses the OKX API to provide buy/sell signals with entry/exit points for multiple cryptocurrencies.

## Features

- **Real-time Market Analysis**: Monitors multiple timeframes (15m, 1h, 4h, 1d) for comprehensive analysis
- **Whalemap Indicator**: Detects large buy/sell orders to identify potential market bottoms or tops
- **Technical Analysis**: Implements multiple indicators (EMA, RSI, MACD, Bollinger Bands)
- **Telegram Notifications**: Sends real-time alerts with trading signals and charts
- **Entry/Exit Points**: Calculates optimal entry points, take profit levels, and stop losses
- **Multi-Coin Support**: Monitors 100+ cryptocurrencies simultaneously
- **Command Interface**: Interact with the bot via Telegram commands

## Installation

### Prerequisites

- Python 3.8 or higher
- OKX API credentials
- Telegram Bot Token and Chat ID

### Setup

1. Clone the repository or download the files

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root directory with your API credentials:

```
# OKX API credentials
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
```

## Usage

Start the bot by running:

```bash
python main.py
```

The bot will begin monitoring the configured cryptocurrencies and send notifications to your Telegram when it detects trading signals.

### Telegram Commands

The following commands are available in the Telegram chat:

- `/start` - Start the bot
- `/help` - Show help message
- `/status` - Show bot status
- `/coins` - List monitored coins

## Project Structure

- `main.py` - Entry point for the trading bot
- `trading_bot.py` - Core trading bot functionality
- `whalemap_indicator.py` - Implementation of the Whalemap indicator
- `technical_analysis.py` - Technical analysis functions and indicators
- `telegram_handler.py` - Telegram notification and command handling
- `config.py` - Configuration settings and parameters
- `requirements.txt` - Required Python packages

## Customization

You can customize the bot's behavior by modifying the following settings in the `.env` file or directly in `config.py`:

- **TRADING_MODE**: Set to 'spot' for spot trading
- **RISK_PERCENTAGE**: Percentage of portfolio to risk per trade
- **TAKE_PROFIT_PERCENTAGE**: Default take profit percentage
- **STOP_LOSS_PERCENTAGE**: Default stop loss percentage
- **WHALE_ORDER_THRESHOLD**: USD value to consider as whale order
- **WHALE_CONFIRMATION_COUNT**: Number of whale orders needed for confirmation

## Disclaimer

This trading bot is for educational and informational purposes only. Use it at your own risk. The creators are not responsible for any financial losses incurred from using this software. Always do your own research before making investment decisions.

## License

MIT License