#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Utilities Module

This module provides utilities for handling API rate limiting,
error handling, and other common API-related tasks.
"""

import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import wraps

import ccxt

# Configure logging
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter for API calls to prevent hitting rate limits
    """
    def __init__(self, calls_per_second: int = 5, calls_per_minute: int = 100):
        """
        Initialize the rate limiter
        
        Args:
            calls_per_second: Maximum number of calls allowed per second
            calls_per_minute: Maximum number of calls allowed per minute
        """
        self.calls_per_second = calls_per_second
        self.calls_per_minute = calls_per_minute
        self.second_bucket = []
        self.minute_bucket = []
    
    def wait_if_needed(self) -> None:
        """
        Wait if necessary to comply with rate limits
        """
        current_time = time.time()
        
        # Clean up old timestamps
        self.second_bucket = [t for t in self.second_bucket if current_time - t < 1.0]
        self.minute_bucket = [t for t in self.minute_bucket if current_time - t < 60.0]
        
        # Check if we need to wait for per-second limit
        if len(self.second_bucket) >= self.calls_per_second:
            sleep_time = 1.0 - (current_time - self.second_bucket[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit: Sleeping for {sleep_time:.2f}s (per-second limit)")
                time.sleep(sleep_time)
                current_time = time.time()  # Update current time after sleeping
        
        # Check if we need to wait for per-minute limit
        if len(self.minute_bucket) >= self.calls_per_minute:
            sleep_time = 60.0 - (current_time - self.minute_bucket[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit: Sleeping for {sleep_time:.2f}s (per-minute limit)")
                time.sleep(sleep_time)
                current_time = time.time()  # Update current time after sleeping
        
        # Add current timestamp to buckets
        self.second_bucket.append(current_time)
        self.minute_bucket.append(current_time)


def rate_limited(func: Callable) -> Callable:
    """
    Decorator for rate-limited API calls
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    # Create a rate limiter for this function
    limiter = RateLimiter()
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        limiter.wait_if_needed()
        return func(*args, **kwargs)
    
    return wrapper


def retry_on_error(max_retries: int = 3, retry_delay: float = 1.0) -> Callable:
    """
    Decorator for retrying API calls on error
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except ccxt.NetworkError as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Network error after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Network error, retrying ({retries}/{max_retries}): {e}")
                    time.sleep(retry_delay * retries)  # Exponential backoff
                
                except ccxt.ExchangeError as e:
                    # Don't retry on exchange errors (invalid API key, insufficient funds, etc.)
                    logger.error(f"Exchange error: {e}")
                    raise
                
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Error after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Error, retrying ({retries}/{max_retries}): {e}")
                    time.sleep(retry_delay * retries)  # Exponential backoff
        
        return wrapper
    
    return decorator


class OKXClient:
    """
    Client for interacting with the OKX API
    """
    def __init__(self, api_key: str, secret_key: str, passphrase: str):
        """
        Initialize the OKX client
        
        Args:
            api_key: OKX API key
            secret_key: OKX secret key
            passphrase: OKX API passphrase
        """
        self.exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': secret_key,
            'password': passphrase,
            'enableRateLimit': True,
        })
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter()
    
    @retry_on_error(max_retries=3)
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List]:
        """
        Fetch OHLCV data for a symbol and timeframe
        
        Args:
            symbol: The trading pair symbol
            timeframe: The timeframe for the data
            limit: Number of candles to fetch
            
        Returns:
            List of OHLCV data
        """
        self.rate_limiter.wait_if_needed()
        return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    @retry_on_error(max_retries=3)
    def fetch_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        Fetch order book data for a symbol
        
        Args:
            symbol: The trading pair symbol
            limit: Depth of the order book to fetch
            
        Returns:
            Order book data
        """
        self.rate_limiter.wait_if_needed()
        return self.exchange.fetch_order_book(symbol, limit=limit)
    
    @retry_on_error(max_retries=3)
    def fetch_ticker(self, symbol: str) -> Dict:
        """
        Fetch ticker data for a symbol
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Ticker data
        """
        self.rate_limiter.wait_if_needed()
        return self.exchange.fetch_ticker(symbol)
    
    @retry_on_error(max_retries=3)
    def fetch_balance(self) -> Dict:
        """
        Fetch account balance
        
        Returns:
            Account balance data
        """
        self.rate_limiter.wait_if_needed()
        return self.exchange.fetch_balance()
    
    def format_symbol(self, base_currency: str) -> str:
        """
        Format a symbol for OKX API
        
        Args:
            base_currency: Base currency code
            
        Returns:
            Formatted symbol string
        """
        return f"{base_currency}/USDT"
    
    def get_price(self, symbol: str) -> float:
        """
        Get current price for a symbol
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Current price
        """
        ticker = self.fetch_ticker(symbol)
        return ticker['last']
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading pairs
        
        Returns:
            List of available symbols
        """
        self.rate_limiter.wait_if_needed()
        markets = self.exchange.load_markets()
        return list(markets.keys())