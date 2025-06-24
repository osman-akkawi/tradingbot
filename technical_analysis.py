#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Technical Analysis Module

This module provides functions for calculating various technical indicators
and performing technical analysis on financial market data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average
    
    Args:
        data: Price series data
        period: EMA period
        
    Returns:
        Series with EMA values
    """
    return data.ewm(span=period, adjust=False).mean()


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average
    
    Args:
        data: Price series data
        period: SMA period
        
    Returns:
        Series with SMA values
    """
    return data.rolling(window=period).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index
    
    Args:
        data: Price series data
        period: RSI period
        
    Returns:
        Series with RSI values
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data: Price series data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)
    
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands
    
    Args:
        data: Price series data
        period: Moving average period
        std_dev: Number of standard deviations
        
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    middle_band = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period
        
    Returns:
        Series with ATR values
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: %K period
        d_period: %D period
        
    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    
    return k, d


def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """
    Calculate Fibonacci retracement levels
    
    Args:
        high: Highest price in range
        low: Lowest price in range
        
    Returns:
        Dictionary with Fibonacci levels
    """
    diff = high - low
    
    return {
        '0.0': low,
        '0.236': low + 0.236 * diff,
        '0.382': low + 0.382 * diff,
        '0.5': low + 0.5 * diff,
        '0.618': low + 0.618 * diff,
        '0.786': low + 0.786 * diff,
        '1.0': high
    }


def calculate_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """
    Calculate pivot points (standard)
    
    Args:
        high: Previous period high
        low: Previous period low
        close: Previous period close
        
    Returns:
        Dictionary with pivot levels
    """
    pivot = (high + low + close) / 3
    
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    
    return {
        'pivot': pivot,
        's1': s1,
        's2': s2,
        's3': s3,
        'r1': r1,
        'r2': r2,
        'r3': r3
    }


def detect_support_resistance(data: pd.DataFrame, window: int = 10, threshold: float = 0.01) -> Tuple[List[float], List[float]]:
    """
    Detect support and resistance levels using price swings
    
    Args:
        data: OHLC DataFrame
        window: Window size for detecting swings
        threshold: Minimum percentage change to consider as a swing
        
    Returns:
        Tuple of (Support levels, Resistance levels)
    """
    highs = data['high']
    lows = data['low']
    
    # Detect swing highs
    resistance_levels = []
    for i in range(window, len(highs) - window):
        if all(highs[i] > highs[i-j] for j in range(1, window+1)) and \
           all(highs[i] > highs[i+j] for j in range(1, window+1)):
            resistance_levels.append(highs[i])
    
    # Detect swing lows
    support_levels = []
    for i in range(window, len(lows) - window):
        if all(lows[i] < lows[i-j] for j in range(1, window+1)) and \
           all(lows[i] < lows[i+j] for j in range(1, window+1)):
            support_levels.append(lows[i])
    
    # Cluster similar levels
    support_levels = cluster_price_levels(support_levels, threshold)
    resistance_levels = cluster_price_levels(resistance_levels, threshold)
    
    return support_levels, resistance_levels


def cluster_price_levels(levels: List[float], threshold: float) -> List[float]:
    """
    Cluster similar price levels
    
    Args:
        levels: List of price levels
        threshold: Percentage threshold for clustering
        
    Returns:
        List of clustered price levels
    """
    if not levels:
        return []
    
    # Sort levels
    sorted_levels = sorted(levels)
    
    # Cluster similar levels
    clusters = []
    current_cluster = [sorted_levels[0]]
    
    for level in sorted_levels[1:]:
        prev_level = current_cluster[-1]
        if (level - prev_level) / prev_level <= threshold:
            # Add to current cluster
            current_cluster.append(level)
        else:
            # Start a new cluster
            clusters.append(sum(current_cluster) / len(current_cluster))
            current_cluster = [level]
    
    # Add the last cluster
    if current_cluster:
        clusters.append(sum(current_cluster) / len(current_cluster))
    
    return clusters


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate multiple technical indicators for a DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicators
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate EMAs
    df['ema_9'] = calculate_ema(df['close'], 9)
    df['ema_21'] = calculate_ema(df['close'], 21)
    df['ema_50'] = calculate_ema(df['close'], 50)
    df['ema_200'] = calculate_ema(df['close'], 200)
    
    # Calculate RSI
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    # Calculate MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    
    # Calculate Bollinger Bands
    df['upper_band'], df['middle_band'], df['lower_band'] = calculate_bollinger_bands(df['close'])
    
    # Calculate ATR
    df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
    
    # Calculate Stochastic Oscillator
    df['stoch_k'], df['stoch_d'] = calculate_stochastic_oscillator(df['high'], df['low'], df['close'])
    
    # Calculate volume indicators
    df['volume_sma'] = calculate_sma(df['volume'], 20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    return df


def identify_trend(df: pd.DataFrame) -> str:
    """
    Identify the current market trend
    
    Args:
        df: DataFrame with indicators
        
    Returns:
        Trend description ('bullish', 'bearish', or 'neutral')
    """
    # Get the latest values
    latest = df.iloc[-1]
    
    # Check EMA relationships
    ema_bullish = latest['ema_9'] > latest['ema_21'] > latest['ema_50']
    ema_bearish = latest['ema_9'] < latest['ema_21'] < latest['ema_50']
    
    # Check price vs EMAs
    price_above_emas = latest['close'] > latest['ema_50']
    price_below_emas = latest['close'] < latest['ema_50']
    
    # Check RSI
    rsi_bullish = latest['rsi'] > 50
    rsi_bearish = latest['rsi'] < 50
    
    # Check MACD
    macd_bullish = latest['macd'] > latest['macd_signal']
    macd_bearish = latest['macd'] < latest['macd_signal']
    
    # Combine signals
    bullish_signals = sum([ema_bullish, price_above_emas, rsi_bullish, macd_bullish])
    bearish_signals = sum([ema_bearish, price_below_emas, rsi_bearish, macd_bearish])
    
    if bullish_signals >= 3:
        return 'bullish'
    elif bearish_signals >= 3:
        return 'bearish'
    else:
        return 'neutral'


def generate_signals(df: pd.DataFrame) -> List[Dict]:
    """
    Generate trading signals based on technical indicators
    
    Args:
        df: DataFrame with indicators
        
    Returns:
        List of signal dictionaries
    """
    signals = []
    
    # Need at least 2 rows for comparison
    if len(df) < 2:
        return signals
    
    current = df.iloc[-1]
    previous = df.iloc[-2]
    
    # EMA crossover signals
    if previous['ema_9'] <= previous['ema_21'] and current['ema_9'] > current['ema_21']:
        signals.append({
            'type': 'ema_golden_cross',
            'direction': 'bullish',
            'strength': 7
        })
    elif previous['ema_9'] >= previous['ema_21'] and current['ema_9'] < current['ema_21']:
        signals.append({
            'type': 'ema_death_cross',
            'direction': 'bearish',
            'strength': 7
        })
    
    # RSI signals
    if current['rsi'] < 30:
        signals.append({
            'type': 'rsi_oversold',
            'direction': 'bullish',
            'strength': 6
        })
    elif current['rsi'] > 70:
        signals.append({
            'type': 'rsi_overbought',
            'direction': 'bearish',
            'strength': 6
        })
    
    # MACD signals
    if previous['macd_hist'] <= 0 and current['macd_hist'] > 0:
        signals.append({
            'type': 'macd_bullish_crossover',
            'direction': 'bullish',
            'strength': 8
        })
    elif previous['macd_hist'] >= 0 and current['macd_hist'] < 0:
        signals.append({
            'type': 'macd_bearish_crossover',
            'direction': 'bearish',
            'strength': 8
        })
    
    # Bollinger Band signals
    if current['close'] < current['lower_band']:
        signals.append({
            'type': 'bb_oversold',
            'direction': 'bullish',
            'strength': 6
        })
    elif current['close'] > current['upper_band']:
        signals.append({
            'type': 'bb_overbought',
            'direction': 'bearish',
            'strength': 6
        })
    
    # Volume signals
    if current['volume_ratio'] > 2.0:
        signals.append({
            'type': 'high_volume',
            'direction': 'neutral',
            'strength': 5
        })
    
    return signals


def calculate_entry_exit_points(df: pd.DataFrame, trend: str) -> Dict:
    """
    Calculate potential entry and exit points based on technical analysis
    
    Args:
        df: DataFrame with indicators
        trend: Current market trend
        
    Returns:
        Dictionary with entry and exit points
    """
    result = {
        'entry_points': [],
        'exit_points': [],
        'take_profit_levels': [],
        'stop_loss_levels': []
    }
    
    current_price = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1]
    
    # Support and resistance levels
    support_levels, resistance_levels = detect_support_resistance(df)
    
    # Entry points
    if trend == 'bullish':
        # Find nearest support level below current price
        supports_below = [level for level in support_levels if level < current_price]
        if supports_below:
            entry_price = max(supports_below)
            result['entry_points'].append({
                'price': entry_price,
                'type': 'buy',
                'confidence': 7
            })
        
        # Add current price as entry if strong bullish signals
        signals = generate_signals(df)
        bullish_signals = [s for s in signals if s['direction'] == 'bullish']
        if len(bullish_signals) >= 2:
            result['entry_points'].append({
                'price': current_price,
                'type': 'buy',
                'confidence': 6
            })
    
    elif trend == 'bearish':
        # Find nearest resistance level above current price
        resistances_above = [level for level in resistance_levels if level > current_price]
        if resistances_above:
            entry_price = min(resistances_above)
            result['entry_points'].append({
                'price': entry_price,
                'type': 'sell',
                'confidence': 7
            })
    
    # Exit points
    if trend == 'bullish':
        # Find nearest resistance level above current price
        resistances_above = [level for level in resistance_levels if level > current_price]
        if resistances_above:
            exit_price = min(resistances_above)
            result['exit_points'].append({
                'price': exit_price,
                'type': 'take_profit',
                'confidence': 7
            })
    
    elif trend == 'bearish':
        # Find nearest support level below current price
        supports_below = [level for level in support_levels if level < current_price]
        if supports_below:
            exit_price = max(supports_below)
            result['exit_points'].append({
                'price': exit_price,
                'type': 'take_profit',
                'confidence': 7
            })
    
    # Take profit levels (using ATR)
    if trend == 'bullish':
        result['take_profit_levels'].append({
            'price': current_price + (2 * atr),
            'percentage': round((current_price + (2 * atr)) / current_price * 100 - 100, 2)
        })
        result['take_profit_levels'].append({
            'price': current_price + (3 * atr),
            'percentage': round((current_price + (3 * atr)) / current_price * 100 - 100, 2)
        })
    
    # Stop loss levels (using ATR)
    if trend == 'bullish':
        result['stop_loss_levels'].append({
            'price': current_price - (1.5 * atr),
            'percentage': round(100 - (current_price - (1.5 * atr)) / current_price * 100, 2)
        })
    
    return result