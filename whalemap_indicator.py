#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Whalemap Indicator Module

This module implements the Whalemap indicator which aims to spot big buying and selling activity
represented as large orders that may indicate potential bottom or top formation on the chart.
"""

import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import pytz

# Configure logging
logger = logging.getLogger(__name__)


class WhalemapIndicator:
    def __init__(self, order_threshold: float = 100000, confirmation_count: int = 3, lookback_hours: int = 24):
        """
        Initialize the Whalemap indicator
        
        Args:
            order_threshold: USD value to consider as whale order
            confirmation_count: Number of whale orders needed for confirmation
            lookback_hours: Hours to look back for whale activity
        """
        self.order_threshold = order_threshold
        self.confirmation_count = confirmation_count
        self.lookback_hours = lookback_hours
        self.whale_orders = {}
        
    def process_order_book(self, symbol: str, order_book: Dict, timestamp: datetime.datetime = None) -> List[Dict]:
        """
        Process order book data to detect whale orders
        
        Args:
            symbol: The trading pair symbol
            order_book: Order book data with bids and asks
            timestamp: Timestamp of the order book data (default: current time)
            
        Returns:
            List of detected whale orders
        """
        if timestamp is None:
            timestamp = datetime.datetime.now(pytz.UTC)
        
        whale_orders = []
        
        # Check for large buy orders
        for price, volume in order_book['bids']:
            order_value = price * volume
            if order_value >= self.order_threshold:
                whale_orders.append({
                    'symbol': symbol,
                    'type': 'buy',
                    'price': price,
                    'volume': volume,
                    'value': order_value,
                    'timestamp': timestamp
                })
        
        # Check for large sell orders
        for price, volume in order_book['asks']:
            order_value = price * volume
            if order_value >= self.order_threshold:
                whale_orders.append({
                    'symbol': symbol,
                    'type': 'sell',
                    'price': price,
                    'volume': volume,
                    'value': order_value,
                    'timestamp': timestamp
                })
        
        # Store whale orders
        if symbol not in self.whale_orders:
            self.whale_orders[symbol] = []
        
        self.whale_orders[symbol].extend(whale_orders)
        
        # Keep only recent orders within lookback period
        cutoff_time = datetime.datetime.now(pytz.UTC) - datetime.timedelta(hours=self.lookback_hours)
        self.whale_orders[symbol] = [order for order in self.whale_orders[symbol] 
                                    if order['timestamp'] > cutoff_time]
        
        if whale_orders:
            logger.info(f"Detected {len(whale_orders)} whale orders for {symbol}")
        
        return whale_orders
    
    def analyze(self, symbol: str) -> Dict:
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
        
        # Calculate average price levels for buy and sell orders
        buy_prices = [order['price'] for order in buy_orders]
        sell_prices = [order['price'] for order in sell_orders]
        
        avg_buy_price = sum(buy_prices) / len(buy_prices) if buy_prices else 0
        avg_sell_price = sum(sell_prices) / len(sell_prices) if sell_prices else 0
        
        # Determine if there's a whale signal
        whale_signal = None
        signal_strength = 0
        support_level = None
        resistance_level = None
        
        if len(buy_orders) >= self.confirmation_count and buy_value > sell_value * 2:
            whale_signal = 'buy'
            signal_strength = min(10, len(buy_orders) / self.confirmation_count * 5)
            support_level = avg_buy_price
        elif len(sell_orders) >= self.confirmation_count and sell_value > buy_value * 2:
            whale_signal = 'sell'
            signal_strength = min(10, len(sell_orders) / self.confirmation_count * 5)
            resistance_level = avg_sell_price
        
        # Detect clusters of orders at specific price levels
        buy_clusters = self._detect_price_clusters(buy_orders)
        sell_clusters = self._detect_price_clusters(sell_orders)
        
        return {
            'symbol': symbol,
            'whale_signal': whale_signal,
            'signal_strength': signal_strength,
            'buy_orders_count': len(buy_orders),
            'sell_orders_count': len(sell_orders),
            'buy_value': buy_value,
            'sell_value': sell_value,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'buy_clusters': buy_clusters,
            'sell_clusters': sell_clusters
        }
    
    def _detect_price_clusters(self, orders: List[Dict], price_tolerance: float = 0.01) -> List[Dict]:
        """
        Detect clusters of orders at similar price levels
        
        Args:
            orders: List of orders to analyze
            price_tolerance: Percentage tolerance for grouping prices
            
        Returns:
            List of price clusters with their total value
        """
        if not orders:
            return []
        
        # Sort orders by price
        sorted_orders = sorted(orders, key=lambda x: x['price'])
        
        clusters = []
        current_cluster = {
            'orders': [sorted_orders[0]],
            'avg_price': sorted_orders[0]['price'],
            'total_value': sorted_orders[0]['value']
        }
        
        for order in sorted_orders[1:]:
            # Check if this order is within tolerance of current cluster
            if abs(order['price'] - current_cluster['avg_price']) / current_cluster['avg_price'] <= price_tolerance:
                # Add to current cluster
                current_cluster['orders'].append(order)
                current_cluster['total_value'] += order['value']
                # Recalculate average price
                current_cluster['avg_price'] = sum(o['price'] for o in current_cluster['orders']) / len(current_cluster['orders'])
            else:
                # Start a new cluster
                clusters.append(current_cluster)
                current_cluster = {
                    'orders': [order],
                    'avg_price': order['price'],
                    'total_value': order['value']
                }
        
        # Add the last cluster
        clusters.append(current_cluster)
        
        # Format the output
        formatted_clusters = [{
            'price': cluster['avg_price'],
            'total_value': cluster['total_value'],
            'order_count': len(cluster['orders'])
        } for cluster in clusters]
        
        # Sort by total value
        return sorted(formatted_clusters, key=lambda x: x['total_value'], reverse=True)
    
    def get_key_levels(self, symbol: str) -> Dict:
        """
        Get key support and resistance levels based on whale orders
        
        Args:
            symbol: The trading pair symbol
            
        Returns:
            Dictionary with key price levels
        """
        analysis = self.analyze(symbol)
        
        key_levels = {
            'support_levels': [],
            'resistance_levels': []
        }
        
        # Add support levels from buy clusters
        if 'buy_clusters' in analysis and analysis['buy_clusters']:
            for cluster in analysis['buy_clusters'][:3]:  # Top 3 clusters
                if cluster['order_count'] >= self.confirmation_count / 2:
                    key_levels['support_levels'].append({
                        'price': cluster['price'],
                        'strength': min(10, cluster['total_value'] / self.order_threshold)
                    })
        
        # Add resistance levels from sell clusters
        if 'sell_clusters' in analysis and analysis['sell_clusters']:
            for cluster in analysis['sell_clusters'][:3]:  # Top 3 clusters
                if cluster['order_count'] >= self.confirmation_count / 2:
                    key_levels['resistance_levels'].append({
                        'price': cluster['price'],
                        'strength': min(10, cluster['total_value'] / self.order_threshold)
                    })
        
        return key_levels
    
    def get_entry_exit_points(self, symbol: str, current_price: float) -> Dict:
        """
        Get suggested entry and exit points based on whale activity
        
        Args:
            symbol: The trading pair symbol
            current_price: Current market price
            
        Returns:
            Dictionary with entry and exit points
        """
        key_levels = self.get_key_levels(symbol)
        analysis = self.analyze(symbol)
        
        result = {
            'entry_points': [],
            'exit_points': [],
            'stop_loss_levels': []
        }
        
        # Determine entry points based on support levels and whale signal
        if analysis['whale_signal'] == 'buy':
            for level in key_levels['support_levels']:
                if level['price'] < current_price * 1.05:  # Within 5% of current price
                    result['entry_points'].append({
                        'price': level['price'],
                        'type': 'buy',
                        'confidence': level['strength']
                    })
        
        # Determine exit points based on resistance levels and whale signal
        if analysis['whale_signal'] == 'sell' or key_levels['resistance_levels']:
            for level in key_levels['resistance_levels']:
                if level['price'] > current_price * 0.95:  # Within 5% of current price
                    result['exit_points'].append({
                        'price': level['price'],
                        'type': 'sell',
                        'confidence': level['strength']
                    })
        
        # Determine stop loss levels
        if result['entry_points'] and key_levels['support_levels']:
            # Use the lowest support level as stop loss
            lowest_support = min(key_levels['support_levels'], key=lambda x: x['price'])
            result['stop_loss_levels'].append({
                'price': lowest_support['price'] * 0.98,  # 2% below support
                'confidence': lowest_support['strength']
            })
        
        return result