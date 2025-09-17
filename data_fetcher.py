"""
Market data fetching module for AI Trading Bot
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class DataFetcher:
    """Handles fetching and preprocessing market data"""
    
    def __init__(self):
        self.cache = {}
        
    def fetch_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch stock data for a given symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            cache_key = f"{symbol}_{period}_{interval}"
            
            # Check cache first
            if cache_key in self.cache:
                cached_data, cache_time = self.cache[cache_key]
                if datetime.now() - cache_time < timedelta(hours=1):  # Cache for 1 hour
                    logger.info(f"Using cached data for {symbol}")
                    return cached_data
            
            logger.info(f"Fetching data for {symbol}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return None
                
            # Cache the data
            self.cache[cache_key] = (data, datetime.now())
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_multiple_stocks(self, symbols: List[str], period: str = "1y", interval: str = "1d") -> dict:
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        data_dict = {}
        
        for symbol in symbols:
            data = self.fetch_stock_data(symbol, period, interval)
            if data is not None:
                data_dict[symbol] = data
                
        logger.info(f"Successfully fetched data for {len(data_dict)} out of {len(symbols)} symbols")
        return data_dict
    
    def get_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given data
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with additional technical indicators
        """
        try:
            df = data.copy()
            
            # Moving averages
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Exponential moving averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            # Price change indicators
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
            
            logger.info("Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return data
    
    def get_market_sentiment_data(self) -> dict:
        """
        Get basic market sentiment indicators (simplified)
        
        Returns:
            Dictionary with sentiment indicators
        """
        try:
            # Fetch VIX (Fear & Greed indicator)
            vix_data = self.fetch_stock_data("^VIX", period="5d", interval="1d")
            
            # Fetch major indices
            spy_data = self.fetch_stock_data("SPY", period="5d", interval="1d")
            
            sentiment = {
                'vix_current': vix_data['Close'].iloc[-1] if vix_data is not None else None,
                'vix_change': vix_data['Close'].pct_change().iloc[-1] if vix_data is not None else None,
                'market_trend': 'bullish' if spy_data is not None and spy_data['Close'].pct_change().iloc[-1] > 0 else 'bearish'
            }
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error getting market sentiment: {str(e)}")
            return {}