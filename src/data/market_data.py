"""
Market Data Fetcher - Retrieves stock/market data for analysis.
Uses yfinance for historical data and Alpaca for real-time data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yfinance as yf
from loguru import logger

import sys
sys.path.insert(0, '/workspace')
from config.settings import settings


class MarketDataFetcher:
    """Fetches market data from various sources."""
    
    def __init__(self):
        self.alpaca_client = None
        self._init_alpaca()
    
    def _init_alpaca(self):
        """Initialize Alpaca client if credentials are available."""
        if settings.broker.alpaca_api_key and settings.broker.alpaca_secret_key:
            try:
                from alpaca_trade_api import REST
                self.alpaca_client = REST(
                    key_id=settings.broker.alpaca_api_key,
                    secret_key=settings.broker.alpaca_secret_key,
                    base_url=settings.broker.alpaca_base_url
                )
                logger.info("Alpaca client initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize Alpaca client: {e}")
    
    def get_historical_data(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            
            # Add symbol column
            df['symbol'] = symbol
            
            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_intraday_data(
        self,
        symbol: str,
        days: int = 5,
        interval: str = "5m"
    ) -> pd.DataFrame:
        """
        Get intraday data for technical analysis.
        
        Args:
            symbol: Stock ticker symbol
            days: Number of days of intraday data
            interval: Candle interval (1m, 5m, 15m, 30m, 1h)
        
        Returns:
            DataFrame with intraday OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d", interval=interval)
            
            if df.empty:
                logger.warning(f"No intraday data for {symbol}")
                return pd.DataFrame()
            
            df.columns = [col.lower() for col in df.columns]
            df['symbol'] = symbol
            
            logger.info(f"Fetched {len(df)} intraday bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get detailed quote information for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "price": info.get("regularMarketPrice"),
                "open": info.get("regularMarketOpen"),
                "high": info.get("regularMarketDayHigh"),
                "low": info.get("regularMarketDayLow"),
                "volume": info.get("regularMarketVolume"),
                "previous_close": info.get("regularMarketPreviousClose"),
                "change_percent": info.get("regularMarketChangePercent"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
            }
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return {}
    
    def get_multiple_symbols_data(
        self,
        symbols: List[str],
        period: str = "1mo",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            period: Data period
            interval: Data interval
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol, period, interval)
            if not df.empty:
                data[symbol] = df
        return data
    
    def get_market_status(self) -> Dict[str, Any]:
        """Check if the market is currently open."""
        if self.alpaca_client:
            try:
                clock = self.alpaca_client.get_clock()
                return {
                    "is_open": clock.is_open,
                    "next_open": clock.next_open.isoformat() if clock.next_open else None,
                    "next_close": clock.next_close.isoformat() if clock.next_close else None,
                }
            except Exception as e:
                logger.warning(f"Could not get market status from Alpaca: {e}")
        
        # Fallback: estimate based on time
        from datetime import datetime
        import pytz
        
        eastern = pytz.timezone('America/New_York')
        now = datetime.now(eastern)
        
        # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        is_weekday = now.weekday() < 5
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_open = is_weekday and market_open <= now <= market_close
        
        return {
            "is_open": is_open,
            "current_time": now.isoformat(),
            "note": "Estimated status (Alpaca not configured)"
        }
    
    def get_premarket_data(self, symbol: str) -> pd.DataFrame:
        """Get pre-market trading data."""
        try:
            ticker = yf.Ticker(symbol)
            # Get today's data with 1-minute intervals (includes pre-market)
            df = ticker.history(period="1d", interval="1m", prepost=True)
            
            if df.empty:
                return pd.DataFrame()
            
            df.columns = [col.lower() for col in df.columns]
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching pre-market data for {symbol}: {e}")
            return pd.DataFrame()


# Convenience function
def fetch_data_for_analysis(symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Fetch all necessary data for the AI bots to analyze.
    
    Returns:
        Dictionary containing:
        - daily_data: Daily OHLCV for trend analysis
        - intraday_data: Intraday data for entry/exit timing
        - quotes: Current quotes and fundamentals
    """
    if symbols is None:
        symbols = settings.trading.watch_symbols
    
    fetcher = MarketDataFetcher()
    
    result = {
        "daily_data": {},
        "intraday_data": {},
        "quotes": {},
        "market_status": fetcher.get_market_status()
    }
    
    for symbol in symbols:
        # Daily data for trend analysis (last 3 months)
        daily = fetcher.get_historical_data(symbol, period="3mo", interval="1d")
        if not daily.empty:
            result["daily_data"][symbol] = daily
        
        # Intraday for entry timing (5-minute candles)
        intraday = fetcher.get_intraday_data(symbol, days=5, interval="5m")
        if not intraday.empty:
            result["intraday_data"][symbol] = intraday
        
        # Current quote
        quote = fetcher.get_quote(symbol)
        if quote:
            result["quotes"][symbol] = quote
    
    return result
