"""
Configuration settings loader for the Trading AI System.
"""
import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class AIConfig:
    """AI API configuration."""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"
    gemini_model: str = "gemini-1.5-pro"


@dataclass
class BrokerConfig:
    """Broker API configuration."""
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"


@dataclass
class TradingConfig:
    """Trading parameters."""
    watch_symbols: List[str] = field(default_factory=lambda: ["SPY", "AAPL", "MSFT", "GOOGL", "NVDA"])
    max_position_size_pct: float = 5.0
    max_daily_loss_pct: float = 2.0
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    max_positions: int = 5
    market_timezone: str = "America/New_York"


@dataclass
class NewsConfig:
    """News API configuration."""
    news_api_key: str = ""
    rss_feeds: List[str] = field(default_factory=lambda: [
        "https://feeds.finance.yahoo.com/rss/2.0/headline",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.bloomberg.com/markets/news.rss",
    ])


class Settings:
    """Main settings class that loads all configuration."""
    
    def __init__(self):
        self.ai = AIConfig(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        )
        
        self.broker = BrokerConfig(
            alpaca_api_key=os.getenv("ALPACA_API_KEY", ""),
            alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
            alpaca_base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        )
        
        # Parse watch symbols from comma-separated string
        symbols_str = os.getenv("WATCH_SYMBOLS", "SPY,AAPL,MSFT,GOOGL,NVDA,TSLA,AMD,META")
        watch_symbols = [s.strip() for s in symbols_str.split(",")]
        
        self.trading = TradingConfig(
            watch_symbols=watch_symbols,
            max_position_size_pct=float(os.getenv("MAX_POSITION_SIZE_PCT", "5")),
            max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "2")),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "2")),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "4")),
            max_positions=int(os.getenv("MAX_POSITIONS", "5")),
            market_timezone=os.getenv("MARKET_TIMEZONE", "America/New_York"),
        )
        
        self.news = NewsConfig(
            news_api_key=os.getenv("NEWS_API_KEY", ""),
        )
    
    def validate(self) -> dict:
        """Validate configuration and return status."""
        status = {
            "ai_claude": bool(self.ai.anthropic_api_key),
            "ai_gemini": bool(self.ai.google_api_key),
            "broker": bool(self.broker.alpaca_api_key and self.broker.alpaca_secret_key),
            "news_api": bool(self.news.news_api_key),
        }
        return status
    
    def print_status(self):
        """Print configuration status."""
        status = self.validate()
        print("\n" + "="*50)
        print("TRADING AI SYSTEM - Configuration Status")
        print("="*50)
        for key, configured in status.items():
            icon = "✅" if configured else "❌"
            print(f"  {icon} {key}: {'Configured' if configured else 'Missing'}")
        print("="*50 + "\n")


# Global settings instance
settings = Settings()
