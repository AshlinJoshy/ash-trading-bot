"""
News Fetcher - Collects news articles and market sentiment data.
Aggregates from multiple sources for comprehensive market analysis.
"""
import feedparser
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from bs4 import BeautifulSoup
from loguru import logger
import time

import sys
sys.path.insert(0, '/workspace')
from config.settings import settings


@dataclass
class NewsArticle:
    """Represents a news article."""
    title: str
    summary: str
    source: str
    url: str
    published: Optional[datetime]
    symbols: List[str]  # Related stock symbols
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "url": self.url,
            "published": self.published.isoformat() if self.published else None,
            "symbols": self.symbols,
        }


class NewsFetcher:
    """Fetches news from multiple sources for market analysis."""
    
    def __init__(self):
        self.news_api_key = settings.news.news_api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_rss_news(self, max_articles: int = 20) -> List[NewsArticle]:
        """Fetch news from RSS feeds."""
        articles = []
        
        rss_feeds = {
            "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
            "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
            "Reuters Business": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        }
        
        for source, url in rss_feeds.items():
            try:
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:max_articles // len(rss_feeds)]:
                    # Parse published date
                    published = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6])
                    
                    # Extract summary
                    summary = ""
                    if hasattr(entry, 'summary'):
                        # Clean HTML from summary
                        soup = BeautifulSoup(entry.summary, 'html.parser')
                        summary = soup.get_text()[:500]
                    
                    article = NewsArticle(
                        title=entry.get('title', ''),
                        summary=summary,
                        source=source,
                        url=entry.get('link', ''),
                        published=published,
                        symbols=self._extract_symbols(entry.get('title', '') + ' ' + summary)
                    )
                    articles.append(article)
                    
                logger.debug(f"Fetched {len(feed.entries)} articles from {source}")
                
            except Exception as e:
                logger.warning(f"Error fetching RSS from {source}: {e}")
        
        return articles
    
    def fetch_newsapi_headlines(
        self,
        query: str = "stock market OR trading OR earnings",
        max_articles: int = 20
    ) -> List[NewsArticle]:
        """Fetch news from NewsAPI (requires API key)."""
        if not self.news_api_key:
            logger.debug("NewsAPI key not configured, skipping")
            return []
        
        articles = []
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "apiKey": self.news_api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_articles,
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for item in data.get("articles", []):
                published = None
                if item.get("publishedAt"):
                    try:
                        published = datetime.fromisoformat(
                            item["publishedAt"].replace("Z", "+00:00")
                        )
                    except:
                        pass
                
                article = NewsArticle(
                    title=item.get("title", ""),
                    summary=item.get("description", "")[:500],
                    source=item.get("source", {}).get("name", "NewsAPI"),
                    url=item.get("url", ""),
                    published=published,
                    symbols=self._extract_symbols(
                        (item.get("title", "") + " " + item.get("description", ""))
                    )
                )
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} articles from NewsAPI")
            
        except Exception as e:
            logger.warning(f"Error fetching from NewsAPI: {e}")
        
        return articles
    
    def fetch_symbol_news(self, symbol: str) -> List[NewsArticle]:
        """Fetch news specific to a stock symbol using yfinance."""
        articles = []
        
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            for item in news[:10]:
                published = None
                if item.get("providerPublishTime"):
                    published = datetime.fromtimestamp(item["providerPublishTime"])
                
                article = NewsArticle(
                    title=item.get("title", ""),
                    summary=item.get("title", ""),  # yfinance doesn't provide summary
                    source=item.get("publisher", ""),
                    url=item.get("link", ""),
                    published=published,
                    symbols=[symbol]
                )
                articles.append(article)
            
            logger.debug(f"Fetched {len(articles)} news items for {symbol}")
            
        except Exception as e:
            logger.warning(f"Error fetching news for {symbol}: {e}")
        
        return articles
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols mentioned in text."""
        # Common symbols to look for
        common_symbols = [
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "FB", "TSLA",
            "NVDA", "AMD", "INTC", "SPY", "QQQ", "DIA", "IWM", "VTI",
            "JPM", "BAC", "GS", "MS", "WFC", "C", "V", "MA",
            "JNJ", "PFE", "UNH", "MRK", "ABBV", "BMY",
            "XOM", "CVX", "COP", "SLB", "OXY",
            "DIS", "NFLX", "CMCSA", "T", "VZ",
            "HD", "LOW", "TGT", "WMT", "COST", "AMZN",
            "BA", "LMT", "RTX", "GE", "CAT",
        ]
        
        found = []
        text_upper = text.upper()
        
        for symbol in common_symbols:
            # Look for symbol as whole word
            if f" {symbol} " in f" {text_upper} " or \
               f"${symbol}" in text_upper or \
               f"({symbol})" in text_upper:
                found.append(symbol)
        
        return list(set(found))
    
    def fetch_all_news(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Fetch comprehensive news from all sources.
        
        Returns:
            Dictionary with:
            - general_news: Market-wide news
            - symbol_news: News specific to watched symbols
            - timestamp: When the data was fetched
        """
        if symbols is None:
            symbols = settings.trading.watch_symbols
        
        result = {
            "general_news": [],
            "symbol_news": {},
            "timestamp": datetime.now().isoformat(),
        }
        
        # Fetch general market news
        logger.info("Fetching general market news...")
        rss_news = self.fetch_rss_news(max_articles=30)
        api_news = self.fetch_newsapi_headlines(max_articles=20)
        
        result["general_news"] = [a.to_dict() for a in (rss_news + api_news)]
        
        # Fetch symbol-specific news
        logger.info(f"Fetching news for {len(symbols)} symbols...")
        for symbol in symbols:
            symbol_articles = self.fetch_symbol_news(symbol)
            if symbol_articles:
                result["symbol_news"][symbol] = [a.to_dict() for a in symbol_articles]
            time.sleep(0.2)  # Rate limiting
        
        total_articles = len(result["general_news"]) + sum(
            len(v) for v in result["symbol_news"].values()
        )
        logger.info(f"Total news articles fetched: {total_articles}")
        
        return result


def fetch_premarket_news(symbols: List[str] = None) -> Dict[str, Any]:
    """
    Convenience function to fetch all news for pre-market analysis.
    """
    fetcher = NewsFetcher()
    return fetcher.fetch_all_news(symbols)
