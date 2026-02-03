"""
Sentiment Analysis Module - Analyzes news and text for market sentiment.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("textblob not available, sentiment analysis limited")


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    text: str
    polarity: float  # -1 (negative) to 1 (positive)
    subjectivity: float  # 0 (objective) to 1 (subjective)
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            "polarity": round(self.polarity, 3),
            "subjectivity": round(self.subjectivity, 3),
            "sentiment": self.sentiment,
            "confidence": round(self.confidence, 2)
        }


@dataclass
class MarketSentiment:
    """Aggregated market sentiment from multiple sources."""
    overall_sentiment: str
    overall_score: float
    bullish_count: int
    bearish_count: int
    neutral_count: int
    key_themes: List[str]
    sentiment_by_symbol: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            "overall_sentiment": self.overall_sentiment,
            "overall_score": round(self.overall_score, 3),
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "key_themes": self.key_themes,
            "sentiment_by_symbol": {
                k: round(v, 3) for k, v in self.sentiment_by_symbol.items()
            }
        }


class SentimentAnalyzer:
    """Analyzes text sentiment for trading signals."""
    
    # Keywords that often indicate market direction
    BULLISH_KEYWORDS = [
        'surge', 'soar', 'rally', 'gain', 'jump', 'rise', 'climb', 'boom',
        'breakthrough', 'upgrade', 'beat', 'exceed', 'growth', 'profit',
        'bullish', 'optimistic', 'strong', 'outperform', 'buy', 'positive',
        'record', 'high', 'upside', 'momentum', 'expansion'
    ]
    
    BEARISH_KEYWORDS = [
        'fall', 'drop', 'plunge', 'decline', 'crash', 'tumble', 'sink',
        'downgrade', 'miss', 'loss', 'weak', 'bearish', 'pessimistic',
        'sell', 'negative', 'concern', 'worry', 'risk', 'fear', 'warning',
        'recession', 'slowdown', 'layoff', 'cut', 'downside'
    ]
    
    def __init__(self):
        pass
    
    def analyze_text(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
        
        Returns:
            SentimentResult with sentiment metrics
        """
        if not text:
            return SentimentResult(
                text="",
                polarity=0,
                subjectivity=0,
                sentiment="neutral",
                confidence=0
            )
        
        # Clean text
        text = text.strip()
        
        # TextBlob analysis
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        else:
            # Fallback: keyword-based analysis
            polarity = self._keyword_sentiment(text)
            subjectivity = 0.5
        
        # Apply keyword boosting
        keyword_boost = self._keyword_sentiment(text)
        
        # Combine TextBlob and keyword analysis
        final_polarity = (polarity * 0.6) + (keyword_boost * 0.4)
        
        # Determine sentiment category
        if final_polarity > 0.1:
            sentiment = "positive"
        elif final_polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Confidence based on how strong the signal is
        confidence = min(abs(final_polarity) * 2, 1.0)
        
        return SentimentResult(
            text=text[:200],  # Truncate for storage
            polarity=final_polarity,
            subjectivity=subjectivity,
            sentiment=sentiment,
            confidence=confidence
        )
    
    def _keyword_sentiment(self, text: str) -> float:
        """Calculate sentiment based on keyword presence."""
        text_lower = text.lower()
        
        bullish_count = sum(1 for word in self.BULLISH_KEYWORDS if word in text_lower)
        bearish_count = sum(1 for word in self.BEARISH_KEYWORDS if word in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0
        
        return (bullish_count - bearish_count) / total
    
    def analyze_news_batch(self, news_items: List[Dict]) -> MarketSentiment:
        """
        Analyze sentiment from a batch of news articles.
        
        Args:
            news_items: List of news items with 'title' and 'summary' keys
        
        Returns:
            MarketSentiment with aggregated analysis
        """
        if not news_items:
            return self._empty_sentiment()
        
        sentiments = []
        symbol_sentiments: Dict[str, List[float]] = {}
        
        for item in news_items:
            # Combine title and summary for analysis
            text = f"{item.get('title', '')} {item.get('summary', '')}"
            result = self.analyze_text(text)
            sentiments.append(result)
            
            # Track sentiment by symbol
            symbols = item.get('symbols', [])
            for symbol in symbols:
                if symbol not in symbol_sentiments:
                    symbol_sentiments[symbol] = []
                symbol_sentiments[symbol].append(result.polarity)
        
        # Calculate aggregates
        polarities = [s.polarity for s in sentiments]
        avg_polarity = sum(polarities) / len(polarities) if polarities else 0
        
        bullish_count = sum(1 for s in sentiments if s.sentiment == 'positive')
        bearish_count = sum(1 for s in sentiments if s.sentiment == 'negative')
        neutral_count = sum(1 for s in sentiments if s.sentiment == 'neutral')
        
        # Determine overall sentiment
        if avg_polarity > 0.1:
            overall = "bullish"
        elif avg_polarity < -0.1:
            overall = "bearish"
        else:
            overall = "neutral"
        
        # Calculate per-symbol sentiment
        symbol_avg = {
            symbol: sum(scores) / len(scores)
            for symbol, scores in symbol_sentiments.items()
            if scores
        }
        
        # Extract key themes
        key_themes = self._extract_themes(news_items)
        
        return MarketSentiment(
            overall_sentiment=overall,
            overall_score=avg_polarity,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            key_themes=key_themes,
            sentiment_by_symbol=symbol_avg
        )
    
    def _extract_themes(self, news_items: List[Dict], max_themes: int = 5) -> List[str]:
        """Extract key themes/topics from news."""
        # Simple keyword extraction
        theme_keywords = {
            'earnings': ['earnings', 'profit', 'revenue', 'quarterly', 'eps'],
            'fed': ['fed', 'federal reserve', 'interest rate', 'powell', 'fomc'],
            'inflation': ['inflation', 'cpi', 'price', 'cost'],
            'jobs': ['jobs', 'employment', 'unemployment', 'hiring', 'layoff'],
            'tech': ['tech', 'ai', 'artificial intelligence', 'software', 'chip'],
            'energy': ['oil', 'gas', 'energy', 'opec', 'crude'],
            'china': ['china', 'chinese', 'beijing', 'trade war'],
            'crypto': ['bitcoin', 'crypto', 'cryptocurrency', 'ethereum'],
            'banking': ['bank', 'banking', 'financial', 'credit'],
            'housing': ['housing', 'real estate', 'mortgage', 'home'],
        }
        
        all_text = ' '.join([
            f"{item.get('title', '')} {item.get('summary', '')}"
            for item in news_items
        ]).lower()
        
        theme_scores = {}
        for theme, keywords in theme_keywords.items():
            score = sum(all_text.count(kw) for kw in keywords)
            if score > 0:
                theme_scores[theme] = score
        
        # Return top themes
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, _ in sorted_themes[:max_themes]]
    
    def _empty_sentiment(self) -> MarketSentiment:
        """Return empty sentiment for no data."""
        return MarketSentiment(
            overall_sentiment="neutral",
            overall_score=0,
            bullish_count=0,
            bearish_count=0,
            neutral_count=0,
            key_themes=[],
            sentiment_by_symbol={}
        )
    
    def get_trading_signal(self, sentiment: MarketSentiment) -> Dict[str, Any]:
        """
        Convert sentiment into actionable trading signal.
        
        Args:
            sentiment: MarketSentiment result
        
        Returns:
            Trading signal with confidence and recommendation
        """
        score = sentiment.overall_score
        total_articles = (
            sentiment.bullish_count +
            sentiment.bearish_count +
            sentiment.neutral_count
        )
        
        # Need enough data for confidence
        if total_articles < 5:
            confidence = 0.3
        elif total_articles < 15:
            confidence = 0.5
        else:
            confidence = 0.7
        
        # Adjust confidence by sentiment strength
        confidence *= min(abs(score) * 3, 1.0)
        
        # Generate signal
        if score > 0.2 and confidence > 0.4:
            signal = "BUY"
            reason = "Strong positive news sentiment"
        elif score < -0.2 and confidence > 0.4:
            signal = "SELL"
            reason = "Strong negative news sentiment"
        elif score > 0.1:
            signal = "LEAN_BULLISH"
            reason = "Moderately positive sentiment"
        elif score < -0.1:
            signal = "LEAN_BEARISH"
            reason = "Moderately negative sentiment"
        else:
            signal = "NEUTRAL"
            reason = "Mixed or neutral sentiment"
        
        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "reason": reason,
            "sentiment_score": round(score, 3),
            "articles_analyzed": total_articles,
            "key_themes": sentiment.key_themes
        }


def analyze_market_sentiment(news_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to analyze market sentiment from news data.
    
    Args:
        news_data: Output from NewsFetcher.fetch_all_news()
    
    Returns:
        Comprehensive sentiment analysis
    """
    analyzer = SentimentAnalyzer()
    
    # Analyze general news
    general_sentiment = analyzer.analyze_news_batch(news_data.get("general_news", []))
    
    # Analyze per-symbol news
    symbol_analysis = {}
    for symbol, articles in news_data.get("symbol_news", {}).items():
        sentiment = analyzer.analyze_news_batch(articles)
        signal = analyzer.get_trading_signal(sentiment)
        symbol_analysis[symbol] = {
            "sentiment": sentiment.to_dict(),
            "signal": signal
        }
    
    # Overall trading signal
    overall_signal = analyzer.get_trading_signal(general_sentiment)
    
    return {
        "overall": {
            "sentiment": general_sentiment.to_dict(),
            "signal": overall_signal
        },
        "by_symbol": symbol_analysis,
        "timestamp": news_data.get("timestamp")
    }
