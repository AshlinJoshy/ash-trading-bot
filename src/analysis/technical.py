"""
Technical Analysis Module - Analyzes price patterns, support/resistance, and indicators.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from loguru import logger

# Technical Analysis library
try:
    import ta
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("ta library not available, some features limited")


@dataclass
class SupportResistance:
    """Support and resistance level."""
    price: float
    strength: int  # Number of times price touched this level
    level_type: str  # 'support' or 'resistance'
    
    def to_dict(self) -> Dict:
        return {
            "price": round(self.price, 2),
            "strength": self.strength,
            "type": self.level_type
        }


@dataclass
class CandlePattern:
    """Detected candlestick pattern."""
    name: str
    signal: str  # 'bullish', 'bearish', or 'neutral'
    confidence: float  # 0-1
    description: str
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "signal": self.signal,
            "confidence": self.confidence,
            "description": self.description
        }


@dataclass
class TechnicalSignal:
    """Overall technical signal summary."""
    trend: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0-1
    support_levels: List[SupportResistance]
    resistance_levels: List[SupportResistance]
    patterns: List[CandlePattern]
    indicators: Dict[str, Any]
    recommendation: str
    
    def to_dict(self) -> Dict:
        return {
            "trend": self.trend,
            "strength": round(self.strength, 2),
            "support_levels": [s.to_dict() for s in self.support_levels],
            "resistance_levels": [r.to_dict() for r in self.resistance_levels],
            "patterns": [p.to_dict() for p in self.patterns],
            "indicators": self.indicators,
            "recommendation": self.recommendation
        }


class TechnicalAnalyzer:
    """Performs technical analysis on price data."""
    
    def __init__(self):
        self.lookback_period = 20  # Default lookback for calculations
    
    def analyze(self, df: pd.DataFrame) -> TechnicalSignal:
        """
        Perform comprehensive technical analysis on price data.
        
        Args:
            df: DataFrame with columns: open, high, low, close, volume
        
        Returns:
            TechnicalSignal with analysis results
        """
        if df.empty or len(df) < 20:
            logger.warning("Insufficient data for technical analysis")
            return self._empty_signal()
        
        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Calculate all indicators
        indicators = self._calculate_indicators(df)
        
        # Find support and resistance levels
        support_levels = self._find_support_levels(df)
        resistance_levels = self._find_resistance_levels(df)
        
        # Detect candlestick patterns
        patterns = self._detect_patterns(df)
        
        # Determine overall trend
        trend, strength = self._determine_trend(df, indicators)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            trend, strength, indicators, patterns
        )
        
        return TechnicalSignal(
            trend=trend,
            strength=strength,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            patterns=patterns,
            indicators=indicators,
            recommendation=recommendation
        )
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators."""
        indicators = {}
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume'] if 'volume' in df.columns else None
        
        # Price info
        current_price = close.iloc[-1]
        indicators['current_price'] = round(current_price, 2)
        indicators['price_change_1d'] = round(
            ((current_price / close.iloc[-2]) - 1) * 100, 2
        ) if len(close) > 1 else 0
        
        if TA_AVAILABLE:
            # Moving Averages
            sma_20 = SMAIndicator(close, window=20).sma_indicator()
            sma_50 = SMAIndicator(close, window=50).sma_indicator() if len(close) >= 50 else None
            ema_12 = EMAIndicator(close, window=12).ema_indicator()
            ema_26 = EMAIndicator(close, window=26).ema_indicator()
            
            indicators['sma_20'] = round(sma_20.iloc[-1], 2) if not sma_20.empty else None
            indicators['sma_50'] = round(sma_50.iloc[-1], 2) if sma_50 is not None and not sma_50.empty else None
            indicators['ema_12'] = round(ema_12.iloc[-1], 2) if not ema_12.empty else None
            indicators['ema_26'] = round(ema_26.iloc[-1], 2) if not ema_26.empty else None
            
            # RSI
            rsi = RSIIndicator(close, window=14).rsi()
            indicators['rsi'] = round(rsi.iloc[-1], 2) if not rsi.empty else None
            
            # MACD
            macd = MACD(close)
            indicators['macd'] = round(macd.macd().iloc[-1], 4) if not macd.macd().empty else None
            indicators['macd_signal'] = round(macd.macd_signal().iloc[-1], 4) if not macd.macd_signal().empty else None
            indicators['macd_histogram'] = round(macd.macd_diff().iloc[-1], 4) if not macd.macd_diff().empty else None
            
            # Bollinger Bands
            bb = BollingerBands(close)
            indicators['bb_upper'] = round(bb.bollinger_hband().iloc[-1], 2) if not bb.bollinger_hband().empty else None
            indicators['bb_lower'] = round(bb.bollinger_lband().iloc[-1], 2) if not bb.bollinger_lband().empty else None
            indicators['bb_middle'] = round(bb.bollinger_mavg().iloc[-1], 2) if not bb.bollinger_mavg().empty else None
            
            # ATR (Average True Range) - for volatility
            atr = AverageTrueRange(high, low, close)
            indicators['atr'] = round(atr.average_true_range().iloc[-1], 2) if not atr.average_true_range().empty else None
            indicators['atr_percent'] = round(
                (indicators['atr'] / current_price) * 100, 2
            ) if indicators['atr'] else None
            
            # Stochastic
            stoch = StochasticOscillator(high, low, close)
            indicators['stoch_k'] = round(stoch.stoch().iloc[-1], 2) if not stoch.stoch().empty else None
            indicators['stoch_d'] = round(stoch.stoch_signal().iloc[-1], 2) if not stoch.stoch_signal().empty else None
            
            # ADX (trend strength)
            if len(df) >= 14:
                adx = ADXIndicator(high, low, close)
                indicators['adx'] = round(adx.adx().iloc[-1], 2) if not adx.adx().empty else None
            
            # Volume indicators
            if volume is not None and not volume.empty:
                indicators['volume_avg_20'] = int(volume.tail(20).mean())
                indicators['volume_ratio'] = round(
                    volume.iloc[-1] / volume.tail(20).mean(), 2
                )
                
                obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
                indicators['obv_trend'] = 'up' if obv.iloc[-1] > obv.iloc[-5] else 'down'
        
        else:
            # Fallback calculations without ta library
            indicators['sma_20'] = round(close.tail(20).mean(), 2)
            indicators['sma_50'] = round(close.tail(50).mean(), 2) if len(close) >= 50 else None
            
            # Simple RSI calculation
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = round(rsi.iloc[-1], 2) if not rsi.empty else None
        
        # Price position relative to moving averages
        if indicators.get('sma_20'):
            indicators['above_sma_20'] = current_price > indicators['sma_20']
        if indicators.get('sma_50'):
            indicators['above_sma_50'] = current_price > indicators['sma_50']
        
        return indicators
    
    def _find_support_levels(self, df: pd.DataFrame, num_levels: int = 3) -> List[SupportResistance]:
        """Find support levels using price pivots."""
        lows = df['low'].values
        support_levels = []
        
        # Find local minima
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_levels.append(lows[i])
        
        if not support_levels:
            # Fallback: use recent lows
            support_levels = sorted(lows)[:5]
        
        # Cluster nearby levels
        clustered = self._cluster_levels(support_levels)
        
        # Sort by proximity to current price and take top N
        current_price = df['close'].iloc[-1]
        clustered = sorted(clustered, key=lambda x: abs(x[0] - current_price))
        
        result = []
        for price, strength in clustered[:num_levels]:
            if price < current_price:  # Only support below current price
                result.append(SupportResistance(
                    price=price,
                    strength=strength,
                    level_type='support'
                ))
        
        return result
    
    def _find_resistance_levels(self, df: pd.DataFrame, num_levels: int = 3) -> List[SupportResistance]:
        """Find resistance levels using price pivots."""
        highs = df['high'].values
        resistance_levels = []
        
        # Find local maxima
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_levels.append(highs[i])
        
        if not resistance_levels:
            # Fallback: use recent highs
            resistance_levels = sorted(highs, reverse=True)[:5]
        
        # Cluster nearby levels
        clustered = self._cluster_levels(resistance_levels)
        
        # Sort by proximity to current price and take top N
        current_price = df['close'].iloc[-1]
        clustered = sorted(clustered, key=lambda x: abs(x[0] - current_price))
        
        result = []
        for price, strength in clustered[:num_levels]:
            if price > current_price:  # Only resistance above current price
                result.append(SupportResistance(
                    price=price,
                    strength=strength,
                    level_type='resistance'
                ))
        
        return result
    
    def _cluster_levels(self, levels: List[float], tolerance: float = 0.02) -> List[Tuple[float, int]]:
        """Cluster nearby price levels together."""
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if (level - current_cluster[0]) / current_cluster[0] < tolerance:
                current_cluster.append(level)
            else:
                # Save current cluster
                avg_price = sum(current_cluster) / len(current_cluster)
                clusters.append((avg_price, len(current_cluster)))
                current_cluster = [level]
        
        # Don't forget last cluster
        avg_price = sum(current_cluster) / len(current_cluster)
        clusters.append((avg_price, len(current_cluster)))
        
        return clusters
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[CandlePattern]:
        """Detect candlestick patterns in recent price action."""
        patterns = []
        
        if len(df) < 5:
            return patterns
        
        # Get last few candles
        recent = df.tail(5)
        
        # Calculate candle properties
        opens = recent['open'].values
        closes = recent['close'].values
        highs = recent['high'].values
        lows = recent['low'].values
        
        body_sizes = np.abs(closes - opens)
        candle_ranges = highs - lows
        
        # Last candle properties
        last_open = opens[-1]
        last_close = closes[-1]
        last_high = highs[-1]
        last_low = lows[-1]
        last_body = abs(last_close - last_open)
        last_range = last_high - last_low
        
        is_bullish = last_close > last_open
        upper_wick = last_high - max(last_open, last_close)
        lower_wick = min(last_open, last_close) - last_low
        
        # Doji
        if last_body < 0.1 * last_range and last_range > 0:
            patterns.append(CandlePattern(
                name="Doji",
                signal="neutral",
                confidence=0.7,
                description="Indecision pattern - potential reversal"
            ))
        
        # Hammer (bullish reversal)
        if lower_wick > 2 * last_body and upper_wick < 0.3 * last_body:
            patterns.append(CandlePattern(
                name="Hammer",
                signal="bullish",
                confidence=0.65,
                description="Potential bullish reversal - buyers stepping in"
            ))
        
        # Shooting Star (bearish reversal)
        if upper_wick > 2 * last_body and lower_wick < 0.3 * last_body:
            patterns.append(CandlePattern(
                name="Shooting Star",
                signal="bearish",
                confidence=0.65,
                description="Potential bearish reversal - sellers taking control"
            ))
        
        # Engulfing patterns (need at least 2 candles)
        if len(recent) >= 2:
            prev_open = opens[-2]
            prev_close = closes[-2]
            prev_body = abs(prev_close - prev_open)
            
            # Bullish engulfing
            if prev_close < prev_open and is_bullish and \
               last_body > prev_body and last_open < prev_close and last_close > prev_open:
                patterns.append(CandlePattern(
                    name="Bullish Engulfing",
                    signal="bullish",
                    confidence=0.75,
                    description="Strong bullish reversal signal"
                ))
            
            # Bearish engulfing
            if prev_close > prev_open and not is_bullish and \
               last_body > prev_body and last_open > prev_close and last_close < prev_open:
                patterns.append(CandlePattern(
                    name="Bearish Engulfing",
                    signal="bearish",
                    confidence=0.75,
                    description="Strong bearish reversal signal"
                ))
        
        # Morning/Evening Star (need 3 candles)
        if len(recent) >= 3:
            first_close = closes[-3]
            first_open = opens[-3]
            second_body = body_sizes[-2]
            avg_body = body_sizes.mean()
            
            # Morning Star (bullish)
            if first_close < first_open and second_body < 0.3 * avg_body and is_bullish:
                patterns.append(CandlePattern(
                    name="Morning Star",
                    signal="bullish",
                    confidence=0.7,
                    description="Three-candle bullish reversal pattern"
                ))
            
            # Evening Star (bearish)
            if first_close > first_open and second_body < 0.3 * avg_body and not is_bullish:
                patterns.append(CandlePattern(
                    name="Evening Star",
                    signal="bearish",
                    confidence=0.7,
                    description="Three-candle bearish reversal pattern"
                ))
        
        # Three consecutive candles
        if len(recent) >= 3:
            last_three_bullish = all(closes[-3:] > opens[-3:])
            last_three_bearish = all(closes[-3:] < opens[-3:])
            
            if last_three_bullish:
                patterns.append(CandlePattern(
                    name="Three White Soldiers",
                    signal="bullish",
                    confidence=0.7,
                    description="Strong bullish continuation"
                ))
            elif last_three_bearish:
                patterns.append(CandlePattern(
                    name="Three Black Crows",
                    signal="bearish",
                    confidence=0.7,
                    description="Strong bearish continuation"
                ))
        
        return patterns
    
    def _determine_trend(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Determine overall trend direction and strength."""
        signals = []
        
        close = df['close']
        current_price = close.iloc[-1]
        
        # Price vs moving averages
        if indicators.get('sma_20'):
            if current_price > indicators['sma_20']:
                signals.append(1)
            else:
                signals.append(-1)
        
        if indicators.get('sma_50'):
            if current_price > indicators['sma_50']:
                signals.append(1)
            else:
                signals.append(-1)
        
        # RSI signals
        rsi = indicators.get('rsi')
        if rsi:
            if rsi > 70:
                signals.append(-0.5)  # Overbought
            elif rsi < 30:
                signals.append(0.5)  # Oversold
            elif rsi > 50:
                signals.append(0.5)
            else:
                signals.append(-0.5)
        
        # MACD signal
        macd_hist = indicators.get('macd_histogram')
        if macd_hist:
            if macd_hist > 0:
                signals.append(1)
            else:
                signals.append(-1)
        
        # Price momentum (last 5 days)
        if len(close) >= 5:
            momentum = (current_price - close.iloc[-5]) / close.iloc[-5]
            if momentum > 0.02:
                signals.append(1)
            elif momentum < -0.02:
                signals.append(-1)
        
        # Calculate overall signal
        if not signals:
            return "neutral", 0.5
        
        avg_signal = sum(signals) / len(signals)
        
        if avg_signal > 0.3:
            trend = "bullish"
        elif avg_signal < -0.3:
            trend = "bearish"
        else:
            trend = "neutral"
        
        strength = min(abs(avg_signal), 1.0)
        
        return trend, strength
    
    def _generate_recommendation(
        self,
        trend: str,
        strength: float,
        indicators: Dict[str, Any],
        patterns: List[CandlePattern]
    ) -> str:
        """Generate trading recommendation based on analysis."""
        rsi = indicators.get('rsi', 50)
        
        # Count pattern signals
        bullish_patterns = sum(1 for p in patterns if p.signal == 'bullish')
        bearish_patterns = sum(1 for p in patterns if p.signal == 'bearish')
        
        # Strong signals
        if trend == "bullish" and strength > 0.6 and rsi < 70:
            if bullish_patterns > 0:
                return "STRONG BUY - Bullish trend with confirming patterns"
            return "BUY - Bullish trend confirmed by indicators"
        
        if trend == "bearish" and strength > 0.6 and rsi > 30:
            if bearish_patterns > 0:
                return "STRONG SELL - Bearish trend with confirming patterns"
            return "SELL - Bearish trend confirmed by indicators"
        
        # Moderate signals
        if trend == "bullish" and strength > 0.4:
            return "LEAN BULLISH - Consider buying on pullbacks"
        
        if trend == "bearish" and strength > 0.4:
            return "LEAN BEARISH - Consider reducing exposure"
        
        # Overbought/Oversold conditions
        if rsi > 75:
            return "CAUTION - Overbought conditions, wait for pullback"
        if rsi < 25:
            return "WATCH - Oversold conditions, potential bounce"
        
        # Neutral
        return "NEUTRAL - No clear direction, wait for confirmation"
    
    def _empty_signal(self) -> TechnicalSignal:
        """Return empty signal for insufficient data."""
        return TechnicalSignal(
            trend="neutral",
            strength=0,
            support_levels=[],
            resistance_levels=[],
            patterns=[],
            indicators={},
            recommendation="INSUFFICIENT DATA"
        )


def analyze_symbol(symbol: str, period: str = "3mo") -> Dict[str, Any]:
    """
    Convenience function to analyze a single symbol.
    
    Args:
        symbol: Stock ticker
        period: Historical data period
    
    Returns:
        Technical analysis results as dictionary
    """
    from src.data.market_data import MarketDataFetcher
    
    fetcher = MarketDataFetcher()
    df = fetcher.get_historical_data(symbol, period=period, interval="1d")
    
    if df.empty:
        return {"error": f"No data available for {symbol}"}
    
    analyzer = TechnicalAnalyzer()
    signal = analyzer.analyze(df)
    
    return {
        "symbol": symbol,
        "analysis": signal.to_dict()
    }
