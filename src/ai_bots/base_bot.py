"""
Base Trader - Abstract base class for AI trading bots.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
from loguru import logger
import json

import sys
sys.path.insert(0, '/workspace')
from config.settings import settings


class TradeAction(Enum):
    """Possible trade actions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"  # Close existing position


@dataclass
class TradeDecision:
    """A trading decision made by an AI bot."""
    symbol: str
    action: TradeAction
    confidence: float  # 0-1
    quantity: Optional[int] = None  # None means "calculate optimal size"
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    bot_name: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "confidence": round(self.confidence, 2),
            "quantity": self.quantity,
            "entry_price": round(self.entry_price, 2) if self.entry_price else None,
            "stop_loss": round(self.stop_loss, 2) if self.stop_loss else None,
            "take_profit": round(self.take_profit, 2) if self.take_profit else None,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "bot_name": self.bot_name
        }


class BaseTrader(ABC):
    """
    Abstract base class for AI trading bots.
    
    Both Claude and Gemini traders inherit from this class.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.decisions_history: List[TradeDecision] = []
        self.min_confidence = 0.6  # Minimum confidence to act
        
    @abstractmethod
    def analyze_and_decide(
        self,
        market_data: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        sentiment_analysis: Dict[str, Any],
        current_positions: List[Dict[str, Any]] = None,
        portfolio_value: float = 100000
    ) -> List[TradeDecision]:
        """
        Analyze market data and make trading decisions.
        
        Args:
            market_data: Current market data (prices, quotes)
            technical_analysis: Technical indicators and patterns
            sentiment_analysis: News sentiment analysis
            current_positions: Current open positions
            portfolio_value: Total portfolio value
        
        Returns:
            List of TradeDecision objects
        """
        pass
    
    def _build_analysis_prompt(
        self,
        market_data: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        sentiment_analysis: Dict[str, Any],
        current_positions: List[Dict[str, Any]] = None,
        portfolio_value: float = 100000
    ) -> str:
        """Build the prompt for the AI to analyze."""
        
        prompt = f"""You are an expert day trader AI assistant. Analyze the following market data and provide trading recommendations.

## Current Portfolio
- Portfolio Value: ${portfolio_value:,.2f}
- Max Position Size: {settings.trading.max_position_size_pct}% of portfolio
- Max Daily Loss: {settings.trading.max_daily_loss_pct}%
- Stop Loss: {settings.trading.stop_loss_pct}%
- Take Profit: {settings.trading.take_profit_pct}%
- Max Concurrent Positions: {settings.trading.max_positions}

## Current Positions
{json.dumps(current_positions or [], indent=2)}

## Market Status
{json.dumps(market_data.get('market_status', {}), indent=2)}

## Technical Analysis by Symbol
"""
        # Add technical analysis for each symbol
        for symbol, analysis in technical_analysis.items():
            prompt += f"""
### {symbol}
{json.dumps(analysis, indent=2)}
"""

        prompt += """
## News Sentiment Analysis
"""
        prompt += json.dumps(sentiment_analysis, indent=2)

        prompt += """

## Your Task
Based on the above analysis, provide your trading recommendations. For each recommendation, include:

1. **Symbol**: The stock ticker
2. **Action**: BUY, SELL, HOLD, or CLOSE
3. **Confidence**: Your confidence level (0.0 to 1.0)
4. **Reasoning**: Brief explanation of your decision
5. **Entry Price**: Suggested entry price (if BUY)
6. **Stop Loss**: Suggested stop loss price
7. **Take Profit**: Suggested take profit price

## Rules
- Only recommend trades with confidence >= 0.6
- Prioritize risk management - never risk more than the stop loss percentage
- Consider both technical AND sentiment analysis
- Look for confluence of signals (multiple indicators agreeing)
- Be conservative - when in doubt, HOLD
- Don't overtrade - quality over quantity
- Consider current positions before adding new ones

## Response Format
Respond with a JSON array of trade decisions. Example:
```json
[
    {
        "symbol": "AAPL",
        "action": "BUY",
        "confidence": 0.75,
        "reasoning": "Strong bullish trend, positive sentiment, RSI not overbought",
        "entry_price": 175.50,
        "stop_loss": 172.00,
        "take_profit": 182.00
    }
]
```

If no good opportunities exist, return an empty array: []

Provide your analysis and recommendations now:
"""
        return prompt
    
    def _parse_decisions(self, response_text: str) -> List[TradeDecision]:
        """Parse AI response into TradeDecision objects."""
        try:
            # Extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning(f"No JSON array found in response")
                return []
            
            json_str = response_text[json_start:json_end]
            decisions_data = json.loads(json_str)
            
            decisions = []
            for d in decisions_data:
                try:
                    action = TradeAction[d.get('action', 'HOLD').upper()]
                    
                    decision = TradeDecision(
                        symbol=d.get('symbol', ''),
                        action=action,
                        confidence=float(d.get('confidence', 0)),
                        entry_price=float(d['entry_price']) if d.get('entry_price') else None,
                        stop_loss=float(d['stop_loss']) if d.get('stop_loss') else None,
                        take_profit=float(d['take_profit']) if d.get('take_profit') else None,
                        reasoning=d.get('reasoning', ''),
                        bot_name=self.name
                    )
                    
                    # Only include decisions above minimum confidence
                    if decision.confidence >= self.min_confidence:
                        decisions.append(decision)
                        self.decisions_history.append(decision)
                    
                except Exception as e:
                    logger.warning(f"Error parsing decision: {e}")
                    continue
            
            return decisions
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return []
    
    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss: float
    ) -> int:
        """
        Calculate position size based on risk management rules.
        
        Uses the formula: Position = (Portfolio * Risk%) / (Entry - StopLoss)
        """
        max_position_value = portfolio_value * (settings.trading.max_position_size_pct / 100)
        max_risk = portfolio_value * (settings.trading.stop_loss_pct / 100)
        
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            risk_per_share = entry_price * (settings.trading.stop_loss_pct / 100)
        
        # Calculate shares based on risk
        risk_based_shares = int(max_risk / risk_per_share)
        
        # Calculate shares based on max position size
        size_based_shares = int(max_position_value / entry_price)
        
        # Take the smaller of the two
        return min(risk_based_shares, size_based_shares)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of bot's trading decisions."""
        if not self.decisions_history:
            return {"message": "No decisions made yet"}
        
        buy_count = sum(1 for d in self.decisions_history if d.action == TradeAction.BUY)
        sell_count = sum(1 for d in self.decisions_history if d.action == TradeAction.SELL)
        hold_count = sum(1 for d in self.decisions_history if d.action == TradeAction.HOLD)
        
        avg_confidence = sum(d.confidence for d in self.decisions_history) / len(self.decisions_history)
        
        return {
            "bot_name": self.name,
            "total_decisions": len(self.decisions_history),
            "buy_decisions": buy_count,
            "sell_decisions": sell_count,
            "hold_decisions": hold_count,
            "average_confidence": round(avg_confidence, 2),
            "recent_decisions": [d.to_dict() for d in self.decisions_history[-5:]]
        }
