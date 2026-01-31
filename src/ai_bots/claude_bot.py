"""
Claude Trader - Trading bot powered by Anthropic's Claude.
"""
from typing import Dict, List, Any
from loguru import logger

import sys
sys.path.insert(0, '/workspace')
from config.settings import settings
from src.ai_bots.base_bot import BaseTrader, TradeDecision


class ClaudeTrader(BaseTrader):
    """
    Trading bot that uses Claude (Anthropic) for analysis and decisions.
    """
    
    def __init__(self):
        super().__init__(name="Claude Trader")
        self.client = None
        self.model = settings.ai.claude_model
        self._init_client()
    
    def _init_client(self):
        """Initialize the Anthropic client."""
        if not settings.ai.anthropic_api_key:
            logger.warning("Anthropic API key not configured")
            return
        
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=settings.ai.anthropic_api_key)
            logger.info(f"Claude client initialized with model: {self.model}")
        except ImportError:
            logger.error("anthropic package not installed")
        except Exception as e:
            logger.error(f"Error initializing Claude client: {e}")
    
    def analyze_and_decide(
        self,
        market_data: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        sentiment_analysis: Dict[str, Any],
        current_positions: List[Dict[str, Any]] = None,
        portfolio_value: float = 100000
    ) -> List[TradeDecision]:
        """
        Use Claude to analyze market data and make trading decisions.
        """
        if not self.client:
            logger.error("Claude client not initialized")
            return []
        
        # Build the analysis prompt
        prompt = self._build_analysis_prompt(
            market_data=market_data,
            technical_analysis=technical_analysis,
            sentiment_analysis=sentiment_analysis,
            current_positions=current_positions,
            portfolio_value=portfolio_value
        )
        
        try:
            # Call Claude API
            logger.info("Requesting analysis from Claude...")
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                system="""You are an expert algorithmic day trader with deep knowledge of:
- Technical analysis (candlestick patterns, support/resistance, indicators)
- Fundamental analysis (news sentiment, market conditions)
- Risk management (position sizing, stop losses)

You make conservative, well-reasoned trading decisions. You prioritize capital preservation
and only trade when there's a clear edge. You always provide your reasoning and never
recommend trades you're not confident about.

Respond with valid JSON only when providing trade recommendations."""
            )
            
            # Extract response text
            response_text = response.content[0].text
            logger.debug(f"Claude response: {response_text[:500]}...")
            
            # Parse decisions from response
            decisions = self._parse_decisions(response_text)
            
            logger.info(f"Claude generated {len(decisions)} trading decisions")
            return decisions
            
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return []
    
    def get_market_commentary(
        self,
        market_data: Dict[str, Any],
        sentiment_analysis: Dict[str, Any]
    ) -> str:
        """
        Get Claude's commentary on overall market conditions.
        """
        if not self.client:
            return "Claude client not initialized"
        
        prompt = f"""Provide a brief market commentary based on the following data:

## Market Data
{market_data}

## Sentiment Analysis
{sentiment_analysis}

Provide a 2-3 paragraph summary of:
1. Current market conditions
2. Key themes and drivers
3. What traders should watch for today

Keep it concise and actionable."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error getting market commentary: {e}")
            return f"Error: {e}"


def create_claude_trader() -> ClaudeTrader:
    """Factory function to create a Claude trader instance."""
    return ClaudeTrader()
