"""
Gemini Trader - Trading bot powered by Google's Gemini.
"""
from typing import Dict, List, Any
from loguru import logger

import sys
sys.path.insert(0, '/workspace')
from config.settings import settings
from src.ai_bots.base_bot import BaseTrader, TradeDecision


class GeminiTrader(BaseTrader):
    """
    Trading bot that uses Gemini (Google) for analysis and decisions.
    """
    
    def __init__(self):
        super().__init__(name="Gemini Trader")
        self.model = None
        self.model_name = settings.ai.gemini_model
        self._init_client()
    
    def _init_client(self):
        """Initialize the Google Generative AI client."""
        if not settings.ai.google_api_key:
            logger.warning("Google API key not configured")
            return
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=settings.ai.google_api_key)
            
            # Configure the model with safety settings and generation config
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config
            )
            
            logger.info(f"Gemini client initialized with model: {self.model_name}")
            
        except ImportError:
            logger.error("google-generativeai package not installed")
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
    
    def analyze_and_decide(
        self,
        market_data: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        sentiment_analysis: Dict[str, Any],
        current_positions: List[Dict[str, Any]] = None,
        portfolio_value: float = 100000
    ) -> List[TradeDecision]:
        """
        Use Gemini to analyze market data and make trading decisions.
        """
        if not self.model:
            logger.error("Gemini model not initialized")
            return []
        
        # Build the analysis prompt
        prompt = self._build_analysis_prompt(
            market_data=market_data,
            technical_analysis=technical_analysis,
            sentiment_analysis=sentiment_analysis,
            current_positions=current_positions,
            portfolio_value=portfolio_value
        )
        
        # Add system context to prompt for Gemini
        full_prompt = """You are an expert algorithmic day trader with deep knowledge of:
- Technical analysis (candlestick patterns, support/resistance, indicators)
- Fundamental analysis (news sentiment, market conditions)
- Risk management (position sizing, stop losses)

You make conservative, well-reasoned trading decisions. You prioritize capital preservation
and only trade when there's a clear edge. You always provide your reasoning and never
recommend trades you're not confident about.

When providing trade recommendations, respond with valid JSON only.

---

""" + prompt

        try:
            # Call Gemini API
            logger.info("Requesting analysis from Gemini...")
            
            response = self.model.generate_content(full_prompt)
            
            # Extract response text
            response_text = response.text
            logger.debug(f"Gemini response: {response_text[:500]}...")
            
            # Parse decisions from response
            decisions = self._parse_decisions(response_text)
            
            logger.info(f"Gemini generated {len(decisions)} trading decisions")
            return decisions
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return []
    
    def get_market_commentary(
        self,
        market_data: Dict[str, Any],
        sentiment_analysis: Dict[str, Any]
    ) -> str:
        """
        Get Gemini's commentary on overall market conditions.
        """
        if not self.model:
            return "Gemini model not initialized"
        
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
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error getting market commentary: {e}")
            return f"Error: {e}"


def create_gemini_trader() -> GeminiTrader:
    """Factory function to create a Gemini trader instance."""
    return GeminiTrader()
