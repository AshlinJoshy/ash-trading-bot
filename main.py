#!/usr/bin/env python3
"""
Trading AI System - Main Orchestrator

This is the main entry point for running the dual-AI trading system.
It coordinates:
1. Pre-market analysis (news, sentiment)
2. Technical analysis (patterns, support/resistance)
3. AI decision making (Claude and Gemini)
4. Trade execution (on demo account)
5. Performance tracking and comparison

Usage:
    python main.py                  # Run full trading cycle
    python main.py --analyze-only   # Run analysis without trading
    python main.py --status         # Show current status and positions
    python main.py --compare        # Compare bot performance
"""

import sys
import argparse
from datetime import datetime
from typing import Dict, Any
import json

sys.path.insert(0, '/workspace')

from loguru import logger
from config.settings import settings
from src.utils.logger import setup_logging
from src.utils.database import PerformanceTracker
from src.data.market_data import MarketDataFetcher, fetch_data_for_analysis
from src.data.news_fetcher import NewsFetcher, fetch_premarket_news
from src.analysis.technical import TechnicalAnalyzer
from src.analysis.sentiment import SentimentAnalyzer, analyze_market_sentiment
from src.ai_bots.claude_bot import ClaudeTrader
from src.ai_bots.gemini_bot import GeminiTrader
from src.broker.alpaca_broker import AlpacaBroker


class TradingOrchestrator:
    """
    Main orchestrator that coordinates all trading activities.
    """
    
    def __init__(self):
        # Setup logging
        setup_logging(log_level="INFO", log_file="logs/trading.log")
        
        # Initialize components
        logger.info("Initializing Trading AI System...")
        
        self.market_data = MarketDataFetcher()
        self.news_fetcher = NewsFetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.performance_tracker = PerformanceTracker()
        
        # Initialize AI bots
        self.claude_trader = ClaudeTrader()
        self.gemini_trader = GeminiTrader()
        
        # Initialize broker
        self.broker = AlpacaBroker()
        
        # Validate configuration
        self._validate_setup()
    
    def _validate_setup(self):
        """Validate that all required components are configured."""
        logger.info("Validating configuration...")
        settings.print_status()
        
        status = settings.validate()
        
        warnings = []
        if not status["ai_claude"]:
            warnings.append("Claude API not configured - Claude bot will not function")
        if not status["ai_gemini"]:
            warnings.append("Gemini API not configured - Gemini bot will not function")
        if not status["broker"]:
            warnings.append("Broker not configured - Paper trading disabled")
        
        for warning in warnings:
            logger.warning(warning)
        
        if not status["ai_claude"] and not status["ai_gemini"]:
            logger.error("At least one AI API must be configured!")
            return False
        
        return True
    
    def run_premarket_analysis(self) -> Dict[str, Any]:
        """
        Run pre-market analysis including news and sentiment.
        Should be run before market open (e.g., 8:00 AM ET).
        """
        logger.info("="*60)
        logger.info("RUNNING PRE-MARKET ANALYSIS")
        logger.info("="*60)
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "market_status": {},
            "news_data": {},
            "sentiment": {},
            "market_data": {},
            "technical_analysis": {}
        }
        
        # Check market status
        logger.info("Checking market status...")
        analysis["market_status"] = self.market_data.get_market_status()
        
        # Fetch news
        logger.info("Fetching news and market headlines...")
        analysis["news_data"] = self.news_fetcher.fetch_all_news(
            symbols=settings.trading.watch_symbols
        )
        
        # Analyze sentiment
        logger.info("Analyzing market sentiment...")
        analysis["sentiment"] = analyze_market_sentiment(analysis["news_data"])
        
        # Fetch market data
        logger.info("Fetching market data for watched symbols...")
        analysis["market_data"] = fetch_data_for_analysis(
            symbols=settings.trading.watch_symbols
        )
        
        # Run technical analysis
        logger.info("Running technical analysis...")
        for symbol in settings.trading.watch_symbols:
            daily_data = analysis["market_data"].get("daily_data", {}).get(symbol)
            if daily_data is not None and not daily_data.empty:
                tech_signal = self.technical_analyzer.analyze(daily_data)
                analysis["technical_analysis"][symbol] = tech_signal.to_dict()
        
        # Print summary
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _print_analysis_summary(self, analysis: Dict[str, Any]):
        """Print a summary of the analysis."""
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*60)
        
        # Market status
        status = analysis["market_status"]
        logger.info(f"Market Open: {status.get('is_open', 'Unknown')}")
        
        # Sentiment summary
        sentiment = analysis["sentiment"].get("overall", {})
        signal = sentiment.get("signal", {})
        logger.info(f"Overall Sentiment: {signal.get('signal', 'N/A')} "
                   f"(Confidence: {signal.get('confidence', 0):.0%})")
        
        # Technical analysis summary
        logger.info("\nTechnical Analysis by Symbol:")
        for symbol, tech in analysis["technical_analysis"].items():
            trend = tech.get("trend", "N/A")
            rec = tech.get("recommendation", "N/A")
            logger.info(f"  {symbol}: {trend.upper()} - {rec}")
        
        logger.info("="*60)
    
    def get_ai_decisions(self, analysis: Dict[str, Any]) -> Dict[str, list]:
        """
        Get trading decisions from both AI bots.
        """
        logger.info("\n" + "="*60)
        logger.info("GETTING AI TRADING DECISIONS")
        logger.info("="*60)
        
        # Get account info for position sizing
        account = self.broker.get_account_info()
        portfolio_value = account.get("portfolio_value", 100000)
        
        # Get current positions
        positions = self.broker.get_positions()
        positions_dict = [p.to_dict() for p in positions]
        
        decisions = {
            "claude": [],
            "gemini": []
        }
        
        # Prepare data for AI
        market_data = {
            "quotes": analysis["market_data"].get("quotes", {}),
            "market_status": analysis["market_status"]
        }
        
        # Get Claude's decisions
        logger.info("\nAsking Claude for trading decisions...")
        try:
            claude_decisions = self.claude_trader.analyze_and_decide(
                market_data=market_data,
                technical_analysis=analysis["technical_analysis"],
                sentiment_analysis=analysis["sentiment"],
                current_positions=positions_dict,
                portfolio_value=portfolio_value
            )
            decisions["claude"] = claude_decisions
            
            for d in claude_decisions:
                logger.info(f"  Claude: {d.action.value} {d.symbol} "
                           f"(Confidence: {d.confidence:.0%})")
                logger.info(f"    Reason: {d.reasoning[:100]}...")
                
                # Record decision
                self.performance_tracker.record_decision(
                    bot_name="Claude Trader",
                    symbol=d.symbol,
                    action=d.action.value,
                    confidence=d.confidence,
                    reasoning=d.reasoning
                )
        except Exception as e:
            logger.error(f"Error getting Claude decisions: {e}")
        
        # Get Gemini's decisions
        logger.info("\nAsking Gemini for trading decisions...")
        try:
            gemini_decisions = self.gemini_trader.analyze_and_decide(
                market_data=market_data,
                technical_analysis=analysis["technical_analysis"],
                sentiment_analysis=analysis["sentiment"],
                current_positions=positions_dict,
                portfolio_value=portfolio_value
            )
            decisions["gemini"] = gemini_decisions
            
            for d in gemini_decisions:
                logger.info(f"  Gemini: {d.action.value} {d.symbol} "
                           f"(Confidence: {d.confidence:.0%})")
                logger.info(f"    Reason: {d.reasoning[:100]}...")
                
                # Record decision
                self.performance_tracker.record_decision(
                    bot_name="Gemini Trader",
                    symbol=d.symbol,
                    action=d.action.value,
                    confidence=d.confidence,
                    reasoning=d.reasoning
                )
        except Exception as e:
            logger.error(f"Error getting Gemini decisions: {e}")
        
        return decisions
    
    def execute_decisions(
        self,
        decisions: Dict[str, list],
        execute_mode: str = "both"
    ) -> Dict[str, list]:
        """
        Execute trading decisions from AI bots.
        
        Args:
            decisions: Dictionary with 'claude' and 'gemini' decision lists
            execute_mode: 'claude', 'gemini', or 'both'
        """
        logger.info("\n" + "="*60)
        logger.info("EXECUTING TRADES")
        logger.info("="*60)
        
        executed = {"claude": [], "gemini": []}
        
        account = self.broker.get_account_info()
        portfolio_value = account.get("portfolio_value", 100000)
        
        # Execute Claude's decisions
        if execute_mode in ["claude", "both"]:
            for decision in decisions.get("claude", []):
                logger.info(f"Executing Claude decision: {decision.action.value} {decision.symbol}")
                order = self.broker.execute_decision(decision, portfolio_value)
                
                if order:
                    executed["claude"].append(order)
                    
                    # Record trade
                    self.performance_tracker.record_trade_entry(
                        bot_name="Claude Trader",
                        symbol=decision.symbol,
                        action=decision.action.value,
                        quantity=order.quantity,
                        entry_price=decision.entry_price or 0,
                        reasoning=decision.reasoning,
                        confidence=decision.confidence
                    )
        
        # Execute Gemini's decisions
        if execute_mode in ["gemini", "both"]:
            for decision in decisions.get("gemini", []):
                logger.info(f"Executing Gemini decision: {decision.action.value} {decision.symbol}")
                order = self.broker.execute_decision(decision, portfolio_value)
                
                if order:
                    executed["gemini"].append(order)
                    
                    # Record trade
                    self.performance_tracker.record_trade_entry(
                        bot_name="Gemini Trader",
                        symbol=decision.symbol,
                        action=decision.action.value,
                        quantity=order.quantity,
                        entry_price=decision.entry_price or 0,
                        reasoning=decision.reasoning,
                        confidence=decision.confidence
                    )
        
        # Update daily performance
        self.performance_tracker.update_daily_performance("Claude Trader", portfolio_value)
        self.performance_tracker.update_daily_performance("Gemini Trader", portfolio_value)
        
        logger.info(f"\nExecuted {len(executed['claude'])} Claude orders, "
                   f"{len(executed['gemini'])} Gemini orders")
        
        return executed
    
    def run_trading_cycle(self, execute_trades: bool = True):
        """
        Run a complete trading cycle:
        1. Pre-market analysis
        2. Get AI decisions
        3. Execute trades (if enabled)
        4. Track performance
        """
        logger.info("\n" + "#"*60)
        logger.info("# TRADING AI SYSTEM - STARTING TRADING CYCLE")
        logger.info(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("#"*60)
        
        # Run analysis
        analysis = self.run_premarket_analysis()
        
        # Get AI decisions
        decisions = self.get_ai_decisions(analysis)
        
        # Execute trades if enabled
        if execute_trades:
            self.execute_decisions(decisions)
        else:
            logger.info("\nTrade execution disabled - analysis only mode")
        
        # Show comparison
        self.show_performance_comparison()
        
        logger.info("\n" + "#"*60)
        logger.info("# TRADING CYCLE COMPLETE")
        logger.info("#"*60)
        
        return {
            "analysis": analysis,
            "decisions": {
                "claude": [d.to_dict() for d in decisions["claude"]],
                "gemini": [d.to_dict() for d in decisions["gemini"]]
            }
        }
    
    def show_status(self):
        """Show current account and position status."""
        logger.info("\n" + "="*60)
        logger.info("CURRENT STATUS")
        logger.info("="*60)
        
        # Account info
        account = self.broker.get_account_info()
        logger.info(f"\nAccount: {account.get('account_number', 'N/A')}")
        logger.info(f"Mode: {'PAPER' if account.get('is_paper') else 'LIVE'}")
        logger.info(f"Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        logger.info(f"Cash: ${account.get('cash', 0):,.2f}")
        logger.info(f"Buying Power: ${account.get('buying_power', 0):,.2f}")
        
        # Positions
        positions = self.broker.get_positions()
        logger.info(f"\nOpen Positions: {len(positions)}")
        for pos in positions:
            logger.info(f"  {pos.symbol}: {pos.quantity} shares @ ${pos.entry_price:.2f} "
                       f"(P&L: ${pos.unrealized_pl:.2f} / {pos.unrealized_pl_pct:.2f}%)")
        
        # Market status
        status = self.market_data.get_market_status()
        logger.info(f"\nMarket: {'OPEN' if status.get('is_open') else 'CLOSED'}")
    
    def show_performance_comparison(self):
        """Show performance comparison between bots."""
        logger.info("\n" + "="*60)
        logger.info("BOT PERFORMANCE COMPARISON")
        logger.info("="*60)
        
        comparison = self.performance_tracker.compare_bots()
        
        claude = comparison["claude"]
        gemini = comparison["gemini"]
        
        logger.info(f"\n{'Metric':<25} {'Claude':<15} {'Gemini':<15} {'Winner':<10}")
        logger.info("-"*65)
        
        metrics = [
            ("Total Trades", "total_trades"),
            ("Win Rate (%)", "win_rate"),
            ("Total P&L ($)", "total_profit_loss"),
            ("Avg P&L ($)", "average_profit_loss"),
            ("Avg Confidence", "average_confidence")
        ]
        
        for label, key in metrics:
            c_val = claude.get(key, 0)
            g_val = gemini.get(key, 0)
            
            if key in ["win_rate", "total_profit_loss", "average_profit_loss"]:
                winner = "Claude" if c_val > g_val else "Gemini" if g_val > c_val else "Tie"
            else:
                winner = "-"
            
            logger.info(f"{label:<25} {c_val:<15.2f} {g_val:<15.2f} {winner:<10}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Trading AI System - Dual AI Trading Bot Comparison"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run analysis without executing trades"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current account status and positions"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show bot performance comparison"
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Show configuration status"
    )
    
    args = parser.parse_args()
    
    try:
        orchestrator = TradingOrchestrator()
        
        if args.config:
            settings.print_status()
        elif args.status:
            orchestrator.show_status()
        elif args.compare:
            orchestrator.show_performance_comparison()
        else:
            execute = not args.analyze_only
            orchestrator.run_trading_cycle(execute_trades=execute)
            
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
