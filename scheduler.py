#!/usr/bin/env python3
"""
Trading AI Scheduler - Runs the trading system at appropriate times.

This script handles:
1. Pre-market analysis (8:00 AM ET)
2. Market open trading (9:30 AM ET)
3. Midday check (12:00 PM ET)
4. End of day review (4:00 PM ET)

Usage:
    python scheduler.py          # Run scheduler
    python scheduler.py --once   # Run once immediately
"""

import sys
import time
import argparse
from datetime import datetime, timedelta
import pytz

sys.path.insert(0, '/workspace')

from loguru import logger
from config.settings import settings


def get_eastern_time():
    """Get current time in Eastern timezone."""
    eastern = pytz.timezone('America/New_York')
    return datetime.now(eastern)


def is_trading_day(dt: datetime) -> bool:
    """Check if the given date is a trading day (weekday)."""
    return dt.weekday() < 5  # Monday = 0, Friday = 4


def time_until(target_hour: int, target_minute: int = 0) -> timedelta:
    """Calculate time until a specific time today."""
    now = get_eastern_time()
    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    
    if target <= now:
        # Target time has passed today, schedule for tomorrow
        target += timedelta(days=1)
    
    return target - now


def run_premarket_analysis():
    """Run pre-market analysis."""
    logger.info("Running pre-market analysis...")
    
    from main import TradingOrchestrator
    orchestrator = TradingOrchestrator()
    analysis = orchestrator.run_premarket_analysis()
    
    return analysis


def run_market_open():
    """Run trading decisions at market open."""
    logger.info("Running market open trading cycle...")
    
    from main import TradingOrchestrator
    orchestrator = TradingOrchestrator()
    orchestrator.run_trading_cycle(execute_trades=True)


def run_midday_check():
    """Check positions and performance midday."""
    logger.info("Running midday position check...")
    
    from main import TradingOrchestrator
    orchestrator = TradingOrchestrator()
    orchestrator.show_status()
    orchestrator.show_performance_comparison()


def run_end_of_day():
    """End of day review and summary."""
    logger.info("Running end of day review...")
    
    from main import TradingOrchestrator
    orchestrator = TradingOrchestrator()
    
    orchestrator.show_status()
    orchestrator.show_performance_comparison()
    
    # Export daily report
    from src.utils.database import PerformanceTracker
    tracker = PerformanceTracker()
    tracker.export_to_json(f"reports/daily_report_{get_eastern_time().strftime('%Y%m%d')}.json")


def run_scheduler():
    """
    Main scheduler loop that runs trading activities at appropriate times.
    """
    logger.info("Starting Trading AI Scheduler...")
    logger.info(f"Timezone: America/New_York")
    logger.info(f"Watch Symbols: {settings.trading.watch_symbols}")
    
    # Schedule times (Eastern Time)
    schedule_times = {
        "premarket": (8, 0),    # 8:00 AM - Pre-market analysis
        "market_open": (9, 35), # 9:35 AM - Trading (5 min after open)
        "midday": (12, 0),      # 12:00 PM - Midday check
        "market_close": (16, 5) # 4:05 PM - End of day review
    }
    
    while True:
        now = get_eastern_time()
        
        if not is_trading_day(now):
            logger.info(f"Weekend - sleeping until Monday...")
            # Sleep until next Monday
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            time.sleep(days_until_monday * 24 * 60 * 60)
            continue
        
        current_hour = now.hour
        current_minute = now.minute
        
        # Check which task to run
        task_to_run = None
        
        for task, (hour, minute) in schedule_times.items():
            if current_hour == hour and current_minute == minute:
                task_to_run = task
                break
        
        if task_to_run:
            logger.info(f"Running scheduled task: {task_to_run}")
            
            try:
                if task_to_run == "premarket":
                    run_premarket_analysis()
                elif task_to_run == "market_open":
                    run_market_open()
                elif task_to_run == "midday":
                    run_midday_check()
                elif task_to_run == "market_close":
                    run_end_of_day()
            except Exception as e:
                logger.error(f"Error running {task_to_run}: {e}")
        
        # Sleep for 1 minute before checking again
        time.sleep(60)


def run_once():
    """Run a single trading cycle immediately."""
    logger.info("Running single trading cycle...")
    
    from main import TradingOrchestrator
    orchestrator = TradingOrchestrator()
    
    # Check if market is open
    from src.data.market_data import MarketDataFetcher
    market = MarketDataFetcher()
    status = market.get_market_status()
    
    if status.get("is_open"):
        logger.info("Market is OPEN - running full cycle")
        orchestrator.run_trading_cycle(execute_trades=True)
    else:
        logger.info("Market is CLOSED - running analysis only")
        orchestrator.run_trading_cycle(execute_trades=False)


def main():
    parser = argparse.ArgumentParser(description="Trading AI Scheduler")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once immediately instead of scheduling"
    )
    
    args = parser.parse_args()
    
    # Create reports directory
    import os
    os.makedirs("reports", exist_ok=True)
    
    try:
        if args.once:
            run_once()
        else:
            run_scheduler()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
        raise


if __name__ == "__main__":
    main()
