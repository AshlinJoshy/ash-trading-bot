#!/usr/bin/env python3
"""
Test script to verify the Trading AI System setup.
Run this after configuring your .env file to check everything works.
"""

import sys
sys.path.insert(0, '/workspace')

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def test_imports():
    """Test that all required packages are installed."""
    console.print("\n[bold]Testing Package Imports...[/bold]")
    
    packages = {
        "anthropic": "Claude API",
        "google.generativeai": "Gemini API",
        "alpaca_trade_api": "Alpaca Broker",
        "yfinance": "Market Data",
        "pandas": "Data Analysis",
        "ta": "Technical Analysis",
        "feedparser": "RSS News",
        "textblob": "Sentiment Analysis",
        "loguru": "Logging",
        "rich": "UI",
    }
    
    results = {}
    for package, description in packages.items():
        try:
            __import__(package)
            results[description] = ("✅", "Installed")
        except ImportError:
            results[description] = ("❌", "Missing")
    
    table = Table(title="Package Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("", style="dim")
    
    for desc, (icon, status) in results.items():
        table.add_row(desc, icon, status)
    
    console.print(table)
    return all(icon == "✅" for icon, _ in results.values())


def test_configuration():
    """Test configuration loading."""
    console.print("\n[bold]Testing Configuration...[/bold]")
    
    try:
        from config.settings import settings
        
        status = settings.validate()
        
        table = Table(title="Configuration Status")
        table.add_column("Setting", style="cyan")
        table.add_column("Status", style="green")
        
        for key, configured in status.items():
            icon = "✅ Configured" if configured else "❌ Missing"
            table.add_row(key.replace("_", " ").title(), icon)
        
        console.print(table)
        
        console.print(f"\nWatch Symbols: {settings.trading.watch_symbols}")
        console.print(f"Max Position Size: {settings.trading.max_position_size_pct}%")
        console.print(f"Stop Loss: {settings.trading.stop_loss_pct}%")
        
        return True
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        return False


def test_market_data():
    """Test market data fetching."""
    console.print("\n[bold]Testing Market Data...[/bold]")
    
    try:
        from src.data.market_data import MarketDataFetcher
        
        fetcher = MarketDataFetcher()
        
        # Test getting SPY data
        console.print("Fetching SPY data...")
        df = fetcher.get_historical_data("SPY", period="5d", interval="1d")
        
        if not df.empty:
            console.print(f"[green]✅ Got {len(df)} days of SPY data[/green]")
            console.print(f"   Latest close: ${df['close'].iloc[-1]:.2f}")
            return True
        else:
            console.print("[yellow]⚠️ No data returned[/yellow]")
            return False
            
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return False


def test_technical_analysis():
    """Test technical analysis."""
    console.print("\n[bold]Testing Technical Analysis...[/bold]")
    
    try:
        from src.data.market_data import MarketDataFetcher
        from src.analysis.technical import TechnicalAnalyzer
        
        fetcher = MarketDataFetcher()
        analyzer = TechnicalAnalyzer()
        
        df = fetcher.get_historical_data("SPY", period="3mo", interval="1d")
        
        if df.empty:
            console.print("[yellow]⚠️ No data for analysis[/yellow]")
            return False
        
        signal = analyzer.analyze(df)
        
        console.print(f"[green]✅ Technical analysis complete[/green]")
        console.print(f"   Trend: {signal.trend}")
        console.print(f"   Strength: {signal.strength:.2f}")
        console.print(f"   Recommendation: {signal.recommendation}")
        console.print(f"   Support levels: {len(signal.support_levels)}")
        console.print(f"   Resistance levels: {len(signal.resistance_levels)}")
        console.print(f"   Patterns detected: {len(signal.patterns)}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return False


def test_news_fetching():
    """Test news fetching."""
    console.print("\n[bold]Testing News Fetching...[/bold]")
    
    try:
        from src.data.news_fetcher import NewsFetcher
        
        fetcher = NewsFetcher()
        
        # Test RSS feeds
        console.print("Fetching RSS news...")
        articles = fetcher.fetch_rss_news(max_articles=5)
        
        console.print(f"[green]✅ Got {len(articles)} news articles[/green]")
        
        if articles:
            console.print(f"   Sample: {articles[0].title[:60]}...")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return False


def test_claude_connection():
    """Test Claude API connection."""
    console.print("\n[bold]Testing Claude API...[/bold]")
    
    try:
        from config.settings import settings
        
        if not settings.ai.anthropic_api_key:
            console.print("[yellow]⚠️ Claude API key not configured[/yellow]")
            return False
        
        from anthropic import Anthropic
        
        client = Anthropic(api_key=settings.ai.anthropic_api_key)
        
        # Quick test message
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'Trading AI ready!' in 3 words or less"}]
        )
        
        console.print(f"[green]✅ Claude connected[/green]")
        console.print(f"   Response: {response.content[0].text}")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return False


def test_gemini_connection():
    """Test Gemini API connection."""
    console.print("\n[bold]Testing Gemini API...[/bold]")
    
    try:
        from config.settings import settings
        
        if not settings.ai.google_api_key:
            console.print("[yellow]⚠️ Gemini API key not configured[/yellow]")
            return False
        
        import google.generativeai as genai
        
        genai.configure(api_key=settings.ai.google_api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        response = model.generate_content("Say 'Trading AI ready!' in 3 words or less")
        
        console.print(f"[green]✅ Gemini connected[/green]")
        console.print(f"   Response: {response.text}")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return False


def test_broker_connection():
    """Test Alpaca broker connection."""
    console.print("\n[bold]Testing Alpaca Broker...[/bold]")
    
    try:
        from config.settings import settings
        
        if not settings.broker.alpaca_api_key:
            console.print("[yellow]⚠️ Alpaca credentials not configured[/yellow]")
            return False
        
        from src.broker.alpaca_broker import AlpacaBroker
        
        broker = AlpacaBroker()
        account = broker.get_account_info()
        
        if "error" in account:
            console.print(f"[red]❌ Error: {account['error']}[/red]")
            return False
        
        mode = "PAPER" if account.get("is_paper") else "LIVE"
        console.print(f"[green]✅ Alpaca connected ({mode} mode)[/green]")
        console.print(f"   Account: {account.get('account_number')}")
        console.print(f"   Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        console.print(f"   Buying Power: ${account.get('buying_power', 0):,.2f}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        return False


def main():
    """Run all tests."""
    console.print(Panel.fit(
        "[bold blue]Trading AI System - Setup Test[/bold blue]\n"
        "Testing all components...",
        border_style="blue"
    ))
    
    results = {
        "Packages": test_imports(),
        "Configuration": test_configuration(),
        "Market Data": test_market_data(),
        "Technical Analysis": test_technical_analysis(),
        "News Fetching": test_news_fetching(),
        "Claude API": test_claude_connection(),
        "Gemini API": test_gemini_connection(),
        "Alpaca Broker": test_broker_connection(),
    }
    
    # Summary
    console.print("\n" + "="*50)
    console.print("[bold]SUMMARY[/bold]")
    console.print("="*50)
    
    all_passed = True
    for test, passed in results.items():
        icon = "✅" if passed else "❌"
        console.print(f"  {icon} {test}")
        if not passed:
            all_passed = False
    
    console.print("="*50)
    
    if all_passed:
        console.print("\n[bold green]🎉 All tests passed! Ready to trade.[/bold green]")
        console.print("\nRun: [cyan]python main.py --analyze-only[/cyan] to start")
    else:
        console.print("\n[bold yellow]⚠️ Some tests failed. Check configuration.[/bold yellow]")
        console.print("\n1. Copy .env.example to .env")
        console.print("2. Add your API keys")
        console.print("3. Run this test again")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
