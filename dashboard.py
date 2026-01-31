#!/usr/bin/env python3
"""
Trading AI Dashboard - Streamlit Web Interface

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import sys

sys.path.insert(0, '/workspace')

# Page config
st.set_page_config(
    page_title="Trading AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 2rem;
        font-weight: bold;
    }
    .green { color: #00c853; }
    .red { color: #ff1744; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_components():
    """Initialize trading components."""
    from config.settings import settings
    from src.data.market_data import MarketDataFetcher
    from src.analysis.technical import TechnicalAnalyzer
    from src.broker.alpaca_broker import AlpacaBroker
    from src.utils.database import PerformanceTracker
    
    return {
        "settings": settings,
        "market_data": MarketDataFetcher(),
        "technical": TechnicalAnalyzer(),
        "broker": AlpacaBroker(),
        "tracker": PerformanceTracker()
    }


def render_header():
    """Render dashboard header."""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title("🤖 Trading AI Dashboard")
        st.caption("Claude vs Gemini - Demo Trading Comparison")
    
    with col2:
        # Market status
        components = init_components()
        status = components["market_data"].get_market_status()
        if status.get("is_open"):
            st.success("🟢 Market OPEN")
        else:
            st.error("🔴 Market CLOSED")
    
    with col3:
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()


def render_portfolio_overview():
    """Render portfolio overview section."""
    st.header("💰 Portfolio Overview")
    
    components = init_components()
    account = components["broker"].get_account_info()
    
    if "error" in account:
        st.warning("⚠️ Broker not connected. Add Alpaca API keys to .env")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Portfolio Value",
            f"${account.get('portfolio_value', 0):,.2f}",
            f"${account.get('portfolio_value', 0) - account.get('last_equity', 0):,.2f}"
        )
    
    with col2:
        st.metric(
            "Cash",
            f"${account.get('cash', 0):,.2f}"
        )
    
    with col3:
        st.metric(
            "Buying Power",
            f"${account.get('buying_power', 0):,.2f}"
        )
    
    with col4:
        mode = "PAPER" if account.get("is_paper") else "LIVE"
        st.metric("Mode", mode)


def render_positions():
    """Render current positions."""
    st.header("📊 Current Positions")
    
    components = init_components()
    positions = components["broker"].get_positions()
    
    if not positions:
        st.info("No open positions")
        return
    
    # Create positions dataframe
    data = []
    for pos in positions:
        data.append({
            "Symbol": pos.symbol,
            "Qty": pos.quantity,
            "Entry": f"${pos.entry_price:.2f}",
            "Current": f"${pos.current_price:.2f}",
            "P&L": f"${pos.unrealized_pl:.2f}",
            "P&L %": f"{pos.unrealized_pl_pct:.2f}%",
            "Value": f"${pos.market_value:.2f}"
        })
    
    df = pd.DataFrame(data)
    
    # Color P&L column
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


def render_bot_comparison():
    """Render bot performance comparison."""
    st.header("🏆 Bot Performance Comparison")
    
    components = init_components()
    comparison = components["tracker"].compare_bots()
    
    claude = comparison["claude"]
    gemini = comparison["gemini"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔵 Claude Trader")
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Trades", claude.get("total_trades", 0))
            st.metric("Win Rate", f"{claude.get('win_rate', 0):.1f}%")
        with c2:
            pl = claude.get("total_profit_loss", 0)
            st.metric(
                "Total P&L",
                f"${pl:,.2f}",
                delta=f"${pl:,.2f}" if pl != 0 else None,
                delta_color="normal" if pl >= 0 else "inverse"
            )
            st.metric("Avg Confidence", f"{claude.get('average_confidence', 0):.0%}")
    
    with col2:
        st.subheader("🟡 Gemini Trader")
        
        g1, g2 = st.columns(2)
        with g1:
            st.metric("Total Trades", gemini.get("total_trades", 0))
            st.metric("Win Rate", f"{gemini.get('win_rate', 0):.1f}%")
        with g2:
            pl = gemini.get("total_profit_loss", 0)
            st.metric(
                "Total P&L",
                f"${pl:,.2f}",
                delta=f"${pl:,.2f}" if pl != 0 else None,
                delta_color="normal" if pl >= 0 else "inverse"
            )
            st.metric("Avg Confidence", f"{gemini.get('average_confidence', 0):.0%}")
    
    # Winner banner
    if claude.get("total_profit_loss", 0) > gemini.get("total_profit_loss", 0):
        st.success("🏆 Claude is winning!")
    elif gemini.get("total_profit_loss", 0) > claude.get("total_profit_loss", 0):
        st.success("🏆 Gemini is winning!")
    else:
        st.info("🤝 It's a tie!")


@st.cache_data(ttl=60)
def get_stock_data(symbol: str, period: str = "1mo"):
    """Get stock data with caching."""
    components = init_components()
    return components["market_data"].get_historical_data(symbol, period=period)


def render_technical_analysis():
    """Render technical analysis section."""
    st.header("📈 Technical Analysis")
    
    components = init_components()
    symbols = components["settings"].trading.watch_symbols
    
    # Symbol selector
    selected = st.selectbox("Select Symbol", symbols)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get data
        df = get_stock_data(selected, "3mo")
        
        if df.empty:
            st.warning("No data available")
            return
        
        # Create candlestick chart
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))
        
        # Add moving averages
        if len(df) >= 20:
            df['SMA20'] = df['close'].rolling(20).mean()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['SMA20'],
                name='SMA 20',
                line=dict(color='orange', width=1)
            ))
        
        if len(df) >= 50:
            df['SMA50'] = df['close'].rolling(50).mean()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['SMA50'],
                name='SMA 50',
                line=dict(color='blue', width=1)
            ))
        
        fig.update_layout(
            title=f"{selected} Price Chart",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Technical signals
        signal = components["technical"].analyze(df)
        
        st.subheader("Analysis")
        
        # Trend indicator
        trend_color = "green" if signal.trend == "bullish" else "red" if signal.trend == "bearish" else "gray"
        st.markdown(f"**Trend:** :{trend_color}[{signal.trend.upper()}]")
        st.progress(signal.strength, text=f"Strength: {signal.strength:.0%}")
        
        # Recommendation
        st.info(f"📋 {signal.recommendation}")
        
        # Key indicators
        st.subheader("Indicators")
        indicators = signal.indicators
        
        if indicators.get('rsi'):
            rsi = indicators['rsi']
            rsi_color = "red" if rsi > 70 else "green" if rsi < 30 else "gray"
            st.markdown(f"**RSI:** :{rsi_color}[{rsi:.1f}]")
        
        if indicators.get('macd_histogram'):
            macd = indicators['macd_histogram']
            macd_color = "green" if macd > 0 else "red"
            st.markdown(f"**MACD:** :{macd_color}[{macd:.4f}]")
        
        # Support/Resistance
        st.subheader("Support Levels")
        for level in signal.support_levels[:3]:
            st.markdown(f"• ${level.price:.2f} (strength: {level.strength})")
        
        st.subheader("Resistance Levels")
        for level in signal.resistance_levels[:3]:
            st.markdown(f"• ${level.price:.2f} (strength: {level.strength})")
        
        # Patterns
        if signal.patterns:
            st.subheader("Patterns Detected")
            for pattern in signal.patterns:
                icon = "🟢" if pattern.signal == "bullish" else "🔴" if pattern.signal == "bearish" else "⚪"
                st.markdown(f"{icon} **{pattern.name}**")


def render_news_sentiment():
    """Render news and sentiment section."""
    st.header("📰 News & Sentiment")
    
    from src.data.news_fetcher import NewsFetcher
    from src.analysis.sentiment import SentimentAnalyzer, analyze_market_sentiment
    
    with st.spinner("Fetching latest news..."):
        fetcher = NewsFetcher()
        news_data = fetcher.fetch_all_news()
        sentiment = analyze_market_sentiment(news_data)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Overall sentiment
        overall = sentiment.get("overall", {})
        signal = overall.get("signal", {})
        
        st.subheader("Market Sentiment")
        
        sentiment_score = signal.get("sentiment_score", 0)
        if sentiment_score > 0.1:
            st.success(f"🟢 BULLISH ({sentiment_score:.2f})")
        elif sentiment_score < -0.1:
            st.error(f"🔴 BEARISH ({sentiment_score:.2f})")
        else:
            st.info(f"⚪ NEUTRAL ({sentiment_score:.2f})")
        
        st.metric("Signal", signal.get("signal", "N/A"))
        st.metric("Confidence", f"{signal.get('confidence', 0):.0%}")
        st.metric("Articles Analyzed", signal.get("articles_analyzed", 0))
        
        # Key themes
        themes = signal.get("key_themes", [])
        if themes:
            st.subheader("Key Themes")
            for theme in themes:
                st.markdown(f"• {theme.title()}")
    
    with col2:
        # News articles
        st.subheader("Latest Headlines")
        
        articles = news_data.get("general_news", [])[:10]
        
        for article in articles:
            with st.expander(f"📄 {article.get('title', 'No title')[:80]}..."):
                st.markdown(f"**Source:** {article.get('source', 'Unknown')}")
                st.markdown(f"**Summary:** {article.get('summary', 'No summary')[:300]}...")
                if article.get('url'):
                    st.markdown(f"[Read more]({article.get('url')})")


def render_run_analysis():
    """Render the run analysis/trading section."""
    st.header("🚀 Run Trading Analysis")
    
    components = init_components()
    settings = components["settings"]
    
    st.markdown("Run the AI trading bots to get their recommendations.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Settings")
        
        # Show current settings
        st.markdown(f"**Watch Symbols:** {', '.join(settings.trading.watch_symbols)}")
        st.markdown(f"**Max Position Size:** {settings.trading.max_position_size_pct}%")
        st.markdown(f"**Stop Loss:** {settings.trading.stop_loss_pct}%")
        st.markdown(f"**Take Profit:** {settings.trading.take_profit_pct}%")
    
    with col2:
        st.subheader("Actions")
        
        # Validate API keys
        status = settings.validate()
        
        if not status["ai_claude"] and not status["ai_gemini"]:
            st.error("❌ No AI APIs configured. Add API keys to .env")
            return
        
        if not status["ai_claude"]:
            st.warning("⚠️ Claude API not configured")
        if not status["ai_gemini"]:
            st.warning("⚠️ Gemini API not configured")
        
        analyze_only = st.checkbox("Analysis only (no trades)", value=True)
        
        if st.button("🤖 Run AI Analysis", type="primary"):
            with st.spinner("Running analysis... This may take a minute."):
                try:
                    from main import TradingOrchestrator
                    orchestrator = TradingOrchestrator()
                    result = orchestrator.run_trading_cycle(execute_trades=not analyze_only)
                    
                    st.success("✅ Analysis complete!")
                    
                    # Show decisions
                    st.subheader("AI Decisions")
                    
                    decisions = result.get("decisions", {})
                    
                    if decisions.get("claude"):
                        st.markdown("**Claude's Recommendations:**")
                        for d in decisions["claude"]:
                            action = d.get("action", "HOLD")
                            symbol = d.get("symbol", "")
                            conf = d.get("confidence", 0)
                            reason = d.get("reasoning", "")[:100]
                            
                            icon = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
                            st.markdown(f"{icon} **{action} {symbol}** (Confidence: {conf:.0%})")
                            st.caption(reason)
                    
                    if decisions.get("gemini"):
                        st.markdown("**Gemini's Recommendations:**")
                        for d in decisions["gemini"]:
                            action = d.get("action", "HOLD")
                            symbol = d.get("symbol", "")
                            conf = d.get("confidence", 0)
                            reason = d.get("reasoning", "")[:100]
                            
                            icon = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
                            st.markdown(f"{icon} **{action} {symbol}** (Confidence: {conf:.0%})")
                            st.caption(reason)
                    
                except Exception as e:
                    st.error(f"Error: {e}")


def render_settings():
    """Render settings page."""
    st.header("⚙️ Configuration")
    
    components = init_components()
    settings = components["settings"]
    status = settings.validate()
    
    st.subheader("API Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if status["ai_claude"]:
            st.success("✅ Claude API")
        else:
            st.error("❌ Claude API")
    
    with col2:
        if status["ai_gemini"]:
            st.success("✅ Gemini API")
        else:
            st.error("❌ Gemini API")
    
    with col3:
        if status["broker"]:
            st.success("✅ Alpaca Broker")
        else:
            st.error("❌ Alpaca Broker")
    
    with col4:
        if status["news_api"]:
            st.success("✅ News API")
        else:
            st.warning("⚠️ News API (optional)")
    
    st.subheader("Trading Parameters")
    
    st.json({
        "watch_symbols": settings.trading.watch_symbols,
        "max_position_size_pct": settings.trading.max_position_size_pct,
        "max_daily_loss_pct": settings.trading.max_daily_loss_pct,
        "stop_loss_pct": settings.trading.stop_loss_pct,
        "take_profit_pct": settings.trading.take_profit_pct,
        "max_positions": settings.trading.max_positions
    })
    
    st.info("💡 Edit the `.env` file to change these settings, then restart the dashboard.")


def main():
    """Main dashboard."""
    render_header()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["📊 Overview", "📈 Technical Analysis", "📰 News & Sentiment", "🚀 Run Analysis", "⚙️ Settings"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")
    
    if st.sidebar.button("📊 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Render selected page
    if page == "📊 Overview":
        render_portfolio_overview()
        st.markdown("---")
        render_positions()
        st.markdown("---")
        render_bot_comparison()
    
    elif page == "📈 Technical Analysis":
        render_technical_analysis()
    
    elif page == "📰 News & Sentiment":
        render_news_sentiment()
    
    elif page == "🚀 Run Analysis":
        render_run_analysis()
    
    elif page == "⚙️ Settings":
        render_settings()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Trading AI System v1.0")
    st.sidebar.caption("⚠️ Paper trading only")


if __name__ == "__main__":
    main()
