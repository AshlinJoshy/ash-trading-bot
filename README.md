# Trading AI System 🤖📈

A dual-AI trading system that uses **Claude (Anthropic)** and **Gemini (Google)** to analyze markets, news sentiment, and technical patterns to make day trading decisions. Run both bots on a demo account to compare their performance.

## Features

- **🖥️ Web Dashboard**: Beautiful Streamlit interface to monitor everything
- **🧠 Dual AI Analysis**: Compare trading decisions between Claude and Gemini
- **📰 News Sentiment Analysis**: Pre-market news scanning and sentiment scoring
- **📊 Technical Analysis**: Support/resistance levels, candlestick patterns, indicators (RSI, MACD, Bollinger Bands, etc.)
- **💹 Demo Trading**: Safe paper trading on Alpaca's paper trading platform
- **📈 Performance Tracking**: SQLite database tracking trades, wins, losses, and bot comparison
- **⚡ Risk Management**: Automatic stop-loss, take-profit, and position sizing

## Dashboard Preview

The web dashboard includes:
- 📊 **Portfolio Overview** - Account balance, positions, P&L
- 🏆 **Bot Comparison** - Side-by-side Claude vs Gemini performance
- 📈 **Technical Charts** - Candlestick charts with indicators
- 📰 **News Feed** - Latest market news with sentiment scores
- 🚀 **Run Analysis** - Trigger AI analysis with one click

## Quick Start

### 1. Clone and Install Dependencies

```bash
cd /workspace
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# AI APIs
ANTHROPIC_API_KEY=your_claude_api_key
GOOGLE_API_KEY=your_gemini_api_key

# Broker (Alpaca - free paper trading)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 3. Run the System

**Option A: Web Dashboard (Recommended)**
```bash
streamlit run dashboard.py
```
Then open http://localhost:8501 in your browser.

**Option B: Command Line**
```bash
# Full trading cycle (analysis + trading)
python main.py

# Analysis only (no trades)
python main.py --analyze-only

# Check status
python main.py --status

# Compare bot performance
python main.py --compare
```

## Getting API Keys

### Claude (Anthropic)
1. Go to https://console.anthropic.com/
2. Sign up and get an API key
3. Add to `.env` as `ANTHROPIC_API_KEY`

### Gemini (Google)
1. Go to https://makersuite.google.com/app/apikey
2. Create an API key
3. Add to `.env` as `GOOGLE_API_KEY`

### Alpaca (Free Paper Trading)
1. Go to https://alpaca.markets/
2. Sign up for free
3. Go to Paper Trading API Keys
4. Copy API Key and Secret Key
5. Add to `.env`

**Important**: Use the paper trading URL (`https://paper-api.alpaca.markets`) for testing!

## Architecture

```
trading-ai-system/
├── main.py                 # Main orchestrator
├── config/
│   └── settings.py         # Configuration management
├── src/
│   ├── data/
│   │   ├── market_data.py  # Price data fetching
│   │   └── news_fetcher.py # News aggregation
│   ├── analysis/
│   │   ├── technical.py    # Technical analysis
│   │   └── sentiment.py    # Sentiment analysis
│   ├── ai_bots/
│   │   ├── base_bot.py     # Base trader class
│   │   ├── claude_bot.py   # Claude trading bot
│   │   └── gemini_bot.py   # Gemini trading bot
│   ├── broker/
│   │   └── alpaca_broker.py # Trade execution
│   └── utils/
│       ├── logger.py       # Logging
│       └── database.py     # Performance tracking
├── .env.example            # Environment template
└── requirements.txt        # Dependencies
```

## How It Works

### 1. Pre-Market Analysis (Run before 9:30 AM ET)
- Fetches latest news from multiple sources (RSS, NewsAPI, Yahoo Finance)
- Analyzes sentiment of news articles
- Identifies key themes and market drivers

### 2. Technical Analysis
- Calculates indicators: RSI, MACD, Bollinger Bands, SMA, EMA
- Identifies support and resistance levels
- Detects candlestick patterns (Doji, Hammer, Engulfing, etc.)
- Determines trend direction and strength

### 3. AI Decision Making
Both Claude and Gemini receive:
- Technical analysis data
- Sentiment scores
- Current portfolio state
- Risk parameters

Each AI independently decides:
- BUY, SELL, HOLD, or CLOSE
- Entry price, stop loss, take profit
- Confidence level (0-100%)
- Reasoning for the decision

### 4. Trade Execution
- Orders executed on Alpaca paper trading
- Automatic position sizing based on risk rules
- Stop loss and take profit orders placed
- All trades logged for analysis

### 5. Performance Tracking
- Win rate, total P&L, average P&L
- Best/worst trades
- Daily summaries
- Bot comparison metrics

## Configuration

### Trading Settings (in `.env`)

```env
# Symbols to watch
WATCH_SYMBOLS=SPY,AAPL,MSFT,GOOGL,NVDA,TSLA,AMD,META

# Risk Management
MAX_POSITION_SIZE_PCT=5    # Max 5% of portfolio per trade
MAX_DAILY_LOSS_PCT=2       # Stop trading if down 2% daily
STOP_LOSS_PCT=2            # 2% stop loss
TAKE_PROFIT_PCT=4          # 4% take profit
MAX_POSITIONS=5            # Max 5 concurrent positions
```

## Example Output

```
================================================================================
TRADING AI SYSTEM - STARTING TRADING CYCLE
Time: 2024-01-15 09:15:00
================================================================================

RUNNING PRE-MARKET ANALYSIS
============================================================
Checking market status...
Fetching news and market headlines...
Total news articles fetched: 47
Analyzing market sentiment...
Running technical analysis...

ANALYSIS SUMMARY
============================================================
Market Open: False
Overall Sentiment: LEAN_BULLISH (Confidence: 62%)

Technical Analysis by Symbol:
  SPY: BULLISH - BUY - Bullish trend confirmed by indicators
  AAPL: NEUTRAL - NEUTRAL - No clear direction
  NVDA: BULLISH - STRONG BUY - Bullish trend with confirming patterns

GETTING AI TRADING DECISIONS
============================================================

Asking Claude for trading decisions...
  Claude: BUY NVDA (Confidence: 78%)
    Reason: Strong bullish momentum, positive sentiment from AI news...

Asking Gemini for trading decisions...
  Gemini: BUY NVDA (Confidence: 72%)
    Reason: Technical breakout above resistance, bullish engulfing pattern...
  Gemini: BUY SPY (Confidence: 65%)
    Reason: Market sentiment positive, trend following strategy...

BOT PERFORMANCE COMPARISON
============================================================
Metric                    Claude          Gemini          Winner    
-----------------------------------------------------------------
Total Trades              12              15              -         
Win Rate (%)              58.33           53.33           Claude    
Total P&L ($)             1,245.50        987.20          Claude    
Avg P&L ($)               103.79          65.81           Claude    
```

## Safety Features

1. **Paper Trading Only by Default**: Uses Alpaca's paper trading API
2. **Confidence Threshold**: Only trades with >60% confidence
3. **Position Limits**: Max 5 positions, max 5% per position
4. **Stop Loss**: Automatic 2% stop loss on all trades
5. **Daily Loss Limit**: Stops trading if down 2% for the day
6. **Human Oversight**: Run `--analyze-only` to review before trading

## Extending the System

### Add Custom Technical Indicators
Edit `src/analysis/technical.py` to add more indicators using the `ta` library.

### Add News Sources
Edit `src/data/news_fetcher.py` to add RSS feeds or APIs.

### Custom Risk Rules
Edit `src/ai_bots/base_bot.py` to modify the trading prompt.

## Disclaimer

⚠️ **This is for educational purposes only.**

- This system is designed for paper/demo trading to learn and experiment
- Past performance does not guarantee future results
- Never trade with money you can't afford to lose
- AI can make mistakes - always review decisions manually
- Consult a financial advisor before real trading

## License

MIT License - Use at your own risk.
