"""
Performance Tracking Database - Tracks and compares bot performance.
"""
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from loguru import logger


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    id: int
    bot_name: str
    symbol: str
    action: str
    quantity: int
    entry_price: float
    exit_price: Optional[float]
    profit_loss: Optional[float]
    profit_loss_pct: Optional[float]
    entry_time: datetime
    exit_time: Optional[datetime]
    reasoning: str
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "bot_name": self.bot_name,
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "profit_loss": self.profit_loss,
            "profit_loss_pct": self.profit_loss_pct,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "reasoning": self.reasoning,
            "confidence": self.confidence
        }


class PerformanceTracker:
    """
    Tracks and analyzes trading performance for both bots.
    Uses SQLite for persistent storage.
    """
    
    def __init__(self, db_path: str = "trading_performance.db"):
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                profit_loss REAL,
                profit_loss_pct REAL,
                entry_time TIMESTAMP NOT NULL,
                exit_time TIMESTAMP,
                reasoning TEXT,
                confidence REAL,
                status TEXT DEFAULT 'open'
            )
        """)
        
        # Daily performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                bot_name TEXT NOT NULL,
                starting_value REAL,
                ending_value REAL,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_profit_loss REAL,
                max_drawdown REAL,
                UNIQUE(date, bot_name)
            )
        """)
        
        # Decisions log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                bot_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL,
                reasoning TEXT,
                market_data TEXT,
                technical_analysis TEXT,
                sentiment_analysis TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def record_trade_entry(
        self,
        bot_name: str,
        symbol: str,
        action: str,
        quantity: int,
        entry_price: float,
        reasoning: str = "",
        confidence: float = 0
    ) -> int:
        """Record a new trade entry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades (bot_name, symbol, action, quantity, entry_price, 
                              entry_time, reasoning, confidence, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
        """, (bot_name, symbol, action, quantity, entry_price,
              datetime.now().isoformat(), reasoning, confidence))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Recorded trade entry: {bot_name} {action} {quantity} {symbol} @ ${entry_price}")
        return trade_id
    
    def record_trade_exit(
        self,
        trade_id: int,
        exit_price: float
    ):
        """Record a trade exit and calculate P&L."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get original trade
        cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()
        
        if not row:
            logger.warning(f"Trade {trade_id} not found")
            conn.close()
            return
        
        entry_price = row[5]
        quantity = row[4]
        action = row[3]
        
        # Calculate P&L
        if action == 'BUY':
            profit_loss = (exit_price - entry_price) * quantity
        else:
            profit_loss = (entry_price - exit_price) * quantity
        
        profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100
        
        # Update trade
        cursor.execute("""
            UPDATE trades 
            SET exit_price = ?, exit_time = ?, profit_loss = ?, 
                profit_loss_pct = ?, status = 'closed'
            WHERE id = ?
        """, (exit_price, datetime.now().isoformat(), profit_loss, 
              profit_loss_pct, trade_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Trade {trade_id} closed: P&L ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
    
    def record_decision(
        self,
        bot_name: str,
        symbol: str,
        action: str,
        confidence: float,
        reasoning: str,
        market_data: Dict = None,
        technical_analysis: Dict = None,
        sentiment_analysis: Dict = None
    ):
        """Record a trading decision for analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO decisions (timestamp, bot_name, symbol, action, confidence,
                                 reasoning, market_data, technical_analysis, sentiment_analysis)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            bot_name, symbol, action, confidence, reasoning,
            json.dumps(market_data) if market_data else None,
            json.dumps(technical_analysis) if technical_analysis else None,
            json.dumps(sentiment_analysis) if sentiment_analysis else None
        ))
        
        conn.commit()
        conn.close()
    
    def update_daily_performance(
        self,
        bot_name: str,
        portfolio_value: float
    ):
        """Update daily performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date().isoformat()
        
        # Get today's trades for this bot
        cursor.execute("""
            SELECT COUNT(*), 
                   SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END),
                   SUM(profit_loss)
            FROM trades 
            WHERE bot_name = ? AND DATE(entry_time) = ? AND status = 'closed'
        """, (bot_name, today))
        
        row = cursor.fetchone()
        total_trades = row[0] or 0
        winning = row[1] or 0
        losing = row[2] or 0
        total_pl = row[3] or 0
        
        # Check if entry exists
        cursor.execute("""
            SELECT starting_value FROM daily_performance 
            WHERE date = ? AND bot_name = ?
        """, (today, bot_name))
        
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute("""
                UPDATE daily_performance 
                SET ending_value = ?, total_trades = ?, winning_trades = ?,
                    losing_trades = ?, total_profit_loss = ?
                WHERE date = ? AND bot_name = ?
            """, (portfolio_value, total_trades, winning, losing, total_pl, today, bot_name))
        else:
            cursor.execute("""
                INSERT INTO daily_performance 
                (date, bot_name, starting_value, ending_value, total_trades,
                 winning_trades, losing_trades, total_profit_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (today, bot_name, portfolio_value, portfolio_value, 
                  total_trades, winning, losing, total_pl))
        
        conn.commit()
        conn.close()
    
    def get_bot_statistics(self, bot_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a bot."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Overall stats
        cursor.execute("""
            SELECT COUNT(*), 
                   SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END),
                   SUM(profit_loss),
                   AVG(profit_loss),
                   AVG(confidence)
            FROM trades 
            WHERE bot_name = ? AND status = 'closed'
        """, (bot_name,))
        
        row = cursor.fetchone()
        
        total_trades = row[0] or 0
        winning_trades = row[1] or 0
        losing_trades = row[2] or 0
        total_pl = row[3] or 0
        avg_pl = row[4] or 0
        avg_confidence = row[5] or 0
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Best and worst trades
        cursor.execute("""
            SELECT symbol, profit_loss, profit_loss_pct 
            FROM trades WHERE bot_name = ? AND status = 'closed'
            ORDER BY profit_loss DESC LIMIT 1
        """, (bot_name,))
        best_trade = cursor.fetchone()
        
        cursor.execute("""
            SELECT symbol, profit_loss, profit_loss_pct 
            FROM trades WHERE bot_name = ? AND status = 'closed'
            ORDER BY profit_loss ASC LIMIT 1
        """, (bot_name,))
        worst_trade = cursor.fetchone()
        
        # Recent trades
        cursor.execute("""
            SELECT * FROM trades WHERE bot_name = ? 
            ORDER BY entry_time DESC LIMIT 10
        """, (bot_name,))
        recent = cursor.fetchall()
        
        conn.close()
        
        return {
            "bot_name": bot_name,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "total_profit_loss": round(total_pl, 2),
            "average_profit_loss": round(avg_pl, 2),
            "average_confidence": round(avg_confidence, 2),
            "best_trade": {
                "symbol": best_trade[0],
                "profit_loss": best_trade[1],
                "profit_loss_pct": best_trade[2]
            } if best_trade else None,
            "worst_trade": {
                "symbol": worst_trade[0],
                "profit_loss": worst_trade[1],
                "profit_loss_pct": worst_trade[2]
            } if worst_trade else None,
            "recent_trades_count": len(recent)
        }
    
    def compare_bots(self) -> Dict[str, Any]:
        """Compare performance between Claude and Gemini bots."""
        claude_stats = self.get_bot_statistics("Claude Trader")
        gemini_stats = self.get_bot_statistics("Gemini Trader")
        
        comparison = {
            "claude": claude_stats,
            "gemini": gemini_stats,
            "comparison": {}
        }
        
        # Determine winners in each category
        categories = [
            ("total_profit_loss", "higher"),
            ("win_rate", "higher"),
            ("average_profit_loss", "higher"),
            ("total_trades", "info")
        ]
        
        for category, direction in categories:
            claude_val = claude_stats.get(category, 0)
            gemini_val = gemini_stats.get(category, 0)
            
            if direction == "higher":
                winner = "Claude" if claude_val > gemini_val else "Gemini" if gemini_val > claude_val else "Tie"
            else:
                winner = "N/A"
            
            comparison["comparison"][category] = {
                "claude": claude_val,
                "gemini": gemini_val,
                "winner": winner
            }
        
        return comparison
    
    def get_daily_summary(self, date: str = None) -> Dict[str, Any]:
        """Get daily trading summary for both bots."""
        if date is None:
            date = datetime.now().date().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT bot_name, starting_value, ending_value, total_trades,
                   winning_trades, losing_trades, total_profit_loss
            FROM daily_performance WHERE date = ?
        """, (date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        summary = {"date": date, "bots": {}}
        
        for row in rows:
            summary["bots"][row[0]] = {
                "starting_value": row[1],
                "ending_value": row[2],
                "total_trades": row[3],
                "winning_trades": row[4],
                "losing_trades": row[5],
                "total_profit_loss": row[6],
                "return_pct": ((row[2] - row[1]) / row[1] * 100) if row[1] else 0
            }
        
        return summary
    
    def export_to_json(self, filepath: str = "performance_report.json"):
        """Export all performance data to JSON."""
        data = {
            "exported_at": datetime.now().isoformat(),
            "claude_stats": self.get_bot_statistics("Claude Trader"),
            "gemini_stats": self.get_bot_statistics("Gemini Trader"),
            "comparison": self.compare_bots()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Performance report exported to {filepath}")
        return data
