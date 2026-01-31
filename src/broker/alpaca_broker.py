"""
Alpaca Broker Integration - Handles order execution on Alpaca paper/live trading.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from loguru import logger

import sys
sys.path.insert(0, '/workspace')
from config.settings import settings
from src.ai_bots.base_bot import TradeDecision, TradeAction


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class Order:
    """Represents a trading order."""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    order_type: str  # 'market', 'limit', 'stop', 'stop_limit'
    status: OrderStatus
    filled_qty: int = 0
    filled_avg_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    bot_name: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "status": self.status.value,
            "filled_qty": self.filled_qty,
            "filled_avg_price": self.filled_avg_price,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "bot_name": self.bot_name
        }


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float
    side: str  # 'long' or 'short'
    bot_name: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": round(self.entry_price, 2),
            "current_price": round(self.current_price, 2),
            "market_value": round(self.market_value, 2),
            "unrealized_pl": round(self.unrealized_pl, 2),
            "unrealized_pl_pct": round(self.unrealized_pl_pct, 2),
            "side": self.side,
            "bot_name": self.bot_name
        }


class AlpacaBroker:
    """
    Broker class for executing trades on Alpaca.
    Supports both paper (demo) and live trading.
    """
    
    def __init__(self):
        self.api = None
        self.is_paper = "paper" in settings.broker.alpaca_base_url.lower()
        self._init_api()
        
        # Track orders by bot
        self.orders_by_bot: Dict[str, List[Order]] = {
            "Claude Trader": [],
            "Gemini Trader": []
        }
    
    def _init_api(self):
        """Initialize Alpaca API connection."""
        if not settings.broker.alpaca_api_key or not settings.broker.alpaca_secret_key:
            logger.warning("Alpaca API credentials not configured")
            return
        
        try:
            from alpaca_trade_api import REST
            
            self.api = REST(
                key_id=settings.broker.alpaca_api_key,
                secret_key=settings.broker.alpaca_secret_key,
                base_url=settings.broker.alpaca_base_url
            )
            
            # Verify connection
            account = self.api.get_account()
            mode = "PAPER" if self.is_paper else "LIVE"
            logger.info(f"Connected to Alpaca ({mode}) - Account: {account.account_number}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            
        except ImportError:
            logger.error("alpaca-trade-api package not installed")
        except Exception as e:
            logger.error(f"Error connecting to Alpaca: {e}")
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        if not self.api:
            return {"error": "Alpaca not connected"}
        
        try:
            account = self.api.get_account()
            return {
                "account_number": account.account_number,
                "status": account.status,
                "currency": account.currency,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "daytrade_count": account.daytrade_count,
                "is_paper": self.is_paper
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {"error": str(e)}
    
    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        if not self.api:
            return []
        
        try:
            positions = self.api.list_positions()
            result = []
            
            for pos in positions:
                position = Position(
                    symbol=pos.symbol,
                    quantity=int(pos.qty),
                    entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    unrealized_pl=float(pos.unrealized_pl),
                    unrealized_pl_pct=float(pos.unrealized_plpc) * 100,
                    side='long' if int(pos.qty) > 0 else 'short'
                )
                result.append(position)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def execute_decision(
        self,
        decision: TradeDecision,
        portfolio_value: float = None
    ) -> Optional[Order]:
        """
        Execute a trading decision.
        
        Args:
            decision: TradeDecision from an AI bot
            portfolio_value: Portfolio value for position sizing
        
        Returns:
            Order object if successful
        """
        if not self.api:
            logger.error("Alpaca not connected")
            return None
        
        if decision.action == TradeAction.HOLD:
            logger.info(f"HOLD decision for {decision.symbol} - no action taken")
            return None
        
        try:
            # Get portfolio value if not provided
            if portfolio_value is None:
                account = self.api.get_account()
                portfolio_value = float(account.portfolio_value)
            
            # Calculate position size
            if decision.quantity:
                quantity = decision.quantity
            elif decision.entry_price and decision.stop_loss:
                # Calculate based on risk management
                quantity = self._calculate_position_size(
                    portfolio_value=portfolio_value,
                    entry_price=decision.entry_price,
                    stop_loss=decision.stop_loss
                )
            else:
                # Default: use max position size
                current_price = self._get_current_price(decision.symbol)
                if not current_price:
                    logger.error(f"Could not get price for {decision.symbol}")
                    return None
                
                max_value = portfolio_value * (settings.trading.max_position_size_pct / 100)
                quantity = int(max_value / current_price)
            
            if quantity <= 0:
                logger.warning(f"Invalid quantity for {decision.symbol}")
                return None
            
            # Determine order side
            if decision.action == TradeAction.BUY:
                side = 'buy'
            elif decision.action in [TradeAction.SELL, TradeAction.CLOSE]:
                side = 'sell'
            else:
                return None
            
            # Submit order
            logger.info(f"Submitting {side.upper()} order: {quantity} shares of {decision.symbol}")
            
            order = self.api.submit_order(
                symbol=decision.symbol,
                qty=quantity,
                side=side,
                type='market',  # Use market orders for simplicity
                time_in_force='day'
            )
            
            # Create Order object
            result = Order(
                id=order.id,
                symbol=order.symbol,
                side=side,
                quantity=int(order.qty),
                order_type=order.type,
                status=OrderStatus.PENDING,
                submitted_at=datetime.now(),
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                bot_name=decision.bot_name
            )
            
            # Track order by bot
            if decision.bot_name in self.orders_by_bot:
                self.orders_by_bot[decision.bot_name].append(result)
            
            logger.info(f"Order submitted: {result.id}")
            
            # Submit stop loss and take profit orders if provided
            if decision.stop_loss and side == 'buy':
                self._submit_bracket_orders(
                    symbol=decision.symbol,
                    quantity=quantity,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing decision: {e}")
            return None
    
    def _submit_bracket_orders(
        self,
        symbol: str,
        quantity: int,
        stop_loss: float,
        take_profit: float = None
    ):
        """Submit stop loss and take profit orders for risk management."""
        try:
            # Submit stop loss order
            self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                type='stop',
                stop_price=stop_loss,
                time_in_force='gtc'  # Good til cancelled
            )
            logger.info(f"Stop loss order submitted at ${stop_loss}")
            
            # Submit take profit order if specified
            if take_profit:
                self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='limit',
                    limit_price=take_profit,
                    time_in_force='gtc'
                )
                logger.info(f"Take profit order submitted at ${take_profit}")
                
        except Exception as e:
            logger.warning(f"Error submitting bracket orders: {e}")
    
    def _calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss: float
    ) -> int:
        """Calculate position size based on risk management."""
        max_position_value = portfolio_value * (settings.trading.max_position_size_pct / 100)
        max_risk = portfolio_value * (settings.trading.stop_loss_pct / 100)
        
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            risk_per_share = entry_price * (settings.trading.stop_loss_pct / 100)
        
        # Calculate shares based on risk
        risk_based_shares = int(max_risk / risk_per_share)
        
        # Calculate shares based on max position size
        size_based_shares = int(max_position_value / entry_price)
        
        return min(risk_based_shares, size_based_shares)
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            quote = self.api.get_latest_trade(symbol)
            return float(quote.price)
        except:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")
                if not data.empty:
                    return float(data['Close'].iloc[-1])
            except:
                pass
        return None
    
    def close_position(self, symbol: str) -> Optional[Order]:
        """Close an existing position."""
        if not self.api:
            return None
        
        try:
            order = self.api.close_position(symbol)
            return Order(
                id=order.id,
                symbol=symbol,
                side='sell',
                quantity=int(order.qty),
                order_type='market',
                status=OrderStatus.PENDING,
                submitted_at=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return None
    
    def close_all_positions(self) -> List[Order]:
        """Close all open positions."""
        if not self.api:
            return []
        
        try:
            orders = self.api.close_all_positions()
            return [
                Order(
                    id=o.id,
                    symbol=o.symbol,
                    side='sell',
                    quantity=int(o.qty),
                    order_type='market',
                    status=OrderStatus.PENDING,
                    submitted_at=datetime.now()
                )
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return []
    
    def get_order_history(self, limit: int = 50) -> List[Dict]:
        """Get recent order history."""
        if not self.api:
            return []
        
        try:
            orders = self.api.list_orders(status='all', limit=limit)
            return [
                {
                    "id": o.id,
                    "symbol": o.symbol,
                    "side": o.side,
                    "qty": int(o.qty),
                    "filled_qty": int(o.filled_qty or 0),
                    "type": o.type,
                    "status": o.status,
                    "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                    "submitted_at": o.submitted_at.isoformat() if o.submitted_at else None,
                    "filled_at": o.filled_at.isoformat() if o.filled_at else None,
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Error getting order history: {e}")
            return []
    
    def get_bot_performance(self, bot_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific bot."""
        orders = self.orders_by_bot.get(bot_name, [])
        
        if not orders:
            return {"bot_name": bot_name, "message": "No orders executed yet"}
        
        total_orders = len(orders)
        filled_orders = [o for o in orders if o.status == OrderStatus.FILLED]
        
        return {
            "bot_name": bot_name,
            "total_orders": total_orders,
            "filled_orders": len(filled_orders),
            "pending_orders": total_orders - len(filled_orders),
            "recent_orders": [o.to_dict() for o in orders[-5:]]
        }


def create_broker() -> AlpacaBroker:
    """Factory function to create a broker instance."""
    return AlpacaBroker()
