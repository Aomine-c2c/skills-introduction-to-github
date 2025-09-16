"""
Portfolio management and tracking for AI Trading Bot
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    shares: float
    entry_price: float
    entry_date: datetime
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.shares * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        return (self.current_price - self.entry_price) * self.shares
    
    @property
    def unrealized_pnl_percentage(self) -> float:
        """Unrealized profit/loss as percentage"""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price
    
    def to_dict(self) -> dict:
        """Convert position to dictionary"""
        return asdict(self)

@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    action: str  # BUY or SELL
    shares: float
    price: float
    date: datetime
    commission: float = 0.0
    
    @property
    def value(self) -> float:
        """Trade value before commission"""
        return self.shares * self.price
    
    @property
    def net_value(self) -> float:
        """Trade value after commission"""
        return self.value - self.commission
    
    def to_dict(self) -> dict:
        """Convert trade to dictionary"""
        return asdict(self)

class Portfolio:
    """Portfolio management class"""
    
    def __init__(self, initial_balance: float = 10000, commission_per_trade: float = 0.0):
        """
        Initialize portfolio
        
        Args:
            initial_balance: Starting cash balance
            commission_per_trade: Commission fee per trade
        """
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.commission_per_trade = commission_per_trade
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_history: List[Dict] = []
        
        # Performance tracking
        self.daily_returns: List[float] = []
        self.peak_value = initial_balance
        self.max_drawdown = 0.0
        
        logger.info(f"Portfolio initialized with ${initial_balance:,.2f}")
    
    def add_position(self, symbol: str, shares: float, price: float, 
                    stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> bool:
        """
        Add a new position to portfolio
        
        Args:
            symbol: Stock symbol
            shares: Number of shares
            price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            True if position added successfully
        """
        try:
            total_cost = shares * price + self.commission_per_trade
            
            if total_cost > self.cash_balance:
                logger.warning(f"Insufficient funds to buy {shares} shares of {symbol}")
                return False
            
            if symbol in self.positions:
                # Add to existing position
                existing_position = self.positions[symbol]
                total_shares = existing_position.shares + shares
                avg_price = ((existing_position.shares * existing_position.entry_price) + (shares * price)) / total_shares
                
                self.positions[symbol].shares = total_shares
                self.positions[symbol].entry_price = avg_price
                self.positions[symbol].entry_date = datetime.now()
            else:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    shares=shares,
                    entry_price=price,
                    entry_date=datetime.now(),
                    current_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            
            # Update cash balance
            self.cash_balance -= total_cost
            
            # Record trade
            trade = Trade(
                symbol=symbol,
                action="BUY",
                shares=shares,
                price=price,
                date=datetime.now(),
                commission=self.commission_per_trade
            )
            self.trades.append(trade)
            
            logger.info(f"Added position: {shares} shares of {symbol} at ${price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding position for {symbol}: {str(e)}")
            return False
    
    def close_position(self, symbol: str, shares: Optional[float] = None, price: Optional[float] = None) -> bool:
        """
        Close position (partial or full)
        
        Args:
            symbol: Stock symbol
            shares: Number of shares to sell (None for all)
            price: Sell price (None to use current price)
            
        Returns:
            True if position closed successfully
        """
        try:
            if symbol not in self.positions:
                logger.warning(f"No position found for {symbol}")
                return False
            
            position = self.positions[symbol]
            
            # Determine shares to sell
            if shares is None:
                shares_to_sell = position.shares
            else:
                shares_to_sell = min(shares, position.shares)
            
            # Determine sell price
            if price is None:
                sell_price = position.current_price
            else:
                sell_price = price
            
            # Calculate proceeds
            gross_proceeds = shares_to_sell * sell_price
            net_proceeds = gross_proceeds - self.commission_per_trade
            
            # Update cash balance
            self.cash_balance += net_proceeds
            
            # Update position
            if shares_to_sell >= position.shares:
                # Close entire position
                del self.positions[symbol]
            else:
                # Partial close
                position.shares -= shares_to_sell
            
            # Record trade
            trade = Trade(
                symbol=symbol,
                action="SELL",
                shares=shares_to_sell,
                price=sell_price,
                date=datetime.now(),
                commission=self.commission_per_trade
            )
            self.trades.append(trade)
            
            # Calculate realized P&L
            realized_pnl = (sell_price - position.entry_price) * shares_to_sell - self.commission_per_trade
            
            logger.info(f"Closed position: {shares_to_sell} shares of {symbol} at ${sell_price:.2f} (P&L: ${realized_pnl:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {str(e)}")
            return False
    
    def update_prices(self, price_data: Dict[str, float]) -> None:
        """
        Update current prices for all positions
        
        Args:
            price_data: Dictionary with symbol: price pairs
        """
        try:
            for symbol, position in self.positions.items():
                if symbol in price_data:
                    position.current_price = price_data[symbol]
            
            # Update portfolio history
            self._update_portfolio_history()
            
        except Exception as e:
            logger.error(f"Error updating prices: {str(e)}")
    
    def _update_portfolio_history(self) -> None:
        """Update portfolio performance history"""
        try:
            total_value = self.get_total_value()
            
            # Calculate daily return
            if self.portfolio_history:
                previous_value = self.portfolio_history[-1]['total_value']
                daily_return = (total_value - previous_value) / previous_value
                self.daily_returns.append(daily_return)
            
            # Update peak value and max drawdown
            if total_value > self.peak_value:
                self.peak_value = total_value
            
            current_drawdown = (self.peak_value - total_value) / self.peak_value
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # Record portfolio snapshot
            snapshot = {
                'timestamp': datetime.now(),
                'cash_balance': self.cash_balance,
                'total_value': total_value,
                'positions_value': self.get_positions_value(),
                'unrealized_pnl': self.get_unrealized_pnl(),
                'daily_return': self.daily_returns[-1] if self.daily_returns else 0.0,
                'total_return': (total_value - self.initial_balance) / self.initial_balance,
                'max_drawdown': self.max_drawdown
            }
            
            self.portfolio_history.append(snapshot)
            
        except Exception as e:
            logger.error(f"Error updating portfolio history: {str(e)}")
    
    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions)"""
        return self.cash_balance + self.get_positions_value()
    
    def get_positions_value(self) -> float:
        """Get total value of all positions"""
        return sum([position.market_value for position in self.positions.values()])
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized profit/loss"""
        return sum([position.unrealized_pnl for position in self.positions.values()])
    
    def get_realized_pnl(self) -> float:
        """Get total realized profit/loss from completed trades"""
        try:
            realized_pnl = 0.0
            
            # Group trades by symbol to calculate realized P&L
            trades_by_symbol = {}
            for trade in self.trades:
                if trade.symbol not in trades_by_symbol:
                    trades_by_symbol[trade.symbol] = []
                trades_by_symbol[trade.symbol].append(trade)
            
            for symbol, symbol_trades in trades_by_symbol.items():
                # Calculate realized P&L using FIFO method
                buys = [t for t in symbol_trades if t.action == "BUY"]
                sells = [t for t in symbol_trades if t.action == "SELL"]
                
                buy_queue = []
                for buy in buys:
                    buy_queue.append({'shares': buy.shares, 'price': buy.price})
                
                for sell in sells:
                    remaining_sell_shares = sell.shares
                    
                    while remaining_sell_shares > 0 and buy_queue:
                        buy_lot = buy_queue[0]
                        
                        if buy_lot['shares'] <= remaining_sell_shares:
                            # Use entire buy lot
                            pnl = (sell.price - buy_lot['price']) * buy_lot['shares']
                            realized_pnl += pnl
                            remaining_sell_shares -= buy_lot['shares']
                            buy_queue.pop(0)
                        else:
                            # Partial use of buy lot
                            pnl = (sell.price - buy_lot['price']) * remaining_sell_shares
                            realized_pnl += pnl
                            buy_lot['shares'] -= remaining_sell_shares
                            remaining_sell_shares = 0
            
            return realized_pnl
            
        except Exception as e:
            logger.error(f"Error calculating realized P&L: {str(e)}")
            return 0.0
    
    def get_portfolio_metrics(self) -> Dict:
        """Get comprehensive portfolio performance metrics"""
        try:
            total_value = self.get_total_value()
            total_return = (total_value - self.initial_balance) / self.initial_balance
            
            # Calculate Sharpe ratio (simplified)
            if len(self.daily_returns) > 1:
                returns_std = np.std(self.daily_returns)
                avg_return = np.mean(self.daily_returns)
                sharpe_ratio = (avg_return / returns_std) * np.sqrt(252) if returns_std > 0 else 0  # Annualized
            else:
                sharpe_ratio = 0.0
            
            # Calculate win rate
            profitable_trades = 0
            total_completed_trades = 0
            
            for symbol in set([t.symbol for t in self.trades]):
                symbol_trades = [t for t in self.trades if t.symbol == symbol]
                buys = [t for t in symbol_trades if t.action == "BUY"]
                sells = [t for t in symbol_trades if t.action == "SELL"]
                
                for sell in sells:
                    # Find corresponding buy (simplified FIFO)
                    for buy in buys:
                        if buy.date <= sell.date:
                            total_completed_trades += 1
                            if sell.price > buy.price:
                                profitable_trades += 1
                            break
            
            win_rate = profitable_trades / total_completed_trades if total_completed_trades > 0 else 0.0
            
            metrics = {
                'total_value': total_value,
                'cash_balance': self.cash_balance,
                'positions_value': self.get_positions_value(),
                'total_return': total_return,
                'realized_pnl': self.get_realized_pnl(),
                'unrealized_pnl': self.get_unrealized_pnl(),
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'total_trades': len(self.trades),
                'open_positions': len(self.positions),
                'days_active': len(self.portfolio_history)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}
    
    def get_position_summary(self) -> List[Dict]:
        """Get summary of all current positions"""
        try:
            summary = []
            for symbol, position in self.positions.items():
                summary.append({
                    'symbol': symbol,
                    'shares': position.shares,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'unrealized_pnl_pct': position.unrealized_pnl_percentage,
                    'days_held': (datetime.now() - position.entry_date).days,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting position summary: {str(e)}")
            return []
    
    def check_stop_loss_take_profit(self) -> List[str]:
        """
        Check if any positions should be closed due to stop loss or take profit
        
        Returns:
            List of symbols that should be closed
        """
        symbols_to_close = []
        
        try:
            for symbol, position in self.positions.items():
                if position.stop_loss and position.current_price <= position.stop_loss:
                    logger.warning(f"Stop loss triggered for {symbol}: ${position.current_price:.2f} <= ${position.stop_loss:.2f}")
                    symbols_to_close.append(symbol)
                elif position.take_profit and position.current_price >= position.take_profit:
                    logger.info(f"Take profit triggered for {symbol}: ${position.current_price:.2f} >= ${position.take_profit:.2f}")
                    symbols_to_close.append(symbol)
            
            return symbols_to_close
            
        except Exception as e:
            logger.error(f"Error checking stop loss/take profit: {str(e)}")
            return []
    
    def save_portfolio_state(self, filepath: str) -> bool:
        """
        Save portfolio state to file
        
        Args:
            filepath: Path to save portfolio state
            
        Returns:
            True if successful
        """
        try:
            portfolio_data = {
                'initial_balance': self.initial_balance,
                'cash_balance': self.cash_balance,
                'commission_per_trade': self.commission_per_trade,
                'positions': {symbol: position.to_dict() for symbol, position in self.positions.items()},
                'trades': [trade.to_dict() for trade in self.trades],
                'portfolio_history': self.portfolio_history,
                'peak_value': self.peak_value,
                'max_drawdown': self.max_drawdown,
                'save_timestamp': datetime.now().isoformat()
            }
            
            # Convert datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                return obj
            
            portfolio_data = convert_datetime(portfolio_data)
            
            with open(filepath, 'w') as f:
                json.dump(portfolio_data, f, indent=2)
            
            logger.info(f"Portfolio state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving portfolio state: {str(e)}")
            return False
    
    def load_portfolio_state(self, filepath: str) -> bool:
        """
        Load portfolio state from file
        
        Args:
            filepath: Path to load portfolio state from
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                portfolio_data = json.load(f)
            
            # Restore basic attributes
            self.initial_balance = portfolio_data['initial_balance']
            self.cash_balance = portfolio_data['cash_balance']
            self.commission_per_trade = portfolio_data['commission_per_trade']
            self.peak_value = portfolio_data['peak_value']
            self.max_drawdown = portfolio_data['max_drawdown']
            
            # Restore positions
            self.positions = {}
            for symbol, pos_data in portfolio_data['positions'].items():
                self.positions[symbol] = Position(
                    symbol=pos_data['symbol'],
                    shares=pos_data['shares'],
                    entry_price=pos_data['entry_price'],
                    entry_date=datetime.fromisoformat(pos_data['entry_date']),
                    current_price=pos_data['current_price'],
                    stop_loss=pos_data['stop_loss'],
                    take_profit=pos_data['take_profit']
                )
            
            # Restore trades
            self.trades = []
            for trade_data in portfolio_data['trades']:
                self.trades.append(Trade(
                    symbol=trade_data['symbol'],
                    action=trade_data['action'],
                    shares=trade_data['shares'],
                    price=trade_data['price'],
                    date=datetime.fromisoformat(trade_data['date']),
                    commission=trade_data['commission']
                ))
            
            # Restore portfolio history
            self.portfolio_history = portfolio_data['portfolio_history']
            
            logger.info(f"Portfolio state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading portfolio state: {str(e)}")
            return False