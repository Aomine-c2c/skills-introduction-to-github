"""
Simplified AI Trading Bot Demo
This version demonstrates the core concepts without external ML dependencies
"""
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math

class SimpleConfig:
    """Simplified configuration"""
    INITIAL_BALANCE = 10000
    MAX_POSITION_SIZE = 0.1
    STOP_LOSS_PERCENTAGE = 0.05
    TAKE_PROFIT_PERCENTAGE = 0.15
    DEFAULT_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    SIMULATION_MODE = True

class MockDataFetcher:
    """Mock data fetcher for demonstration"""
    
    def __init__(self):
        self.prices = {
            'AAPL': 150.0,
            'GOOGL': 2500.0,
            'MSFT': 300.0,
            'TSLA': 800.0,
            'AMZN': 3000.0
        }
        self.price_history = {symbol: [price] for symbol, price in self.prices.items()}
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price (simulated with random walk)"""
        if symbol not in self.prices:
            return 100.0
        
        # Random walk simulation
        change = random.gauss(0, 0.02)  # 2% daily volatility
        new_price = self.prices[symbol] * (1 + change)
        self.prices[symbol] = max(1.0, new_price)  # Ensure positive price
        
        # Store in history
        self.price_history[symbol].append(self.prices[symbol])
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
        
        return self.prices[symbol]
    
    def get_technical_indicators(self, symbol: str) -> Dict[str, float]:
        """Calculate simple technical indicators"""
        history = self.price_history.get(symbol, [100.0])
        current_price = history[-1]
        
        # Simple moving averages
        sma_10 = sum(history[-10:]) / min(10, len(history))
        sma_20 = sum(history[-20:]) / min(20, len(history))
        
        # Simple RSI approximation
        gains = []
        losses = []
        for i in range(1, min(15, len(history))):
            change = history[i] - history[i-1]
            if change > 0:
                gains.append(change)
            else:
                losses.append(abs(change))
        
        avg_gain = sum(gains) / len(gains) if gains else 0.01
        avg_loss = sum(losses) / len(losses) if losses else 0.01
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'price': current_price,
            'sma_10': sma_10,
            'sma_20': sma_20,
            'rsi': rsi,
            'trend': 'up' if current_price > sma_20 else 'down'
        }

class SimpleAIPredictor:
    """Simplified AI predictor using basic rules"""
    
    def __init__(self):
        self.is_trained = True  # Always "trained" for demo
    
    def predict(self, symbol: str, indicators: Dict[str, float]) -> Tuple[int, float]:
        """
        Simple prediction based on technical indicators
        Returns: (prediction, confidence) where prediction is 1=buy, 0=sell
        """
        price = indicators['price']
        sma_10 = indicators['sma_10']
        sma_20 = indicators['sma_20']
        rsi = indicators['rsi']
        
        # Simple decision logic
        score = 0.5  # Neutral starting point
        
        # Moving average signals
        if price > sma_10 > sma_20:
            score += 0.3  # Strong uptrend
        elif price < sma_10 < sma_20:
            score -= 0.3  # Strong downtrend
        elif price > sma_10:
            score += 0.1  # Weak uptrend
        else:
            score -= 0.1  # Weak downtrend
        
        # RSI signals
        if rsi < 30:
            score += 0.2  # Oversold
        elif rsi > 70:
            score -= 0.2  # Overbought
        
        # Add some randomness to simulate AI uncertainty
        score += random.gauss(0, 0.1)
        
        # Convert to binary prediction
        prediction = 1 if score > 0.5 else 0
        confidence = abs(score - 0.5) * 2  # Convert to 0-1 confidence
        confidence = min(0.95, max(0.05, confidence))  # Clamp confidence
        
        return prediction, confidence

class SimplePosition:
    """Simple position tracking"""
    
    def __init__(self, symbol: str, shares: float, entry_price: float):
        self.symbol = symbol
        self.shares = shares
        self.entry_price = entry_price
        self.entry_date = datetime.now()
        self.current_price = entry_price
    
    def update_price(self, price: float):
        self.current_price = price
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.shares
    
    @property
    def unrealized_pnl_percentage(self) -> float:
        return (self.current_price - self.entry_price) / self.entry_price

class SimpleTradingBot:
    """Simplified trading bot for demonstration"""
    
    def __init__(self):
        self.config = SimpleConfig()
        self.data_fetcher = MockDataFetcher()
        self.ai_predictor = SimpleAIPredictor()
        
        # Portfolio state
        self.cash_balance = self.config.INITIAL_BALANCE
        self.positions = {}  # symbol -> SimplePosition
        self.trades = []
        self.portfolio_history = []
        
        print("🤖 Simple AI Trading Bot initialized")
        print(f"💰 Starting balance: ${self.cash_balance:,.2f}")
        print(f"📈 Tracking symbols: {self.config.DEFAULT_SYMBOLS}")
    
    def run_trading_cycle(self):
        """Run one trading cycle"""
        print(f"\n{'='*50}")
        print(f"🔄 Trading Cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print('='*50)
        
        signals_generated = 0
        trades_executed = 0
        
        # Update prices for existing positions
        current_prices = {}
        for symbol in self.config.DEFAULT_SYMBOLS:
            current_prices[symbol] = self.data_fetcher.get_current_price(symbol)
            if symbol in self.positions:
                self.positions[symbol].update_price(current_prices[symbol])
        
        # Check stop loss / take profit
        positions_to_close = []
        for symbol, position in self.positions.items():
            stop_loss_price = position.entry_price * (1 - self.config.STOP_LOSS_PERCENTAGE)
            take_profit_price = position.entry_price * (1 + self.config.TAKE_PROFIT_PERCENTAGE)
            
            if position.current_price <= stop_loss_price:
                print(f"🛑 Stop loss triggered for {symbol}")
                positions_to_close.append(symbol)
            elif position.current_price >= take_profit_price:
                print(f"🎯 Take profit triggered for {symbol}")
                positions_to_close.append(symbol)
        
        # Close triggered positions
        for symbol in positions_to_close:
            self._close_position(symbol)
            trades_executed += 1
        
        # Generate new signals
        for symbol in self.config.DEFAULT_SYMBOLS:
            # Skip if we already have a position
            if symbol in self.positions:
                continue
            
            # Get technical indicators
            indicators = self.data_fetcher.get_technical_indicators(symbol)
            
            # Get AI prediction
            prediction, confidence = self.ai_predictor.predict(symbol, indicators)
            signals_generated += 1
            
            # Display signal info
            signal_type = "🟢 BUY" if prediction == 1 else "🔴 SELL"
            print(f"{signal_type} {symbol}: ${indicators['price']:.2f} (confidence: {confidence:.1%})")
            print(f"  📊 SMA10: ${indicators['sma_10']:.2f}, SMA20: ${indicators['sma_20']:.2f}, RSI: {indicators['rsi']:.1f}")
            
            # Execute trade if confidence is high enough (lowered threshold for demo)
            if prediction == 1 and confidence > 0.3 and len(self.positions) < 3:  # Max 3 positions
                self._open_position(symbol, indicators['price'], confidence)
                trades_executed += 1
        
        # Update portfolio history
        total_value = self._calculate_total_value()
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'total_value': total_value,
            'cash_balance': self.cash_balance,
            'positions_count': len(self.positions)
        })
        
        # Print cycle summary
        print(f"\n📊 Cycle Summary:")
        print(f"   Signals generated: {signals_generated}")
        print(f"   Trades executed: {trades_executed}")
        print(f"   Portfolio value: ${total_value:,.2f}")
        print(f"   Open positions: {len(self.positions)}")
        
        return {
            'signals_generated': signals_generated,
            'trades_executed': trades_executed,
            'portfolio_value': total_value
        }
    
    def _open_position(self, symbol: str, price: float, confidence: float):
        """Open a new position"""
        # Calculate position size
        portfolio_value = self._calculate_total_value()
        max_position_value = portfolio_value * self.config.MAX_POSITION_SIZE
        
        # Adjust for confidence (higher confidence = larger position)
        position_value = max_position_value * confidence
        shares = int(position_value / price)
        
        if shares > 0 and shares * price <= self.cash_balance:
            self.positions[symbol] = SimplePosition(symbol, shares, price)
            self.cash_balance -= shares * price
            
            trade = {
                'symbol': symbol,
                'action': 'BUY',
                'shares': shares,
                'price': price,
                'timestamp': datetime.now(),
                'confidence': confidence
            }
            self.trades.append(trade)
            
            print(f"💰 Bought {shares} shares of {symbol} at ${price:.2f} (${shares * price:,.2f})")
    
    def _close_position(self, symbol: str):
        """Close an existing position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        proceeds = position.shares * position.current_price
        self.cash_balance += proceeds
        
        pnl = position.unrealized_pnl
        pnl_pct = position.unrealized_pnl_percentage
        
        trade = {
            'symbol': symbol,
            'action': 'SELL',
            'shares': position.shares,
            'price': position.current_price,
            'timestamp': datetime.now(),
            'pnl': pnl,
            'pnl_percentage': pnl_pct
        }
        self.trades.append(trade)
        
        pnl_emoji = "📈" if pnl > 0 else "📉"
        print(f"{pnl_emoji} Sold {position.shares} shares of {symbol} at ${position.current_price:.2f}")
        print(f"   P&L: ${pnl:+.2f} ({pnl_pct:+.1%})")
        
        del self.positions[symbol]
    
    def _calculate_total_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash_balance + positions_value
    
    def print_portfolio_status(self):
        """Print current portfolio status"""
        total_value = self._calculate_total_value()
        total_return = (total_value - self.config.INITIAL_BALANCE) / self.config.INITIAL_BALANCE
        
        print(f"\n{'='*60}")
        print("📊 PORTFOLIO STATUS")
        print('='*60)
        print(f"💰 Total Value: ${total_value:,.2f}")
        print(f"💵 Cash Balance: ${self.cash_balance:,.2f}")
        print(f"📈 Total Return: {total_return:+.2%}")
        print(f"📋 Open Positions: {len(self.positions)}")
        
        if self.positions:
            print("\n🏢 Current Positions:")
            for symbol, position in self.positions.items():
                pnl_pct = position.unrealized_pnl_percentage
                status_emoji = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"
                print(f"   {status_emoji} {symbol}: {position.shares} shares @ ${position.current_price:.2f} ({pnl_pct:+.1%})")
        
        if self.trades:
            recent_trades = self.trades[-5:]  # Last 5 trades
            print(f"\n📜 Recent Trades:")
            for trade in recent_trades:
                action_emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
                print(f"   {action_emoji} {trade['action']} {trade['shares']} {trade['symbol']} @ ${trade['price']:.2f}")
        
        print('='*60)
    
    def run_demo(self, cycles: int = 10):
        """Run a demonstration of the trading bot"""
        print("🚀 Starting AI Trading Bot Demo")
        print(f"🔄 Will run {cycles} trading cycles")
        
        try:
            for i in range(cycles):
                print(f"\n⏰ Cycle {i+1}/{cycles}")
                self.run_trading_cycle()
                
                if i % 3 == 2:  # Print status every 3 cycles
                    self.print_portfolio_status()
                
                # Simulate time passing
                time.sleep(1)
            
            # Final status
            print(f"\n🏁 Demo completed!")
            self.print_portfolio_status()
            
            # Performance summary
            total_value = self._calculate_total_value()
            total_return = (total_value - self.config.INITIAL_BALANCE) / self.config.INITIAL_BALANCE
            total_trades = len(self.trades)
            profitable_trades = len([t for t in self.trades if t.get('pnl', 0) > 0])
            win_rate = profitable_trades / (total_trades // 2) if total_trades > 0 else 0  # Divide by 2 for buy/sell pairs
            
            print(f"\n📊 FINAL PERFORMANCE SUMMARY")
            print('='*60)
            print(f"💰 Final Value: ${total_value:,.2f}")
            print(f"📈 Total Return: {total_return:+.2%}")
            print(f"📋 Total Trades: {total_trades}")
            print(f"🎯 Win Rate: {win_rate:.1%}")
            print('='*60)
            
        except KeyboardInterrupt:
            print("\n⏹️  Demo stopped by user")
            self.print_portfolio_status()

def main():
    """Main demonstration function"""
    print("🤖 AI Trading Bot - Simplified Demo")
    print("="*50)
    print("This is a demonstration of an AI trading bot using")
    print("simulated data and simplified algorithms.")
    print("="*50)
    
    # Create and run the bot
    bot = SimpleTradingBot()
    
    print("\nChoose demo mode:")
    print("1. Quick demo (5 cycles)")
    print("2. Extended demo (15 cycles)")
    print("3. Single cycle test")
    print("4. Manual cycle-by-cycle")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            bot.run_demo(5)
        elif choice == "2":
            bot.run_demo(15)
        elif choice == "3":
            bot.run_trading_cycle()
            bot.print_portfolio_status()
        elif choice == "4":
            while True:
                input("\nPress Enter to run next cycle (Ctrl+C to stop)...")
                bot.run_trading_cycle()
                bot.print_portfolio_status()
        else:
            print("Running default demo...")
            bot.run_demo(10)
            
    except KeyboardInterrupt:
        print("\n👋 Demo ended. Thanks for trying the AI Trading Bot!")

if __name__ == "__main__":
    main()