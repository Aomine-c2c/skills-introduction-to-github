"""
Main AI Trading Bot Application
"""
import logging
import coloredlogs
import time
import schedule
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

from config import Config
from data_fetcher import DataFetcher
from ai_predictor import AIPredictor
from trading_strategy import TradingStrategy
from portfolio import Portfolio
from risk_manager import RiskManager

class TradingBot:
    """Main AI Trading Bot class"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Initialize the trading bot
        
        Args:
            config_override: Optional configuration overrides
        """
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self.config = Config()
        if config_override:
            for key, value in config_override.items():
                setattr(self.config, key, value)
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.ai_predictor = AIPredictor()
        self.trading_strategy = TradingStrategy(self.config)
        self.portfolio = Portfolio(
            initial_balance=self.config.INITIAL_BALANCE,
            commission_per_trade=0.0  # Assuming commission-free trading
        )
        self.risk_manager = RiskManager(self.config)
        
        # State variables
        self.is_running = False
        self.last_training_date = None
        self.model_trained = False
        self.symbols = self.config.DEFAULT_SYMBOLS.copy()
        
        # Performance tracking
        self.daily_reports = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("AI Trading Bot initialized")
        
        # Load existing portfolio state if available
        if os.path.exists('portfolio_state.json'):
            self.portfolio.load_portfolio_state('portfolio_state.json')
            self.logger.info("Loaded existing portfolio state")
        
        # Load existing model if available
        if os.path.exists('ai_model.joblib'):
            self.ai_predictor.load_model('ai_model.joblib')
            self.model_trained = True
            self.logger.info("Loaded existing AI model")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # File handler
        file_handler = logging.FileHandler('logs/trading_bot.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[file_handler, console_handler]
        )
        
        # Setup colored logging for console
        coloredlogs.install(
            level='INFO',
            fmt=log_format,
            field_styles={
                'levelname': {'bold': True},
                'name': {'color': 'blue'},
                'asctime': {'color': 'green'}
            }
        )
    
    def train_model(self, retrain: bool = False) -> bool:
        """
        Train the AI prediction model
        
        Args:
            retrain: Force retraining even if model exists
            
        Returns:
            True if training successful
        """
        try:
            if self.model_trained and not retrain:
                self.logger.info("Model already trained. Use retrain=True to force retraining.")
                return True
            
            self.logger.info("Starting AI model training...")
            
            # Fetch training data for all symbols
            training_data = {}
            for symbol in self.symbols:
                self.logger.info(f"Fetching training data for {symbol}")
                data = self.data_fetcher.fetch_stock_data(
                    symbol, 
                    period="2y",  # 2 years of data for training
                    interval=self.config.DATA_INTERVAL
                )
                
                if data is not None and not data.empty:
                    # Add technical indicators
                    data_with_indicators = self.data_fetcher.get_technical_indicators(data)
                    training_data[symbol] = data_with_indicators
                    self.logger.info(f"Successfully prepared {len(data_with_indicators)} data points for {symbol}")
                else:
                    self.logger.warning(f"Failed to fetch data for {symbol}")
            
            if not training_data:
                self.logger.error("No training data available")
                return False
            
            # Combine data from all symbols for training
            combined_data = []
            for symbol, data in training_data.items():
                if len(data) > 50:  # Minimum data requirement
                    combined_data.append(data)
            
            if not combined_data:
                self.logger.error("Insufficient data for training")
                return False
            
            # Use the largest dataset for training (or combine them)
            largest_dataset = max(combined_data, key=len)
            
            # Train the model
            training_results = self.ai_predictor.train(largest_dataset)
            
            if training_results:
                self.model_trained = True
                self.last_training_date = datetime.now()
                
                # Save the trained model
                self.ai_predictor.save_model('ai_model.joblib')
                
                self.logger.info("AI model training completed successfully!")
                self.logger.info(f"Training metrics: {training_results}")
                
                # Log feature importance if available
                feature_importance = self.ai_predictor.get_feature_importance()
                if feature_importance:
                    self.logger.info("Top 5 most important features:")
                    for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
                        self.logger.info(f"  {i+1}. {feature}: {importance:.4f}")
                
                return True
            else:
                self.logger.error("Model training failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            return False
    
    def run_trading_cycle(self) -> Dict:
        """
        Run a single trading cycle
        
        Returns:
            Dictionary with cycle results
        """
        try:
            self.logger.info("Starting trading cycle...")
            cycle_start_time = datetime.now()
            cycle_results = {
                'timestamp': cycle_start_time,
                'signals_generated': 0,
                'trades_executed': 0,
                'positions_closed': 0,
                'errors': []
            }
            
            # Check if model is trained
            if not self.model_trained:
                self.logger.warning("Model not trained. Running training first...")
                if not self.train_model():
                    cycle_results['errors'].append("Model training failed")
                    return cycle_results
            
            # Fetch current market data
            current_data = {}
            current_prices = {}
            
            for symbol in self.symbols:
                data = self.data_fetcher.fetch_stock_data(
                    symbol,
                    period="3mo",  # 3 months for current analysis
                    interval=self.config.DATA_INTERVAL
                )
                
                if data is not None and not data.empty:
                    # Add technical indicators
                    data_with_indicators = self.data_fetcher.get_technical_indicators(data)
                    current_data[symbol] = data_with_indicators
                    current_prices[symbol] = data_with_indicators['Close'].iloc[-1]
                    self.logger.info(f"Fetched data for {symbol}: ${current_prices[symbol]:.2f}")
                else:
                    self.logger.warning(f"Failed to fetch current data for {symbol}")
                    cycle_results['errors'].append(f"Data fetch failed for {symbol}")
            
            if not current_data:
                self.logger.error("No current market data available")
                cycle_results['errors'].append("No market data available")
                return cycle_results
            
            # Update portfolio with current prices
            self.portfolio.update_prices(current_prices)
            
            # Check for stop loss / take profit triggers
            symbols_to_close = self.portfolio.check_stop_loss_take_profit()
            for symbol in symbols_to_close:
                if self.portfolio.close_position(symbol, price=current_prices.get(symbol)):
                    cycle_results['positions_closed'] += 1
                    self.logger.info(f"Closed position for {symbol} due to stop loss/take profit")
            
            # Assess portfolio risk
            risk_metrics = self.risk_manager.assess_portfolio_risk(self.portfolio, current_data)
            should_stop_trading, stop_reasons = self.risk_manager.should_stop_trading(self.portfolio, risk_metrics)
            
            if should_stop_trading:
                self.logger.warning(f"Trading stopped due to risk: {stop_reasons}")
                cycle_results['trading_stopped'] = True
                cycle_results['stop_reasons'] = stop_reasons
                return cycle_results
            
            # Generate trading signals for each symbol
            for symbol in self.symbols:
                if symbol not in current_data:
                    continue
                
                try:
                    # Skip if we already have a position and max positions reached
                    if (len(self.portfolio.positions) >= self.config.MAX_OPEN_POSITIONS and 
                        symbol not in self.portfolio.positions):
                        continue
                    
                    # Get AI prediction
                    ai_prediction = self.ai_predictor.predict(current_data[symbol])
                    
                    # Generate trading signal
                    signal = self.trading_strategy.generate_signal(symbol, current_data[symbol], ai_prediction)
                    cycle_results['signals_generated'] += 1
                    
                    self.logger.info(f"Signal for {symbol}: {signal.signal_type.value} (confidence: {signal.confidence:.3f})")
                    
                    # Execute trades based on signals
                    if signal.signal_type.value == "BUY" and symbol not in self.portfolio.positions:
                        # Calculate position size
                        portfolio_value = self.portfolio.get_total_value()
                        
                        # Calculate volatility for position sizing
                        returns = current_data[symbol]['Close'].pct_change().tail(20)
                        volatility = returns.std() if len(returns) > 1 else 0.02
                        
                        shares = self.risk_manager.calculate_position_size(
                            portfolio_value, 
                            signal.price, 
                            volatility, 
                            signal.confidence
                        )
                        
                        if shares > 0:
                            success = self.portfolio.add_position(
                                symbol, 
                                shares, 
                                signal.price,
                                signal.stop_loss,
                                signal.take_profit
                            )
                            
                            if success:
                                cycle_results['trades_executed'] += 1
                                self.logger.info(f"Bought {shares} shares of {symbol} at ${signal.price:.2f}")
                    
                    elif signal.signal_type.value == "SELL" and symbol in self.portfolio.positions:
                        success = self.portfolio.close_position(symbol, price=signal.price)
                        if success:
                            cycle_results['trades_executed'] += 1
                            self.logger.info(f"Sold position in {symbol} at ${signal.price:.2f}")
                
                except Exception as e:
                    error_msg = f"Error processing {symbol}: {str(e)}"
                    self.logger.error(error_msg)
                    cycle_results['errors'].append(error_msg)
            
            # Update cycle results with portfolio metrics
            portfolio_metrics = self.portfolio.get_portfolio_metrics()
            cycle_results.update(portfolio_metrics)
            
            # Save portfolio state
            self.portfolio.save_portfolio_state('portfolio_state.json')
            
            cycle_duration = (datetime.now() - cycle_start_time).total_seconds()
            self.logger.info(f"Trading cycle completed in {cycle_duration:.2f} seconds")
            
            return cycle_results
            
        except Exception as e:
            error_msg = f"Error in trading cycle: {str(e)}"
            self.logger.error(error_msg)
            return {'timestamp': datetime.now(), 'errors': [error_msg]}
    
    def generate_daily_report(self) -> Dict:
        """
        Generate daily performance report
        
        Returns:
            Dictionary with daily report
        """
        try:
            portfolio_metrics = self.portfolio.get_portfolio_metrics()
            position_summary = self.portfolio.get_position_summary()
            
            # Get recent trading signals summary
            signals_summary = self.trading_strategy.get_signals_summary(days=1)
            
            # Get risk assessment
            current_data = {}
            for symbol in self.symbols:
                data = self.data_fetcher.fetch_stock_data(symbol, period="1mo")
                if data is not None:
                    current_data[symbol] = self.data_fetcher.get_technical_indicators(data)
            
            risk_metrics = self.risk_manager.assess_portfolio_risk(self.portfolio, current_data)
            risk_summary = self.risk_manager.get_risk_summary(risk_metrics)
            
            report = {
                'date': datetime.now().date().isoformat(),
                'portfolio_metrics': portfolio_metrics,
                'positions': position_summary,
                'signals_summary': signals_summary,
                'risk_assessment': risk_summary,
                'model_status': {
                    'trained': self.model_trained,
                    'last_training': self.last_training_date.isoformat() if self.last_training_date else None
                }
            }
            
            self.daily_reports.append(report)
            
            # Save report to file
            with open(f'logs/daily_report_{datetime.now().date()}.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info("Daily report generated")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating daily report: {str(e)}")
            return {}
    
    def print_status(self):
        """Print current bot status"""
        try:
            print("\n" + "="*60)
            print("AI TRADING BOT STATUS")
            print("="*60)
            
            # Portfolio status
            total_value = self.portfolio.get_total_value()
            total_return = (total_value - self.config.INITIAL_BALANCE) / self.config.INITIAL_BALANCE
            
            print(f"Portfolio Value: ${total_value:,.2f}")
            print(f"Cash Balance: ${self.portfolio.cash_balance:,.2f}")
            print(f"Total Return: {total_return:.2%}")
            print(f"Max Drawdown: {self.portfolio.max_drawdown:.2%}")
            
            # Positions
            print(f"\nOpen Positions: {len(self.portfolio.positions)}")
            for symbol, position in self.portfolio.positions.items():
                pnl_pct = position.unrealized_pnl_percentage
                status = "🟢" if pnl_pct > 0 else "🔴" if pnl_pct < 0 else "⚪"
                print(f"  {status} {symbol}: {position.shares:.0f} shares @ ${position.current_price:.2f} ({pnl_pct:+.1%})")
            
            # Model status
            print(f"\nAI Model: {'✅ Trained' if self.model_trained else '❌ Not Trained'}")
            if self.last_training_date:
                days_since_training = (datetime.now() - self.last_training_date).days
                print(f"Last Training: {days_since_training} days ago")
            
            # Recent performance
            if len(self.portfolio.daily_returns) > 0:
                recent_returns = self.portfolio.daily_returns[-5:]  # Last 5 days
                avg_daily_return = sum(recent_returns) / len(recent_returns)
                print(f"Avg Daily Return (5d): {avg_daily_return:.2%}")
            
            print("="*60 + "\n")
            
        except Exception as e:
            self.logger.error(f"Error printing status: {str(e)}")
    
    def start_bot(self, mode: str = "simulation"):
        """
        Start the trading bot
        
        Args:
            mode: 'simulation' or 'live' (always use simulation for safety)
        """
        try:
            if mode != "simulation":
                self.logger.warning("Only simulation mode is supported for safety")
                mode = "simulation"
            
            self.logger.info(f"Starting AI Trading Bot in {mode} mode...")
            
            # Initial training if needed
            if not self.model_trained:
                self.logger.info("Training AI model...")
                if not self.train_model():
                    self.logger.error("Failed to train model. Exiting.")
                    return
            
            # Schedule trading cycles
            schedule.every(1).hours.do(self.run_trading_cycle)  # Run every hour
            schedule.every().day.at("09:00").do(self.generate_daily_report)  # Daily report at 9 AM
            schedule.every().week.do(lambda: self.train_model(retrain=True))  # Retrain weekly
            
            self.is_running = True
            self.logger.info("AI Trading Bot started successfully!")
            
            # Print initial status
            self.print_status()
            
            # Main loop
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                    
                    # Print status every hour
                    if datetime.now().minute == 0:
                        self.print_status()
                        
                except KeyboardInterrupt:
                    self.logger.info("Received stop signal...")
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {str(e)}")
                    time.sleep(60)  # Wait before retrying
            
        except Exception as e:
            self.logger.error(f"Error starting bot: {str(e)}")
        finally:
            self.stop_bot()
    
    def stop_bot(self):
        """Stop the trading bot"""
        try:
            self.is_running = False
            
            # Save final state
            self.portfolio.save_portfolio_state('portfolio_state.json')
            
            # Generate final report
            final_report = self.generate_daily_report()
            
            self.logger.info("AI Trading Bot stopped")
            print("\nBot stopped. Final status:")
            self.print_status()
            
        except Exception as e:
            self.logger.error(f"Error stopping bot: {str(e)}")

def main():
    """Main function to run the trading bot"""
    try:
        # Create and start the trading bot
        bot = TradingBot()
        
        print("AI Trading Bot")
        print("=" * 50)
        print("1. Train model and run single cycle")
        print("2. Start continuous trading (simulation)")
        print("3. Generate daily report")
        print("4. Show status")
        print("5. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                print("\nTraining model and running single cycle...")
                bot.train_model(retrain=True)
                results = bot.run_trading_cycle()
                print(f"Cycle completed: {results}")
                bot.print_status()
                
            elif choice == "2":
                print("\nStarting continuous trading in simulation mode...")
                print("Press Ctrl+C to stop the bot")
                bot.start_bot("simulation")
                
            elif choice == "3":
                print("\nGenerating daily report...")
                report = bot.generate_daily_report()
                print("Report generated and saved to logs/")
                
            elif choice == "4":
                bot.print_status()
                
            elif choice == "5":
                print("Exiting...")
                bot.stop_bot()
                break
                
            else:
                print("Invalid choice. Please enter 1-5.")
    
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()