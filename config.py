"""
Configuration management for AI Trading Bot
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for trading bot settings"""
    
    # Trading parameters
    INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', '10000'))
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))  # 10% of portfolio
    STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', '0.05'))  # 5%
    TAKE_PROFIT_PERCENTAGE = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '0.15'))  # 15%
    
    # AI/ML parameters
    LOOKBACK_PERIOD = int(os.getenv('LOOKBACK_PERIOD', '60'))  # Days for training data
    PREDICTION_CONFIDENCE_THRESHOLD = float(os.getenv('PREDICTION_CONFIDENCE_THRESHOLD', '0.6'))
    
    # Data settings
    DEFAULT_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    DATA_INTERVAL = '1d'  # Daily data
    
    # Risk management
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '0.02'))  # 2% max daily loss
    MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '5'))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'trading_bot.log')
    
    # Simulation mode (set to False for live trading - NOT RECOMMENDED without proper testing)
    SIMULATION_MODE = os.getenv('SIMULATION_MODE', 'True').lower() == 'true'