# AI Trading Bot

An intelligent automated trading system that uses machine learning to analyze market data and make trading decisions. This bot combines technical analysis, risk management, and AI-powered predictions to automate stock trading in a simulated environment.

## 🚀 Features

- **AI-Powered Predictions**: Uses machine learning models (Random Forest, Gradient Boosting, Logistic Regression) to predict stock price movements
- **Technical Analysis**: Incorporates 15+ technical indicators including RSI, MACD, Bollinger Bands, and moving averages
- **Risk Management**: Advanced risk assessment with position sizing, stop losses, and portfolio risk monitoring
- **Portfolio Management**: Complete portfolio tracking with P&L calculation, performance metrics, and position management
- **Real-time Data**: Fetches live market data using Yahoo Finance API
- **Comprehensive Logging**: Detailed logging and reporting with daily performance reports
- **Simulation Mode**: Safe testing environment with no real money at risk

## 📋 Requirements

- Python 3.8+
- Required packages (see requirements.txt)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Armutimbire223373Q/skills-introduction-to-github.git
   cd skills-introduction-to-github
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the bot**:
   ```bash
   cp .env.example .env
   # Edit .env file with your preferred settings
   ```

## 🎯 Quick Start

### Option 1: Interactive Mode
```bash
python trading_bot.py
```

Then choose from the menu:
1. Train model and run single cycle
2. Start continuous trading (simulation)
3. Generate daily report
4. Show status
5. Exit

### Option 2: Direct Usage
```python
from trading_bot import TradingBot

# Create and configure the bot
bot = TradingBot()

# Train the AI model
bot.train_model()

# Run a single trading cycle
results = bot.run_trading_cycle()

# Check portfolio status
bot.print_status()

# Start continuous trading
bot.start_bot("simulation")
```

## 🧠 AI Models

The bot supports three machine learning models:

1. **Random Forest** (default): Robust ensemble method good for non-linear patterns
2. **Gradient Boosting**: Advanced boosting technique for complex market relationships  
3. **Logistic Regression**: Linear model for interpretable predictions

To change the model:
```python
from ai_predictor import AIPredictor

# Use Gradient Boosting
predictor = AIPredictor(model_type='gradient_boosting')
```

## 📊 Technical Indicators

The bot analyzes multiple technical indicators:

- **Trend Indicators**: SMA (10, 20, 50), EMA (12, 26)
- **Momentum**: RSI, MACD, Price Rate of Change
- **Volatility**: Bollinger Bands, Standard Deviation
- **Volume**: Volume Ratio, Volume Moving Average
- **Custom**: Price vs Moving Average ratios, Momentum indicators

## ⚡ Trading Strategy

The bot uses a hybrid approach:

1. **AI Prediction**: ML model predicts price direction (up/down)
2. **Technical Confirmation**: Technical analysis confirms/contradicts AI signal
3. **Risk Assessment**: Position sizing based on volatility and confidence
4. **Execution**: Places trades with automatic stop loss and take profit

### Signal Generation Logic:
- **BUY**: AI predicts price increase + technical indicators not bearish
- **SELL**: AI predicts price decrease + technical indicators not bullish  
- **HOLD**: Conflicting signals or low confidence

## 🛡️ Risk Management

Comprehensive risk controls:

- **Position Sizing**: Dynamic sizing based on volatility and confidence
- **Stop Losses**: Automatic 5% stop loss (configurable)
- **Take Profits**: Automatic 15% take profit (configurable)
- **Portfolio Limits**: Maximum 10% per position, 5 open positions max
- **Daily Loss Limit**: Stops trading at 2% daily loss
- **Drawdown Protection**: Monitors maximum drawdown

### Risk Metrics:
- Concentration Risk
- Drawdown Risk  
- Volatility Risk
- Correlation Risk
- Daily Loss Risk

## 📈 Performance Tracking

The bot tracks comprehensive metrics:

- **Returns**: Total return, daily returns, Sharpe ratio
- **Trades**: Win rate, profitable trades, trade history
- **Risk**: Maximum drawdown, volatility, risk score
- **Portfolio**: Cash balance, positions value, unrealized P&L

### Reports Generated:
- Real-time status updates
- Daily performance reports  
- Trade execution logs
- Risk assessment summaries

## 🔧 Configuration

Key configuration options in `.env`:

```bash
# Trading
INITIAL_BALANCE=10000          # Starting capital
MAX_POSITION_SIZE=0.1          # 10% max per position
STOP_LOSS_PERCENTAGE=0.05      # 5% stop loss
TAKE_PROFIT_PERCENTAGE=0.15    # 15% take profit

# AI Model
PREDICTION_CONFIDENCE_THRESHOLD=0.6  # Minimum confidence for trades
LOOKBACK_PERIOD=60                   # Days of historical data

# Risk Management  
MAX_DAILY_LOSS=0.02            # 2% max daily loss
MAX_OPEN_POSITIONS=5           # Maximum open positions

# Safety
SIMULATION_MODE=True           # Always keep True for safety
```

## 📁 Project Structure

```
├── trading_bot.py        # Main application
├── config.py            # Configuration management
├── data_fetcher.py      # Market data fetching
├── ai_predictor.py      # Machine learning models
├── trading_strategy.py  # Trading signal generation
├── portfolio.py         # Portfolio management
├── risk_manager.py      # Risk assessment
├── requirements.txt     # Python dependencies
├── .env.example        # Configuration template
└── logs/               # Log files and reports
```

## 🚨 Important Disclaimers

⚠️ **SIMULATION ONLY**: This bot is designed for educational and simulation purposes only.

⚠️ **NO FINANCIAL ADVICE**: This is not financial advice. Do not use for real trading without extensive testing.

⚠️ **RISK WARNING**: Trading involves substantial risk of loss. Past performance does not guarantee future results.

⚠️ **USE AT YOUR OWN RISK**: The authors are not responsible for any financial losses.

## 🧪 Testing & Development

### Running Tests:
```bash
# Test individual components
python -c "from data_fetcher import DataFetcher; df = DataFetcher(); print(df.fetch_stock_data('AAPL'))"

# Test AI model training  
python -c "from ai_predictor import AIPredictor; ai = AIPredictor(); print('AI Predictor loaded')"

# Test full trading cycle
python trading_bot.py
# Choose option 1: Train model and run single cycle
```

### Adding New Features:
1. **New Indicators**: Add to `data_fetcher.py` in `get_technical_indicators()`
2. **New Models**: Extend `ai_predictor.py` with additional ML algorithms
3. **New Strategies**: Modify `trading_strategy.py` signal generation logic
4. **New Risk Metrics**: Add to `risk_manager.py` assessment functions

## 📚 Algorithm Details

### Machine Learning Pipeline:
1. **Feature Engineering**: 20+ technical indicators and price features
2. **Target Creation**: Binary classification (price up/down)
3. **Training**: Cross-validated model training with hyperparameter tuning
4. **Prediction**: Real-time inference with confidence scoring
5. **Validation**: Backtesting and performance evaluation

### Risk Management Algorithm:
1. **Position Sizing**: Kelly criterion inspired, volatility adjusted
2. **Risk Scoring**: Multi-factor risk assessment (0-100 scale)
3. **Dynamic Limits**: Adaptive position limits based on market conditions
4. **Portfolio Monitoring**: Real-time risk metric calculation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is for educational purposes. Please ensure compliance with your local regulations before any trading activities.

## 🙏 Acknowledgments

- Yahoo Finance for market data API
- scikit-learn for machine learning algorithms  
- pandas/numpy for data processing
- The open source community for various libraries used

---

**Remember: This is a simulation tool for learning about algorithmic trading. Always practice responsible investing and never risk money you cannot afford to lose.**
