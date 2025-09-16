"""
Trading strategy engine for AI Trading Bot
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal_type: SignalType
    confidence: float
    price: float
    timestamp: datetime
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class TradingStrategy:
    """Main trading strategy engine"""
    
    def __init__(self, config):
        """
        Initialize trading strategy
        
        Args:
            config: Configuration object with trading parameters
        """
        self.config = config
        self.signals_history = []
        
    def generate_signal(self, symbol: str, data: pd.DataFrame, ai_prediction: Tuple[Optional[int], Optional[float]]) -> TradingSignal:
        """
        Generate trading signal based on AI prediction and technical analysis
        
        Args:
            symbol: Stock symbol
            data: DataFrame with market data and technical indicators
            ai_prediction: Tuple of (prediction, confidence) from AI model
            
        Returns:
            TradingSignal object
        """
        try:
            if data.empty:
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    price=0.0,
                    timestamp=datetime.now(),
                    reason="No data available"
                )
            
            latest_data = data.iloc[-1]
            current_price = latest_data['Close']
            
            # Get AI prediction
            ai_signal, ai_confidence = ai_prediction
            
            if ai_signal is None or ai_confidence is None:
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    price=current_price,
                    timestamp=datetime.now(),
                    reason="AI prediction unavailable"
                )
            
            # Check if AI confidence meets threshold
            if ai_confidence < self.config.PREDICTION_CONFIDENCE_THRESHOLD:
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    confidence=ai_confidence,
                    price=current_price,
                    timestamp=datetime.now(),
                    reason=f"AI confidence too low: {ai_confidence:.3f}"
                )
            
            # Technical analysis confirmation
            technical_score = self._calculate_technical_score(latest_data)
            
            # Combine AI prediction with technical analysis
            final_confidence = (ai_confidence * 0.7) + (technical_score * 0.3)
            
            # Determine signal type
            if ai_signal == 1 and technical_score > 0.3:  # AI says buy and technical is not strongly bearish
                signal_type = SignalType.BUY
                stop_loss = current_price * (1 - self.config.STOP_LOSS_PERCENTAGE)
                take_profit = current_price * (1 + self.config.TAKE_PROFIT_PERCENTAGE)
                reason = f"AI: BUY ({ai_confidence:.3f}), Technical: {technical_score:.3f}"
            elif ai_signal == 0 and technical_score < 0.7:  # AI says sell and technical is not strongly bullish
                signal_type = SignalType.SELL
                stop_loss = current_price * (1 + self.config.STOP_LOSS_PERCENTAGE)
                take_profit = current_price * (1 - self.config.TAKE_PROFIT_PERCENTAGE)
                reason = f"AI: SELL ({ai_confidence:.3f}), Technical: {technical_score:.3f}"
            else:
                signal_type = SignalType.HOLD
                stop_loss = None
                take_profit = None
                reason = f"Conflicting signals - AI: {ai_signal}, Technical: {technical_score:.3f}"
            
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=final_confidence,
                price=current_price,
                timestamp=datetime.now(),
                reason=reason,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            self.signals_history.append(signal)
            
            logger.info(f"Generated signal for {symbol}: {signal_type.value} (confidence: {final_confidence:.3f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=0.0,
                timestamp=datetime.now(),
                reason=f"Error: {str(e)}"
            )
    
    def _calculate_technical_score(self, data: pd.Series) -> float:
        """
        Calculate technical analysis score (0-1, where 1 is most bullish)
        
        Args:
            data: Latest market data point
            
        Returns:
            Technical score between 0 and 1
        """
        try:
            score = 0.5  # Neutral starting point
            
            # RSI analysis
            if 'RSI' in data.index and not pd.isna(data['RSI']):
                rsi = data['RSI']
                if rsi < 30:  # Oversold - bullish
                    score += 0.2
                elif rsi > 70:  # Overbought - bearish
                    score -= 0.2
                elif 40 <= rsi <= 60:  # Neutral
                    pass
                else:
                    # Gradual scoring
                    score += (50 - rsi) / 100
            
            # MACD analysis
            if all(col in data.index for col in ['MACD', 'MACD_Signal']) and not any(pd.isna(data[col]) for col in ['MACD', 'MACD_Signal']):
                if data['MACD'] > data['MACD_Signal']:  # Bullish crossover
                    score += 0.15
                else:  # Bearish crossover
                    score -= 0.15
            
            # Moving average analysis
            if all(col in data.index for col in ['Close', 'SMA_10', 'SMA_20']) and not any(pd.isna(data[col]) for col in ['Close', 'SMA_10', 'SMA_20']):
                price = data['Close']
                sma_10 = data['SMA_10']
                sma_20 = data['SMA_20']
                
                if price > sma_10 > sma_20:  # Strong uptrend
                    score += 0.2
                elif price < sma_10 < sma_20:  # Strong downtrend
                    score -= 0.2
                elif price > sma_10:  # Above short-term MA
                    score += 0.1
                elif price < sma_10:  # Below short-term MA
                    score -= 0.1
            
            # Bollinger Bands analysis
            if all(col in data.index for col in ['Close', 'BB_Upper', 'BB_Lower']) and not any(pd.isna(data[col]) for col in ['Close', 'BB_Upper', 'BB_Lower']):
                price = data['Close']
                bb_upper = data['BB_Upper']
                bb_lower = data['BB_Lower']
                
                if price <= bb_lower:  # Near lower band - potential bounce
                    score += 0.1
                elif price >= bb_upper:  # Near upper band - potential reversal
                    score -= 0.1
            
            # Volume analysis
            if 'Volume_Ratio' in data.index and not pd.isna(data['Volume_Ratio']):
                volume_ratio = data['Volume_Ratio']
                if volume_ratio > 1.5:  # High volume - confirms trend
                    # This doesn't change score direction, just confidence
                    pass
            
            # Ensure score is between 0 and 1
            score = max(0, min(1, score))
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {str(e)}")
            return 0.5  # Return neutral score on error
    
    def backtest_strategy(self, symbols: List[str], data_dict: Dict[str, pd.DataFrame], ai_predictions: Dict[str, List[Tuple]], 
                         initial_balance: float = 10000) -> Dict:
        """
        Backtest the trading strategy
        
        Args:
            symbols: List of symbols to backtest
            data_dict: Dictionary of historical data for each symbol
            ai_predictions: Dictionary of AI predictions for each symbol
            initial_balance: Starting balance for backtest
            
        Returns:
            Dictionary with backtest results
        """
        try:
            logger.info("Starting strategy backtest...")
            
            portfolio_value = []
            balance = initial_balance
            positions = {}  # symbol: {'shares': int, 'entry_price': float, 'entry_date': datetime}
            trades = []
            
            # Get common date range
            min_date = max([data.index.min() for data in data_dict.values()])
            max_date = min([data.index.max() for data in data_dict.values()])
            
            current_date = min_date
            
            while current_date <= max_date:
                daily_portfolio_value = balance
                
                for symbol in symbols:
                    if symbol not in data_dict or current_date not in data_dict[symbol].index:
                        continue
                    
                    # Get data up to current date
                    historical_data = data_dict[symbol].loc[:current_date]
                    
                    if len(historical_data) < 50:  # Need enough data for indicators
                        continue
                    
                    # Simulate AI prediction (in real backtest, you'd use actual predictions)
                    # For this simulation, we'll use a simple momentum-based prediction
                    recent_return = historical_data['Close'].pct_change(5).iloc[-1]
                    ai_pred = (1 if recent_return > 0 else 0, abs(recent_return) * 10)  # Simple simulation
                    
                    # Generate signal
                    signal = self.generate_signal(symbol, historical_data, ai_pred)
                    current_price = historical_data['Close'].iloc[-1]
                    
                    # Execute trades based on signals
                    if signal.signal_type == SignalType.BUY and symbol not in positions:
                        # Calculate position size
                        position_value = balance * self.config.MAX_POSITION_SIZE
                        shares = int(position_value / current_price)
                        
                        if shares > 0 and balance >= shares * current_price:
                            positions[symbol] = {
                                'shares': shares,
                                'entry_price': current_price,
                                'entry_date': current_date
                            }
                            balance -= shares * current_price
                            trades.append({
                                'symbol': symbol,
                                'action': 'BUY',
                                'shares': shares,
                                'price': current_price,
                                'date': current_date,
                                'signal_confidence': signal.confidence
                            })
                    
                    elif signal.signal_type == SignalType.SELL and symbol in positions:
                        # Sell position
                        position = positions[symbol]
                        shares = position['shares']
                        balance += shares * current_price
                        
                        # Calculate profit/loss
                        profit_loss = shares * (current_price - position['entry_price'])
                        
                        trades.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'shares': shares,
                            'price': current_price,
                            'date': current_date,
                            'profit_loss': profit_loss,
                            'signal_confidence': signal.confidence
                        })
                        
                        del positions[symbol]
                    
                    # Add position value to portfolio
                    if symbol in positions:
                        daily_portfolio_value += positions[symbol]['shares'] * current_price
                
                portfolio_value.append({
                    'date': current_date,
                    'value': daily_portfolio_value
                })
                
                current_date += timedelta(days=1)
            
            # Calculate final results
            final_value = portfolio_value[-1]['value'] if portfolio_value else initial_balance
            total_return = (final_value - initial_balance) / initial_balance
            
            # Calculate additional metrics
            profitable_trades = len([t for t in trades if t.get('profit_loss', 0) > 0])
            total_trades = len([t for t in trades if 'profit_loss' in t])
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            results = {
                'initial_balance': initial_balance,
                'final_value': final_value,
                'total_return': total_return,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': win_rate,
                'trades': trades,
                'portfolio_value': portfolio_value
            }
            
            logger.info(f"Backtest completed. Total return: {total_return:.2%}, Win rate: {win_rate:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            return {}
    
    def get_signals_summary(self, days: int = 30) -> Dict:
        """
        Get summary of recent signals
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with signal statistics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_signals = [s for s in self.signals_history if s.timestamp >= cutoff_date]
            
            if not recent_signals:
                return {'total_signals': 0}
            
            buy_signals = len([s for s in recent_signals if s.signal_type == SignalType.BUY])
            sell_signals = len([s for s in recent_signals if s.signal_type == SignalType.SELL])
            hold_signals = len([s for s in recent_signals if s.signal_type == SignalType.HOLD])
            
            avg_confidence = sum([s.confidence for s in recent_signals]) / len(recent_signals)
            
            return {
                'total_signals': len(recent_signals),
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals,
                'average_confidence': avg_confidence,
                'date_range': f"{cutoff_date.date()} to {datetime.now().date()}"
            }
            
        except Exception as e:
            logger.error(f"Error getting signals summary: {str(e)}")
            return {}