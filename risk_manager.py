"""
Risk management module for AI Trading Bot
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class RiskMetric:
    """Risk metric data structure"""
    name: str
    value: float
    threshold: float
    risk_level: RiskLevel
    description: str

class RiskManager:
    """Risk management and position sizing"""
    
    def __init__(self, config):
        """
        Initialize risk manager
        
        Args:
            config: Configuration object with risk parameters
        """
        self.config = config
        self.risk_metrics_history = []
        
    def calculate_position_size(self, portfolio_value: float, price: float, volatility: float = 0.02, 
                              confidence: float = 0.5) -> float:
        """
        Calculate optimal position size based on risk management rules
        
        Args:
            portfolio_value: Current portfolio value
            price: Entry price of the asset
            volatility: Historical volatility of the asset
            confidence: AI prediction confidence
            
        Returns:
            Number of shares to buy
        """
        try:
            # Base position size as percentage of portfolio
            base_position_value = portfolio_value * self.config.MAX_POSITION_SIZE
            
            # Adjust for volatility (lower volatility = larger position)
            volatility_adjustment = max(0.5, min(1.5, 1 / (1 + volatility * 10)))
            
            # Adjust for confidence (higher confidence = larger position)
            confidence_adjustment = max(0.5, min(1.5, confidence * 1.5))
            
            # Calculate final position value
            adjusted_position_value = base_position_value * volatility_adjustment * confidence_adjustment
            
            # Convert to number of shares
            shares = int(adjusted_position_value / price)
            
            # Ensure minimum position size
            if shares < 1 and adjusted_position_value >= price:
                shares = 1
            
            logger.info(f"Position size calculated: {shares} shares (value: ${shares * price:.2f})")
            logger.debug(f"Adjustments - Volatility: {volatility_adjustment:.2f}, Confidence: {confidence_adjustment:.2f}")
            
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    def assess_portfolio_risk(self, portfolio, market_data: Dict[str, pd.DataFrame]) -> List[RiskMetric]:
        """
        Assess overall portfolio risk
        
        Args:
            portfolio: Portfolio object
            market_data: Dictionary of market data for each symbol
            
        Returns:
            List of risk metrics
        """
        try:
            risk_metrics = []
            
            # Portfolio concentration risk
            concentration_risk = self._calculate_concentration_risk(portfolio)
            risk_metrics.append(concentration_risk)
            
            # Drawdown risk
            drawdown_risk = self._calculate_drawdown_risk(portfolio)
            risk_metrics.append(drawdown_risk)
            
            # Volatility risk
            volatility_risk = self._calculate_volatility_risk(portfolio, market_data)
            risk_metrics.append(volatility_risk)
            
            # Correlation risk
            correlation_risk = self._calculate_correlation_risk(portfolio, market_data)
            risk_metrics.append(correlation_risk)
            
            # Daily loss risk
            daily_loss_risk = self._calculate_daily_loss_risk(portfolio)
            risk_metrics.append(daily_loss_risk)
            
            # Store metrics history
            self.risk_metrics_history.append({
                'timestamp': datetime.now(),
                'metrics': risk_metrics
            })
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {str(e)}")
            return []
    
    def _calculate_concentration_risk(self, portfolio) -> RiskMetric:
        """Calculate portfolio concentration risk"""
        try:
            if not portfolio.positions:
                return RiskMetric(
                    name="Concentration Risk",
                    value=0.0,
                    threshold=0.3,
                    risk_level=RiskLevel.LOW,
                    description="No positions - no concentration risk"
                )
            
            total_value = portfolio.get_positions_value()
            if total_value == 0:
                concentration = 0.0
            else:
                # Calculate largest position as percentage of total
                largest_position = max([pos.market_value for pos in portfolio.positions.values()])
                concentration = largest_position / total_value
            
            # Determine risk level
            if concentration <= 0.2:
                risk_level = RiskLevel.LOW
            elif concentration <= 0.3:
                risk_level = RiskLevel.MEDIUM
            elif concentration <= 0.4:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            return RiskMetric(
                name="Concentration Risk",
                value=concentration,
                threshold=0.3,
                risk_level=risk_level,
                description=f"Largest position represents {concentration:.1%} of portfolio"
            )
            
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {str(e)}")
            return RiskMetric("Concentration Risk", 0.0, 0.3, RiskLevel.LOW, "Error calculating")
    
    def _calculate_drawdown_risk(self, portfolio) -> RiskMetric:
        """Calculate current drawdown risk"""
        try:
            drawdown = portfolio.max_drawdown
            
            # Determine risk level
            if drawdown <= 0.05:
                risk_level = RiskLevel.LOW
            elif drawdown <= 0.1:
                risk_level = RiskLevel.MEDIUM
            elif drawdown <= 0.2:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            return RiskMetric(
                name="Drawdown Risk",
                value=drawdown,
                threshold=0.1,
                risk_level=risk_level,
                description=f"Maximum drawdown: {drawdown:.1%}"
            )
            
        except Exception as e:
            logger.error(f"Error calculating drawdown risk: {str(e)}")
            return RiskMetric("Drawdown Risk", 0.0, 0.1, RiskLevel.LOW, "Error calculating")
    
    def _calculate_volatility_risk(self, portfolio, market_data: Dict[str, pd.DataFrame]) -> RiskMetric:
        """Calculate portfolio volatility risk"""
        try:
            if not portfolio.positions or not market_data:
                return RiskMetric(
                    name="Volatility Risk",
                    value=0.0,
                    threshold=0.03,
                    risk_level=RiskLevel.LOW,
                    description="No data for volatility calculation"
                )
            
            # Calculate weighted average volatility
            total_value = portfolio.get_positions_value()
            weighted_volatility = 0.0
            
            for symbol, position in portfolio.positions.items():
                if symbol in market_data and not market_data[symbol].empty:
                    # Calculate 20-day volatility
                    returns = market_data[symbol]['Close'].pct_change().tail(20)
                    volatility = returns.std() * np.sqrt(252)  # Annualized
                    weight = position.market_value / total_value if total_value > 0 else 0
                    weighted_volatility += volatility * weight
            
            # Determine risk level
            if weighted_volatility <= 0.2:
                risk_level = RiskLevel.LOW
            elif weighted_volatility <= 0.3:
                risk_level = RiskLevel.MEDIUM
            elif weighted_volatility <= 0.5:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            return RiskMetric(
                name="Volatility Risk",
                value=weighted_volatility,
                threshold=0.3,
                risk_level=risk_level,
                description=f"Portfolio volatility: {weighted_volatility:.1%}"
            )
            
        except Exception as e:
            logger.error(f"Error calculating volatility risk: {str(e)}")
            return RiskMetric("Volatility Risk", 0.0, 0.3, RiskLevel.LOW, "Error calculating")
    
    def _calculate_correlation_risk(self, portfolio, market_data: Dict[str, pd.DataFrame]) -> RiskMetric:
        """Calculate portfolio correlation risk"""
        try:
            symbols = list(portfolio.positions.keys())
            
            if len(symbols) < 2:
                return RiskMetric(
                    name="Correlation Risk",
                    value=0.0,
                    threshold=0.7,
                    risk_level=RiskLevel.LOW,
                    description="Less than 2 positions - no correlation risk"
                )
            
            # Calculate correlation matrix
            returns_data = {}
            for symbol in symbols:
                if symbol in market_data and not market_data[symbol].empty:
                    returns = market_data[symbol]['Close'].pct_change().tail(60).dropna()
                    if len(returns) > 30:  # Minimum data requirement
                        returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                return RiskMetric(
                    name="Correlation Risk",
                    value=0.0,
                    threshold=0.7,
                    risk_level=RiskLevel.LOW,
                    description="Insufficient data for correlation calculation"
                )
            
            # Create DataFrame and calculate correlation
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            # Calculate average correlation (excluding diagonal)
            correlations = []
            for i in range(len(correlation_matrix)):
                for j in range(i+1, len(correlation_matrix)):
                    correlations.append(abs(correlation_matrix.iloc[i, j]))
            
            avg_correlation = np.mean(correlations) if correlations else 0.0
            
            # Determine risk level
            if avg_correlation <= 0.5:
                risk_level = RiskLevel.LOW
            elif avg_correlation <= 0.7:
                risk_level = RiskLevel.MEDIUM
            elif avg_correlation <= 0.85:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            return RiskMetric(
                name="Correlation Risk",
                value=avg_correlation,
                threshold=0.7,
                risk_level=risk_level,
                description=f"Average correlation: {avg_correlation:.1%}"
            )
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {str(e)}")
            return RiskMetric("Correlation Risk", 0.0, 0.7, RiskLevel.LOW, "Error calculating")
    
    def _calculate_daily_loss_risk(self, portfolio) -> RiskMetric:
        """Calculate daily loss risk"""
        try:
            if len(portfolio.daily_returns) == 0:
                return RiskMetric(
                    name="Daily Loss Risk",
                    value=0.0,
                    threshold=self.config.MAX_DAILY_LOSS,
                    risk_level=RiskLevel.LOW,
                    description="No daily returns data"
                )
            
            # Get today's return
            today_return = portfolio.daily_returns[-1] if portfolio.daily_returns else 0.0
            daily_loss = abs(min(0, today_return))  # Only consider losses
            
            # Determine risk level
            threshold = self.config.MAX_DAILY_LOSS
            if daily_loss <= threshold * 0.5:
                risk_level = RiskLevel.LOW
            elif daily_loss <= threshold:
                risk_level = RiskLevel.MEDIUM
            elif daily_loss <= threshold * 1.5:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            return RiskMetric(
                name="Daily Loss Risk",
                value=daily_loss,
                threshold=threshold,
                risk_level=risk_level,
                description=f"Today's loss: {daily_loss:.1%}"
            )
            
        except Exception as e:
            logger.error(f"Error calculating daily loss risk: {str(e)}")
            return RiskMetric("Daily Loss Risk", 0.0, self.config.MAX_DAILY_LOSS, RiskLevel.LOW, "Error calculating")
    
    def should_stop_trading(self, portfolio, risk_metrics: List[RiskMetric]) -> Tuple[bool, List[str]]:
        """
        Determine if trading should be stopped due to high risk
        
        Args:
            portfolio: Portfolio object
            risk_metrics: List of current risk metrics
            
        Returns:
            Tuple of (should_stop, reasons)
        """
        try:
            should_stop = False
            reasons = []
            
            # Check for critical risk levels
            critical_metrics = [metric for metric in risk_metrics if metric.risk_level == RiskLevel.CRITICAL]
            if critical_metrics:
                should_stop = True
                reasons.extend([f"Critical {metric.name}" for metric in critical_metrics])
            
            # Check daily loss limit
            if len(portfolio.daily_returns) > 0:
                today_return = portfolio.daily_returns[-1]
                if today_return < -self.config.MAX_DAILY_LOSS:
                    should_stop = True
                    reasons.append(f"Daily loss limit exceeded: {today_return:.1%}")
            
            # Check maximum drawdown
            if portfolio.max_drawdown > 0.25:  # 25% max drawdown hard limit
                should_stop = True
                reasons.append(f"Maximum drawdown exceeded: {portfolio.max_drawdown:.1%}")
            
            # Check number of positions
            if len(portfolio.positions) >= self.config.MAX_OPEN_POSITIONS:
                should_stop = True
                reasons.append(f"Maximum positions reached: {len(portfolio.positions)}")
            
            if should_stop:
                logger.warning(f"Trading stopped due to: {', '.join(reasons)}")
            
            return should_stop, reasons
            
        except Exception as e:
            logger.error(f"Error checking if trading should stop: {str(e)}")
            return False, []
    
    def get_risk_score(self, risk_metrics: List[RiskMetric]) -> float:
        """
        Calculate overall risk score (0-100, where 100 is highest risk)
        
        Args:
            risk_metrics: List of risk metrics
            
        Returns:
            Overall risk score
        """
        try:
            if not risk_metrics:
                return 0.0
            
            risk_scores = []
            for metric in risk_metrics:
                if metric.risk_level == RiskLevel.LOW:
                    score = 10
                elif metric.risk_level == RiskLevel.MEDIUM:
                    score = 30
                elif metric.risk_level == RiskLevel.HIGH:
                    score = 60
                elif metric.risk_level == RiskLevel.CRITICAL:
                    score = 90
                else:
                    score = 50
                
                risk_scores.append(score)
            
            # Take weighted average (higher risk metrics have more weight)
            weights = [score / 100 for score in risk_scores]
            total_weight = sum(weights)
            
            if total_weight == 0:
                return 0.0
            
            weighted_score = sum(score * weight for score, weight in zip(risk_scores, weights)) / total_weight
            
            return min(100, max(0, weighted_score))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return 50.0  # Return moderate risk on error
    
    def get_risk_summary(self, risk_metrics: List[RiskMetric]) -> Dict:
        """
        Get summary of risk assessment
        
        Args:
            risk_metrics: List of risk metrics
            
        Returns:
            Dictionary with risk summary
        """
        try:
            if not risk_metrics:
                return {'overall_risk': 'Unknown', 'risk_score': 0, 'recommendations': []}
            
            risk_score = self.get_risk_score(risk_metrics)
            
            # Determine overall risk level
            if risk_score <= 25:
                overall_risk = "LOW"
                color = "green"
            elif risk_score <= 50:
                overall_risk = "MEDIUM"
                color = "yellow"
            elif risk_score <= 75:
                overall_risk = "HIGH"
                color = "orange"
            else:
                overall_risk = "CRITICAL"
                color = "red"
            
            # Generate recommendations
            recommendations = []
            for metric in risk_metrics:
                if metric.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    if metric.name == "Concentration Risk":
                        recommendations.append("Consider diversifying portfolio - reduce large positions")
                    elif metric.name == "Drawdown Risk":
                        recommendations.append("Consider reducing position sizes or stopping trading temporarily")
                    elif metric.name == "Volatility Risk":
                        recommendations.append("Consider reducing exposure to high-volatility assets")
                    elif metric.name == "Correlation Risk":
                        recommendations.append("Consider diversifying into uncorrelated assets")
                    elif metric.name == "Daily Loss Risk":
                        recommendations.append("Consider stopping trading for today to limit losses")
            
            if not recommendations:
                recommendations.append("Risk levels are acceptable - continue with current strategy")
            
            summary = {
                'overall_risk': overall_risk,
                'risk_score': risk_score,
                'color': color,
                'metrics': [
                    {
                        'name': metric.name,
                        'value': metric.value,
                        'threshold': metric.threshold,
                        'level': metric.risk_level.value,
                        'description': metric.description
                    }
                    for metric in risk_metrics
                ],
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating risk summary: {str(e)}")
            return {'overall_risk': 'Unknown', 'risk_score': 0, 'recommendations': []}