"""
AI/ML prediction model for stock price forecasting
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import logging
from typing import Tuple, Optional, Dict, Any
import joblib
import os

logger = logging.getLogger(__name__)

class AIPredictor:
    """AI-based stock price prediction model"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the AI predictor
        
        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 'logistic_regression')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        # Initialize the model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for machine learning
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            
        Returns:
            DataFrame with ML features
        """
        try:
            df = data.copy()
            
            # Ensure we have technical indicators
            if 'SMA_10' not in df.columns:
                logger.warning("Technical indicators not found. Basic features only.")
                return df[['Close', 'Volume', 'High', 'Low', 'Open']].dropna()
            
            # Select relevant features
            feature_columns = [
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'BB_Upper', 'BB_Lower', 'BB_Middle',
                'SMA_10', 'SMA_20', 'SMA_50',
                'EMA_12', 'EMA_26',
                'Volume_Ratio', 'Price_Change', 'Price_Change_5d'
            ]
            
            # Add price position relative to moving averages
            df['Price_vs_SMA10'] = (df['Close'] - df['SMA_10']) / df['SMA_10']
            df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
            df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
            
            # Add volatility features
            df['Volatility_10d'] = df['Close'].rolling(window=10).std()
            df['Volatility_20d'] = df['Close'].rolling(window=20).std()
            
            # Add momentum features
            df['Momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
            df['Momentum_10d'] = df['Close'] / df['Close'].shift(10) - 1
            
            feature_columns.extend([
                'Price_vs_SMA10', 'Price_vs_SMA20', 'Price_vs_SMA50',
                'Volatility_10d', 'Volatility_20d',
                'Momentum_5d', 'Momentum_10d'
            ])
            
            # Select available features
            available_features = [col for col in feature_columns if col in df.columns]
            self.feature_names = available_features
            
            return df[available_features].dropna()
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()
    
    def create_target(self, data: pd.DataFrame, prediction_days: int = 1) -> pd.Series:
        """
        Create target variable for prediction
        
        Args:
            data: DataFrame with price data
            prediction_days: Number of days ahead to predict
            
        Returns:
            Series with target labels (1 for price increase, 0 for decrease)
        """
        try:
            # Calculate future returns
            future_price = data['Close'].shift(-prediction_days)
            current_price = data['Close']
            future_return = (future_price - current_price) / current_price
            
            # Create binary target: 1 if price will increase, 0 if decrease
            target = (future_return > 0).astype(int)
            
            return target
            
        except Exception as e:
            logger.error(f"Error creating target: {str(e)}")
            return pd.Series()
    
    def train(self, data: pd.DataFrame, prediction_days: int = 1) -> Dict[str, Any]:
        """
        Train the AI model
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            prediction_days: Number of days ahead to predict
            
        Returns:
            Dictionary with training metrics
        """
        try:
            logger.info(f"Training {self.model_type} model...")
            
            # Prepare features and target
            features_df = self.prepare_features(data)
            target = self.create_target(data, prediction_days)
            
            if features_df.empty or target.empty:
                raise ValueError("Unable to prepare features or target")
            
            # Align features and target (remove NaN values)
            min_length = min(len(features_df), len(target))
            features_df = features_df.iloc[:min_length]
            target = target.iloc[:min_length]
            
            # Remove rows with NaN values
            valid_indices = ~(features_df.isna().any(axis=1) | target.isna())
            features_df = features_df[valid_indices]
            target = target[valid_indices]
            
            if len(features_df) < 50:
                raise ValueError(f"Insufficient data for training: {len(features_df)} samples")
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, target, test_size=0.2, random_state=42, stratify=target
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, features_scaled, target, cv=5)
            
            # Predictions for detailed metrics
            y_pred = self.model.predict(X_test)
            
            metrics = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean_accuracy': cv_scores.mean(),
                'cv_std_accuracy': cv_scores.std(),
                'n_samples': len(features_df),
                'n_features': len(self.feature_names),
                'prediction_days': prediction_days
            }
            
            self.is_trained = True
            
            logger.info(f"Model training completed. Test accuracy: {test_score:.4f}")
            logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {}
    
    def predict(self, data: pd.DataFrame) -> Tuple[Optional[int], Optional[float]]:
        """
        Make a prediction for the latest data point
        
        Args:
            data: DataFrame with latest market data
            
        Returns:
            Tuple of (prediction, confidence) where prediction is 1 (buy) or 0 (sell)
        """
        try:
            if not self.is_trained:
                logger.error("Model not trained. Call train() first.")
                return None, None
            
            # Prepare features for the latest data point
            features_df = self.prepare_features(data)
            
            if features_df.empty:
                logger.error("Unable to prepare features for prediction")
                return None, None
            
            # Use the latest data point
            latest_features = features_df.iloc[-1:][self.feature_names]
            
            if latest_features.isna().any().any():
                logger.warning("NaN values in features for prediction")
                return None, None
            
            # Scale features
            features_scaled = self.scaler.transform(latest_features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Get prediction probability (confidence)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = max(probabilities)
            else:
                confidence = 0.6  # Default confidence for models without probability
            
            logger.info(f"Prediction: {'BUY' if prediction == 1 else 'SELL'} (confidence: {confidence:.3f})")
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None, None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores
        
        Returns:
            Dictionary with feature names and their importance scores
        """
        try:
            if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
                return None
            
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return None
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.is_trained:
                logger.error("No trained model to save")
                return False
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False