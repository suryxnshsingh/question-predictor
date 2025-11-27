"""
ARIMA time series model for topic appearance prediction
Uses Auto-ARIMA to find optimal parameters for each chapter
"""

import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

class ARIMAPredictor:
    def __init__(self, topic_matrix: pd.DataFrame):
        """
        Initialize ARIMA predictor with topic appearance matrix

        Args:
            topic_matrix: Binary matrix (years x chapters) showing appearance
        """
        self.topic_matrix = topic_matrix
        self.models = {}
        self.predictions = {}

    def check_stationarity(self, series: np.array) -> bool:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test

        Args:
            series: Time series array

        Returns:
            True if stationary (p-value < 0.05)
        """
        if len(series) < 3:
            return False

        try:
            result = adfuller(series)
            return result[1] < 0.05  # p-value < 0.05 means stationary
        except:
            return False

    def fit_topic_model(self, topic: str) -> float:
        """
        Fit ARIMA model for a single chapter/topic

        Args:
            topic: Chapter name

        Returns:
            Probability of appearance in next year (0-1)
        """
        # Get binary time series for topic
        if topic not in self.topic_matrix.columns:
            return 0.5  # Default probability

        series = self.topic_matrix[topic].values

        # Need at least 3 data points for ARIMA
        if len(series) < 3:
            return series.mean()

        try:
            # Auto-find best ARIMA parameters
            # For binary data, keep it simple with low max orders
            model = auto_arima(
                series,
                start_p=0, start_q=0,
                max_p=2, max_q=2,  # Keep simple for small datasets
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_order=5,
                trace=False
            )

            # Predict next value
            forecast = model.predict(n_periods=1)[0]

            # Bound between 0 and 1 (it's a probability)
            probability = min(max(forecast, 0), 1)

            self.models[topic] = model
            return probability

        except Exception as e:
            # Fallback to simple mean if ARIMA fails
            return series.mean()

    def predict_all_topics(self) -> pd.Series:
        """
        Generate ARIMA predictions for all chapters

        Returns:
            Series with chapter names as index and probabilities as values
        """
        predictions = {}

        for topic in self.topic_matrix.columns:
            prob = self.fit_topic_model(topic)
            predictions[topic] = prob

        return pd.Series(predictions)

    def get_top_predictions(self, top_n: int = 25) -> pd.DataFrame:
        """
        Return top N chapters by ARIMA probability

        Args:
            top_n: Number of top predictions to return

        Returns:
            DataFrame with chapters and ARIMA scores
        """
        all_predictions = self.predict_all_topics()
        top = all_predictions.nlargest(min(top_n, len(all_predictions))).to_frame()
        top.columns = ['arima_score']
        top['arima_score'] = top['arima_score'].round(4)
        return top

    def get_model_summary(self, topic: str) -> dict:
        """
        Get ARIMA model summary for a specific chapter

        Args:
            topic: Chapter name

        Returns:
            Dictionary with model info
        """
        if topic not in self.models:
            return {'error': 'Model not fitted for this chapter'}

        model = self.models[topic]
        series = self.topic_matrix[topic].values

        return {
            'chapter': topic,
            'order': model.order,
            'aic': round(model.aic(), 2) if hasattr(model, 'aic') else None,
            'historical_pattern': series.tolist(),
            'mean_appearance': round(series.mean(), 2),
            'is_stationary': self.check_stationarity(series)
        }
