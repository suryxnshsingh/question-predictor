"""
Enhanced Prediction Model using sub-topics and historical patterns
"""

import pandas as pd
import numpy as np
from typing import Dict

class EnhancedPredictor:
    """Predictor using enhanced topics and historical patterns"""

    def __init__(self, df: pd.DataFrame, pattern_features: pd.DataFrame):
        """
        Args:
            df: DataFrame with enhanced topics
            pattern_features: Historical pattern features
        """
        self.df = df
        self.pattern_features = pattern_features
        self.predictions = None

    def predict(self):
        """Predict using enhanced features"""
        scores = {}

        for _, row in self.pattern_features.iterrows():
            topic = row['enhanced_topic']

            # Feature 1: Frequency (how often it appears)
            freq_score = row['frequency']

            # Feature 2: Overdue bonus (if past average gap)
            if row['is_overdue']:
                overdue_bonus = min(row['years_since_last'] / (row['avg_gap'] + 0.1), 2.0)
            else:
                # Slight penalty if just appeared (unless very high frequency)
                if row['frequency'] >= 0.75:
                    overdue_bonus = 1.0  # Core topics: no penalty
                else:
                    overdue_bonus = max(0.5, row['years_since_last'] / 4.0)

            # Feature 3: Regularity bonus (predictable patterns score higher)
            regularity_bonus = 1.0 + (row['regularity'] * 0.2)

            # Feature 4: Trend (appearing more recently?)
            trend_bonus = np.sqrt(row['trend'])

            # Feature 5: Cycle position (closer to expected appearance?)
            if row['avg_gap'] > 0 and row['cycle_position'] < 1.0:
                cycle_bonus = 1.2  # Within 1 year of expected cycle
            else:
                cycle_bonus = 1.0

            # Combined score
            score = (
                freq_score *
                overdue_bonus *
                regularity_bonus *
                trend_bonus *
                cycle_bonus
            )

            scores[topic] = score

        # Normalize
        max_score = max(scores.values()) if scores else 1.0
        self.predictions = {k: v/max_score for k, v in scores.items()}

        return pd.DataFrame([
            {'enhanced_topic': topic, 'probability': score}
            for topic, score in self.predictions.items()
        ]).sort_values('probability', ascending=False)

    def validate(self, actual_topics: set, top_n: int = 25):
        """Validate against actual enhanced topics"""
        if self.predictions is None:
            self.predict()

        pred_df = pd.DataFrame([
            {'enhanced_topic': k, 'probability': v}
            for k, v in self.predictions.items()
        ]).sort_values('probability', ascending=False)

        predicted = set(pred_df.head(top_n)['enhanced_topic'])
        hits = len(predicted & actual_topics)
        precision = hits / len(predicted)
        recall = hits / len(actual_topics)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'hits': hits
        }
