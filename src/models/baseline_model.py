"""
Baseline model using frequency and recency heuristics
Simple but effective predictor based on historical patterns
"""

import pandas as pd
import numpy as np
from typing import Dict

class BaselinePredictor:
    def __init__(self, features_df: pd.DataFrame, recency_weight: float = 0.3):
        """
        Initialize baseline predictor with engineered features

        Args:
            features_df: DataFrame with columns [chapter, frequency, recency, ...]
            recency_weight: Weight for recency in heat score calculation
        """
        self.features = features_df.copy()
        self.recency_weight = recency_weight
        self.predictions = None

    def calculate_heat_score(self) -> pd.Series:
        """
        Heat score formula:
        score = frequency × (1 + recency_weight × recency) × trend_factor

        Logic:
        - Higher frequency = more likely to appear
        - Higher recency (not appeared recently) = more likely to come back
        - Positive trend = appearing more frequently in recent years
        """
        # Normalize recency (0-1 scale)
        max_recency = self.features['recency'].max()
        if max_recency > 0:
            norm_recency = self.features['recency'] / max_recency
        else:
            norm_recency = 0

        # Calculate base heat score
        self.features['heat_score'] = (
            self.features['frequency'] *
            (1 + self.recency_weight * norm_recency) *
            np.sqrt(self.features['trend'])  # Trend boost (sqrt to moderate effect)
        )

        # Normalize to 0-1 range
        max_score = self.features['heat_score'].max()
        if max_score > 0:
            self.features['heat_score'] = self.features['heat_score'] / max_score

        return self.features.set_index('chapter')['heat_score']

    def predict(self, top_n: int = 25) -> pd.DataFrame:
        """
        Get top N chapters by heat score

        Args:
            top_n: Number of top predictions to return

        Returns:
            DataFrame with chapters and their baseline scores
        """
        scores = self.calculate_heat_score()

        # Get top N
        self.predictions = scores.nlargest(min(top_n, len(scores))).to_frame()
        self.predictions.columns = ['baseline_score']

        # Round for readability
        self.predictions['baseline_score'] = self.predictions['baseline_score'].round(4)

        return self.predictions

    def get_explanation(self, chapter: str) -> Dict:
        """
        Get explanation for why a chapter was predicted

        Args:
            chapter: Chapter name

        Returns:
            Dictionary with breakdown of score components
        """
        if chapter not in self.features['chapter'].values:
            return {'error': 'Chapter not found'}

        chapter_data = self.features[self.features['chapter'] == chapter].iloc[0]

        return {
            'chapter': chapter,
            'frequency': f"{chapter_data['frequency']:.2f}",
            'recency': f"{chapter_data['recency']} years",
            'trend': f"{chapter_data['trend']:.2f}",
            'total_appearances': int(chapter_data['total_appearances']),
            'last_year': int(chapter_data['last_year']),
            'reasoning': self._generate_reasoning(chapter_data)
        }

    def _generate_reasoning(self, chapter_data: pd.Series) -> str:
        """Generate human-readable reasoning for prediction"""
        reasons = []

        if chapter_data['frequency'] >= 0.8:
            reasons.append("appears very frequently in exams")
        elif chapter_data['frequency'] >= 0.5:
            reasons.append("appears regularly in exams")

        if chapter_data['recency'] >= 2:
            reasons.append(f"hasn't appeared in {int(chapter_data['recency'])} years (due for return)")
        elif chapter_data['recency'] == 0:
            reasons.append("appeared in most recent exam")

        if chapter_data['trend'] > 1.2:
            reasons.append("showing increasing trend in recent years")

        if not reasons:
            reasons.append(f"appeared {int(chapter_data['total_appearances'])} times historically")

        return "; ".join(reasons)
