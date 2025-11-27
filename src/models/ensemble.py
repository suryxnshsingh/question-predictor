"""
Ensemble model combining all predictors
Uses weighted voting to combine baseline, ARIMA, and association rules
"""

import pandas as pd
import numpy as np
from typing import Dict, List

class EnsemblePredictor:
    def __init__(self):
        """Initialize ensemble predictor"""
        self.predictors = {}
        self.weights = {}
        self.predictions = None

    def add_predictor(self, name: str, predictor_scores: pd.Series,
                     weight: float = 0.33):
        """
        Add a predictor with its scores and weight

        Args:
            name: Predictor name (e.g., 'baseline', 'arima')
            predictor_scores: Series with chapter names as index and scores as values
            weight: Weight for this predictor (should sum to 1.0 across all predictors)
        """
        self.predictors[name] = predictor_scores
        self.weights[name] = weight
        print(f"Added predictor '{name}' with weight {weight}")

    def combine_predictions(self) -> pd.DataFrame:
        """
        Combine all predictor scores using weighted average

        Returns:
            DataFrame with ensemble scores
        """
        # Get all unique topics/chapters
        all_topics = set()
        for scores in self.predictors.values():
            all_topics.update(scores.index)

        # Calculate weighted scores
        combined_scores = {}
        for topic in all_topics:
            score = 0
            total_weight = 0

            for name, predictor_scores in self.predictors.items():
                if topic in predictor_scores.index:
                    score += predictor_scores[topic] * self.weights[name]
                    total_weight += self.weights[name]

            if total_weight > 0:
                combined_scores[topic] = score / total_weight
            else:
                combined_scores[topic] = 0

        # Convert to DataFrame
        self.predictions = pd.DataFrame.from_dict(
            combined_scores,
            orient='index',
            columns=['ensemble_score']
        )

        # Sort by score
        self.predictions = self.predictions.sort_values(
            'ensemble_score',
            ascending=False
        )

        print(f"\n✓ Combined predictions from {len(self.predictors)} models")
        return self.predictions

    def apply_boosting_factors(self,
                              high_frequency_chapters: List[str] = None,
                              recent_trend_boost: Dict[str, float] = None):
        """
        Apply multiplicative boosting factors to certain chapters

        Args:
            high_frequency_chapters: Chapters that appear very frequently (1.1x boost)
            recent_trend_boost: Dict mapping chapters to their boost factors
        """
        if self.predictions is None:
            self.combine_predictions()

        if high_frequency_chapters:
            for chapter in high_frequency_chapters:
                if chapter in self.predictions.index:
                    self.predictions.loc[chapter, 'ensemble_score'] *= 1.1

        if recent_trend_boost:
            for chapter, boost in recent_trend_boost.items():
                if chapter in self.predictions.index:
                    self.predictions.loc[chapter, 'ensemble_score'] *= boost

        # Re-sort after boosting
        self.predictions = self.predictions.sort_values(
            'ensemble_score',
            ascending=False
        )

        return self.predictions

    def get_final_predictions(self, top_n: int = 25) -> pd.DataFrame:
        """
        Get final predictions with confidence levels

        Args:
            top_n: Number of top predictions to return

        Returns:
            DataFrame with top N predictions and confidence levels
        """
        if self.predictions is None:
            self.combine_predictions()

        # Normalize scores to 0-1 range
        max_score = self.predictions['ensemble_score'].max()
        if max_score > 0:
            self.predictions['ensemble_score'] = (
                self.predictions['ensemble_score'] / max_score
            )

        # Get top N
        final = self.predictions.head(top_n).copy()

        # Add confidence levels
        def get_confidence(score):
            if score > 0.7:
                return 'Very High'
            elif score > 0.5:
                return 'High'
            elif score > 0.3:
                return 'Moderate'
            else:
                return 'Low'

        final['confidence'] = final['ensemble_score'].apply(get_confidence)

        # Round scores for display
        final['ensemble_score'] = final['ensemble_score'].round(4)

        return final

    def get_score_breakdown(self, chapter: str) -> Dict:
        """
        Get detailed score breakdown for a specific chapter

        Args:
            chapter: Chapter name

        Returns:
            Dictionary with scores from each predictor
        """
        breakdown = {'chapter': chapter}

        for name, predictor_scores in self.predictors.items():
            if chapter in predictor_scores.index:
                breakdown[name] = round(predictor_scores[chapter], 4)
            else:
                breakdown[name] = 0.0

        if self.predictions is not None and chapter in self.predictions.index:
            breakdown['ensemble'] = round(self.predictions.loc[chapter, 'ensemble_score'], 4)
        else:
            breakdown['ensemble'] = 0.0

        return breakdown

    def compare_predictors(self) -> pd.DataFrame:
        """
        Compare how different predictors rank chapters

        Returns:
            DataFrame showing top chapters from each predictor
        """
        comparison = {}

        for name, scores in self.predictors.items():
            top_chapters = scores.nlargest(10).index.tolist()
            comparison[name] = top_chapters

        return pd.DataFrame(comparison)
