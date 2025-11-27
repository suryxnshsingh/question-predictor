"""
Decoupled prediction system:
- Model A: Predict which topics will appear
- Model B: Predict question type distribution
- Combine them probabilistically
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

class TopicOnlyPredictor:
    """Predicts which topics will appear (ignoring question type)"""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with processed topics
        """
        self.df = df
        self.topic_features = None
        self.predictions = None

    def build_features(self):
        """Build features for each topic"""
        max_year = self.df['Year'].max()
        total_years = self.df['Year'].nunique()

        features = []

        for topic in self.df['primary_topic'].unique():
            topic_data = self.df[self.df['primary_topic'] == topic]

            # Years appeared
            years_appeared = topic_data['Year'].unique()
            frequency = len(years_appeared) / total_years

            # Recency
            last_year = years_appeared.max()
            recency = max_year - last_year

            # Trend (more in recent years?)
            recent_years = topic_data[topic_data['Year'] >= max_year - 1]
            older_years = topic_data[topic_data['Year'] < max_year - 1]
            if len(older_years) > 0:
                trend = len(recent_years) / len(older_years)
            else:
                trend = 2.0  # New pattern

            # Total questions and marks
            total_questions = len(topic_data)
            total_marks = topic_data['Marks'].sum()
            avg_marks = topic_data['Marks'].mean()

            # Question type diversity (how many different types)
            type_diversity = topic_data['Question type'].nunique()

            features.append({
                'topic': topic,
                'frequency': frequency,
                'recency': recency,
                'trend': trend,
                'total_questions': total_questions,
                'total_marks': total_marks,
                'avg_marks': avg_marks,
                'type_diversity': type_diversity,
                'last_year': last_year,
                'years_appeared': len(years_appeared)
            })

        self.topic_features = pd.DataFrame(features)
        return self.topic_features

    def predict(self, method='baseline'):
        """
        Predict topic probabilities

        Args:
            method: 'baseline' or 'ml' (for future ML models)
        """
        if self.topic_features is None:
            self.build_features()

        if method == 'baseline':
            return self._predict_baseline()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _predict_baseline(self):
        """Simple baseline prediction"""
        # Heat score
        max_recency = self.topic_features['recency'].max()
        if max_recency > 0:
            norm_recency = self.topic_features['recency'] / max_recency
        else:
            norm_recency = 0

        score = (
            self.topic_features['frequency'] *
            (1 + 0.3 * norm_recency) *
            np.sqrt(self.topic_features['trend'])
        )

        # Normalize to probabilities (0-1)
        if score.max() > 0:
            score = score / score.max()

        self.predictions = pd.DataFrame({
            'topic': self.topic_features['topic'],
            'probability': score
        }).sort_values('probability', ascending=False)

        return self.predictions


class QuestionTypeDistributionPredictor:
    """Predicts the distribution of question types (globally)"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.predictions = None

    def predict(self, method='baseline'):
        """
        Predict question type distribution for next year

        Args:
            method: 'baseline' or 'trend'
        """
        if method == 'baseline':
            return self._predict_baseline()
        elif method == 'trend':
            return self._predict_with_trend()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _predict_baseline(self):
        """Simple: use average distribution from recent years"""
        # Weight recent years more heavily
        max_year = self.df['Year'].max()

        # Get distributions per year
        yearly_dist = []
        for year in self.df['Year'].unique():
            year_data = self.df[self.df['Year'] == year]
            dist = year_data['Question type'].value_counts(normalize=True)

            # Weight: recent years get higher weight
            weight = 1.0 + (year - (max_year - 3)) * 0.2 if year >= max_year - 3 else 1.0
            yearly_dist.append((dist, weight))

        # Weighted average
        all_types = set()
        for dist, _ in yearly_dist:
            all_types.update(dist.index)

        weighted_dist = {}
        for qtype in all_types:
            weighted_sum = sum(dist.get(qtype, 0) * weight for dist, weight in yearly_dist)
            total_weight = sum(weight for _, weight in yearly_dist)
            weighted_dist[qtype] = weighted_sum / total_weight

        self.predictions = pd.Series(weighted_dist).sort_values(ascending=False)
        return self.predictions

    def _predict_with_trend(self):
        """Predict with trend analysis (for recent shifts)"""
        max_year = self.df['Year'].max()

        # Compare recent 2 years vs older years
        recent = self.df[self.df['Year'] >= max_year - 1]
        older = self.df[self.df['Year'] < max_year - 1]

        recent_dist = recent['Question type'].value_counts(normalize=True)
        older_dist = older['Question type'].value_counts(normalize=True)

        # Blend: 70% recent, 30% older (favor recent trends)
        all_types = set(list(recent_dist.index) + list(older_dist.index))

        blended = {}
        for qtype in all_types:
            recent_val = recent_dist.get(qtype, 0)
            older_val = older_dist.get(qtype, 0)
            blended[qtype] = 0.7 * recent_val + 0.3 * older_val

        self.predictions = pd.Series(blended).sort_values(ascending=False)
        return self.predictions


class DecoupledCombiner:
    """Combines topic predictions and question type distribution"""

    def __init__(self, topic_predictor: TopicOnlyPredictor,
                 qtype_predictor: QuestionTypeDistributionPredictor):
        self.topic_predictor = topic_predictor
        self.qtype_predictor = qtype_predictor
        self.combined_predictions = None

    def combine(self):
        """Combine topic and question type predictions"""
        topic_probs = self.topic_predictor.predictions
        qtype_dist = self.qtype_predictor.predictions

        # Create all combinations
        combinations = []
        for _, topic_row in topic_probs.iterrows():
            topic = topic_row['topic']
            topic_prob = topic_row['probability']

            for qtype, qtype_prob in qtype_dist.items():
                # Combined probability (independent assumption)
                combined_prob = topic_prob * qtype_prob

                combinations.append({
                    'topic': topic,
                    'question_type': qtype,
                    'topic_probability': topic_prob,
                    'qtype_probability': qtype_prob,
                    'combined_score': combined_prob
                })

        self.combined_predictions = pd.DataFrame(combinations)
        self.combined_predictions = self.combined_predictions.sort_values(
            'combined_score', ascending=False
        )

        # Normalize combined scores
        max_score = self.combined_predictions['combined_score'].max()
        if max_score > 0:
            self.combined_predictions['combined_score'] = (
                self.combined_predictions['combined_score'] / max_score
            )

        return self.combined_predictions

    def get_top_predictions(self, top_n: int = 50) -> pd.DataFrame:
        """Get top N predictions"""
        if self.combined_predictions is None:
            self.combine()

        return self.combined_predictions.head(top_n)

    def get_confidence(self, score: float) -> str:
        """Map score to confidence level"""
        if score > 0.7:
            return 'Very High'
        elif score > 0.5:
            return 'High'
        elif score > 0.3:
            return 'Moderate'
        else:
            return 'Low'
