"""
Optimized Prediction System

Key improvements:
1. Fixed baseline to handle high-frequency topics correctly
2. Only 3 diverse models (no redundancy)
3. Performance-based weighting
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from collections import defaultdict

class OptimizedBaselineModel:
    """
    Improved baseline with CORRECT handling of high-frequency topics
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.max_year = df['Year'].max()
        self.total_years = df['Year'].nunique()
        self.predictions = None

    def predict(self):
        """Predict with improved logic"""
        scores = {}

        for topic in self.df['primary_topic'].unique():
            topic_data = self.df[self.df['primary_topic'] == topic]

            years_appeared = topic_data['Year'].unique()
            frequency = len(years_appeared) / self.total_years
            last_year = years_appeared.max()
            recency = self.max_year - last_year

            # Trend
            recent = topic_data[topic_data['Year'] >= self.max_year - 1]
            older = topic_data[topic_data['Year'] < self.max_year - 1]
            trend = len(recent) / len(older) if len(older) > 0 else 2.0

            # KEY FIX: Different scoring for high vs low frequency topics
            if frequency >= 0.75:  # High frequency (appears 6+ out of 8 years)
                # Core topics: frequency matters most, recency doesn't penalize
                score = frequency * (1.0 + 0.1 * (1 / (recency + 1)))  # Slight boost if just appeared
            else:  # Low/medium frequency
                # Rare topics: recency matters (if not appeared recently, more likely)
                score = frequency * (1 + 0.4 * recency / 8.0) * np.sqrt(trend)

            scores[topic] = score

        # Normalize
        max_score = max(scores.values())
        self.predictions = {k: v/max_score for k, v in scores.items()}

        return pd.DataFrame([
            {'topic': topic, 'probability': score}
            for topic, score in scores.items()
        ]).sort_values('probability', ascending=False)


class OptimizedAssociationModel:
    """Association rules - adds diversity by looking at co-occurrence"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.max_year = df['Year'].max()
        self.total_years = df['Year'].nunique()
        self.predictions = None

    def predict(self):
        """Score based on co-occurrence with high-confidence topics"""
        # Get high frequency topics (appeared 6+ years)
        topic_freq = self.df.groupby('primary_topic')['Year'].nunique()
        high_freq_topics = set(topic_freq[topic_freq >= 6].index)

        scores = {}
        for topic in self.df['primary_topic'].unique():
            topic_data = self.df[self.df['primary_topic'] == topic]
            years_appeared = set(topic_data['Year'].unique())

            # Base score
            base_score = len(years_appeared) / self.total_years

            # Boost based on co-occurrence with high-freq topics
            boost = 0
            for high_freq_topic in high_freq_topics:
                if high_freq_topic == topic:
                    continue
                high_freq_years = set(
                    self.df[self.df['primary_topic'] == high_freq_topic]['Year'].unique()
                )
                overlap = len(years_appeared & high_freq_years)
                if overlap >= 4:  # Co-occurred in 4+ years
                    boost += 0.15

            scores[topic] = min(base_score + boost, 1.0)

        self.predictions = scores
        return pd.DataFrame([
            {'topic': topic, 'probability': score}
            for topic, score in scores.items()
        ]).sort_values('probability', ascending=False)


class OptimizedMarkovModel:
    """Markov chain - adds diversity by looking at transition patterns"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.max_year = df['Year'].max()
        self.total_years = df['Year'].nunique()
        self.predictions = None

    def predict(self):
        """Predict based on appearance patterns"""
        transition_counts = defaultdict(lambda: {'appeared': 0, 'not_appeared': 0})

        all_years = sorted(self.df['Year'].unique())

        for topic in self.df['primary_topic'].unique():
            topic_years = set(self.df[self.df['primary_topic'] == topic]['Year'].unique())

            for i in range(len(all_years) - 1):
                year = all_years[i]
                next_year = all_years[i + 1]

                appeared_this_year = year in topic_years
                appeared_next_year = next_year in topic_years

                if appeared_this_year:
                    if appeared_next_year:
                        transition_counts[topic]['appeared'] += 1
                    else:
                        transition_counts[topic]['not_appeared'] += 1

        scores = {}
        for topic in self.df['primary_topic'].unique():
            appeared = transition_counts[topic]['appeared']
            not_appeared = transition_counts[topic]['not_appeared']
            total = appeared + not_appeared

            if total > 0:
                prob = appeared / total
            else:
                prob = len(self.df[self.df['primary_topic'] == topic]['Year'].unique()) / self.total_years

            # Boost if appeared in last year
            last_year_appeared = self.max_year in self.df[self.df['primary_topic'] == topic]['Year'].values
            if last_year_appeared:
                prob = min(prob * 1.15, 1.0)

            scores[topic] = prob

        self.predictions = scores
        return pd.DataFrame([
            {'topic': topic, 'probability': score}
            for topic, score in scores.items()
        ]).sort_values('probability', ascending=False)


class OptimizedEnsemble:
    """
    Optimized ensemble with only 3 diverse models
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {
            'baseline': OptimizedBaselineModel(df),
            'association': OptimizedAssociationModel(df),
            'markov': OptimizedMarkovModel(df)
        }
        self.predictions = {}
        self.ensemble_predictions = None

    def build_all(self):
        """Build all models"""
        print("Building Optimized Ensemble...")
        print("-"*70)

        for name, model in self.models.items():
            model.predict()
            self.predictions[name] = model.predictions
            print(f"✓ {name.capitalize()} model ready")

        print("✓ All models built")

    def ensemble_predict(self, weights=None):
        """
        Combine with performance-based weights

        Default weights based on expected performance:
        - Baseline: 60% (strongest, most reliable)
        - Association: 25% (adds co-occurrence signal)
        - Markov: 15% (adds transition pattern)
        """
        if weights is None:
            weights = {
                'baseline': 0.60,
                'association': 0.25,
                'markov': 0.15
            }

        ensemble_scores = {}
        all_topics = self.df['primary_topic'].unique()

        for topic in all_topics:
            score = 0
            for model_name in self.models.keys():
                if isinstance(self.predictions[model_name], dict):
                    model_score = self.predictions[model_name].get(topic, 0)
                else:  # DataFrame
                    model_df = self.predictions[model_name]
                    model_score = model_df[model_df['topic'] == topic]['probability'].values
                    model_score = model_score[0] if len(model_score) > 0 else 0

                score += model_score * weights[model_name]

            ensemble_scores[topic] = score

        # Normalize
        max_score = max(ensemble_scores.values())
        if max_score > 0:
            ensemble_scores = {k: v/max_score for k, v in ensemble_scores.items()}

        self.ensemble_predictions = pd.DataFrame([
            {'topic': topic, 'probability': score}
            for topic, score in ensemble_scores.items()
        ]).sort_values('probability', ascending=False)

        return self.ensemble_predictions

    def validate(self, actual_topics: set, top_n: int = 25):
        """Validate against actual topics"""
        if self.ensemble_predictions is None:
            self.ensemble_predict()

        predicted = set(self.ensemble_predictions.head(top_n)['topic'])
        hits = len(predicted & actual_topics)
        precision = hits / len(predicted)
        recall = hits / len(actual_topics)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'hits': hits,
            'predicted': len(predicted),
            'actual': len(actual_topics)
        }
