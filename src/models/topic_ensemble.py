"""
Complete Ensemble System for Topic Prediction

Combines multiple models:
1. Baseline (frequency + recency + trend)
2. ARIMA (time series)
3. Association Rules (co-occurrence)
4. Random Forest
5. Markov Chain (sequences)
6. Weighted Average
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class TopicEnsemble:
    """Ensemble combining multiple topic prediction models"""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with processed topics
        """
        self.df = df
        self.models = {}
        self.predictions = {}
        self.ensemble_predictions = None

        self.max_year = df['Year'].max()
        self.total_years = df['Year'].nunique()
        self.all_topics = df['primary_topic'].unique()

    def build_all_models(self):
        """Build all individual models"""
        print("Building Ensemble Models...")
        print("="*70)

        self.build_baseline_model()
        self.build_arima_model()
        self.build_association_model()
        self.build_random_forest_model()
        self.build_markov_model()
        self.build_weighted_average_model()

        print("\n✓ All models built")

    def build_baseline_model(self):
        """Model 1: Baseline (frequency + recency + trend)"""
        print("\n1. Building Baseline Model...")

        scores = {}
        for topic in self.all_topics:
            topic_data = self.df[self.df['primary_topic'] == topic]

            years_appeared = topic_data['Year'].unique()
            frequency = len(years_appeared) / self.total_years
            last_year = years_appeared.max()
            recency = self.max_year - last_year

            # Trend
            recent = topic_data[topic_data['Year'] >= self.max_year - 1]
            older = topic_data[topic_data['Year'] < self.max_year - 1]
            trend = len(recent) / len(older) if len(older) > 0 else 2.0

            # Score
            norm_recency = recency / 8.0  # Max 8 years
            score = frequency * (1 + 0.3 * norm_recency) * np.sqrt(trend)
            scores[topic] = score

        # Normalize
        max_score = max(scores.values())
        self.predictions['baseline'] = {k: v/max_score for k, v in scores.items()}
        print(f"   ✓ Scored {len(scores)} topics")

    def build_arima_model(self):
        """Model 2: ARIMA time series"""
        print("\n2. Building ARIMA Model...")

        from pmdarima import auto_arima

        # Create topic appearance matrix
        topic_matrix = pd.crosstab(
            index=self.df['Year'],
            columns=self.df['primary_topic'],
            values=self.df['Marks'],
            aggfunc='sum'
        ).fillna(0)
        topic_matrix = (topic_matrix > 0).astype(int)

        scores = {}
        for topic in self.all_topics:
            if topic not in topic_matrix.columns:
                scores[topic] = 0.5
                continue

            series = topic_matrix[topic].values

            if len(series) < 3:
                scores[topic] = series.mean()
                continue

            try:
                model = auto_arima(
                    series,
                    start_p=0, start_q=0,
                    max_p=2, max_q=2,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    max_order=5
                )

                forecast = model.predict(n_periods=1)[0]
                probability = min(max(forecast, 0), 1)
                scores[topic] = probability

            except:
                scores[topic] = series.mean()

        self.predictions['arima'] = scores
        print(f"   ✓ Scored {len(scores)} topics")

    def build_association_model(self):
        """Model 3: Association Rules (topic co-occurrence)"""
        print("\n3. Building Association Rules Model...")

        # Simple co-occurrence based scoring
        # Topics that often appear together with high-frequency topics get boosted

        # Get high frequency topics
        topic_freq = self.df.groupby('primary_topic')['Year'].nunique()
        high_freq_topics = set(topic_freq[topic_freq >= 6].index)  # Appeared 6+ years

        scores = {}
        for topic in self.all_topics:
            topic_data = self.df[self.df['primary_topic'] == topic]
            years_appeared = set(topic_data['Year'].unique())

            # Base score: frequency
            base_score = len(years_appeared) / self.total_years

            # Boost if appears in same years as high-freq topics
            boost = 0
            for high_freq_topic in high_freq_topics:
                if high_freq_topic == topic:
                    continue
                high_freq_years = set(self.df[self.df['primary_topic'] == high_freq_topic]['Year'].unique())
                overlap = len(years_appeared & high_freq_years)
                if overlap >= 4:  # Co-occurred in 4+ years
                    boost += 0.1

            scores[topic] = min(base_score + boost, 1.0)

        self.predictions['association'] = scores
        print(f"   ✓ Scored {len(scores)} topics")

    def build_random_forest_model(self):
        """Model 4: Random Forest"""
        print("\n4. Building Random Forest Model...")

        # Build features (same as XGBoost but use RF)
        features = []
        labels = []

        for topic in self.all_topics:
            topic_data = self.df[self.df['primary_topic'] == topic]

            years_appeared = topic_data['Year'].unique()
            frequency = len(years_appeared) / self.total_years
            last_year = years_appeared.max()
            recency = self.max_year - last_year

            # More features
            total_questions = len(topic_data)
            total_marks = topic_data['Marks'].sum()
            avg_marks = topic_data['Marks'].mean()

            # Recent appearances
            recent_count = len(topic_data[topic_data['Year'] >= self.max_year - 1])
            mid_count = len(topic_data[(topic_data['Year'] >= self.max_year - 3) & (topic_data['Year'] < self.max_year - 1)])

            features.append([
                frequency, recency, total_questions, total_marks,
                avg_marks, recent_count, mid_count, len(years_appeared)
            ])

            # Label: appeared in last year?
            labels.append(1 if last_year == self.max_year else 0)

        # Train RF
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            min_samples_split=5
        )

        X = np.array(features)
        y = np.array(labels)

        rf.fit(X, y)

        # Get probabilities
        probabilities = rf.predict_proba(X)[:, 1]

        scores = {topic: float(prob) for topic, prob in zip(self.all_topics, probabilities)}
        self.predictions['random_forest'] = scores
        print(f"   ✓ Scored {len(scores)} topics")

    def build_markov_model(self):
        """Model 5: Markov Chain (topic appearance sequences)"""
        print("\n5. Building Markov Chain Model...")

        # Simple: if appeared in year N, what's prob of appearing in N+1?
        transition_counts = defaultdict(lambda: {'appeared': 0, 'not_appeared': 0})

        all_years = sorted(self.df['Year'].unique())

        for topic in self.all_topics:
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

        # Calculate probabilities
        scores = {}
        for topic in self.all_topics:
            appeared = transition_counts[topic]['appeared']
            not_appeared = transition_counts[topic]['not_appeared']
            total = appeared + not_appeared

            if total > 0:
                prob = appeared / total
            else:
                # Not enough data, use frequency
                prob = len(self.df[self.df['primary_topic'] == topic]['Year'].unique()) / self.total_years

            # Check if appeared in last year (boost if yes)
            last_year_appeared = self.max_year in self.df[self.df['primary_topic'] == topic]['Year'].values
            if last_year_appeared:
                prob = min(prob * 1.2, 1.0)

            scores[topic] = prob

        self.predictions['markov'] = scores
        print(f"   ✓ Scored {len(scores)} topics")

    def build_weighted_average_model(self):
        """Model 6: Weighted Average (recent years weighted more)"""
        print("\n6. Building Weighted Average Model...")

        scores = {}
        years = sorted(self.df['Year'].unique())

        # Weights: recent years get higher weight
        weights = {year: 1.0 + (i / len(years)) for i, year in enumerate(years)}

        for topic in self.all_topics:
            topic_data = self.df[self.df['primary_topic'] == topic]
            topic_years = topic_data['Year'].unique()

            # Weighted sum
            weighted_sum = sum(weights[year] for year in topic_years)
            total_weight = sum(weights.values())

            scores[topic] = weighted_sum / total_weight

        self.predictions['weighted_avg'] = scores
        print(f"   ✓ Scored {len(scores)} topics")

    def ensemble_predict(self, weights=None):
        """
        Combine all models using weighted average

        Args:
            weights: Dict of model weights (default: equal weights)
        """
        if weights is None:
            weights = {
                'baseline': 0.30,
                'arima': 0.25,
                'association': 0.15,
                'random_forest': 0.20,
                'markov': 0.05,
                'weighted_avg': 0.05
            }

        print(f"\nEnsemble Weights: {weights}")

        ensemble_scores = {}

        for topic in self.all_topics:
            score = 0
            for model_name, model_preds in self.predictions.items():
                score += model_preds.get(topic, 0) * weights.get(model_name, 0)

            ensemble_scores[topic] = score

        # Normalize
        max_score = max(ensemble_scores.values())
        if max_score > 0:
            ensemble_scores = {k: v/max_score for k, v in ensemble_scores.items()}

        # Sort
        self.ensemble_predictions = pd.DataFrame([
            {'topic': topic, 'probability': score}
            for topic, score in ensemble_scores.items()
        ]).sort_values('probability', ascending=False)

        return self.ensemble_predictions

    def get_individual_model_predictions(self, model_name: str, top_n: int = 30) -> List[str]:
        """Get top N predictions from a specific model"""
        if model_name not in self.predictions:
            return []

        scores = self.predictions[model_name]
        sorted_topics = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:top_n]]

    def compare_models(self, actual_topics: set, top_n: int = 25):
        """Compare all models against actual topics"""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)

        results = {}

        # Individual models
        for model_name in self.predictions.keys():
            predicted = set(self.get_individual_model_predictions(model_name, top_n))
            hits = len(predicted & actual_topics)
            precision = hits / len(predicted) if predicted else 0
            recall = hits / len(actual_topics) if actual_topics else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            results[model_name] = {'precision': precision, 'recall': recall, 'f1': f1}
            print(f"{model_name:20} P={precision:.2%} R={recall:.2%} F1={f1:.2%}")

        # Ensemble
        if self.ensemble_predictions is not None:
            predicted = set(self.ensemble_predictions.head(top_n)['topic'])
            hits = len(predicted & actual_topics)
            precision = hits / len(predicted) if predicted else 0
            recall = hits / len(actual_topics) if actual_topics else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            results['ensemble'] = {'precision': precision, 'recall': recall, 'f1': f1}
            print(f"{'ENSEMBLE':20} P={precision:.2%} R={recall:.2%} F1={f1:.2%} ⭐")

        return results
