"""
XGBoost model for topic prediction
Focus: Predict which topics will appear (ignore question type)
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

class TopicXGBoostPredictor:
    """XGBoost model for predicting topic appearance"""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with processed topics
        """
        self.df = df
        self.model = None
        self.feature_names = None
        self.topic_features = None
        self.predictions = None

    def engineer_features(self):
        """Engineer rich features for each topic"""
        max_year = self.df['Year'].max()
        total_years = self.df['Year'].nunique()
        all_years = sorted(self.df['Year'].unique())

        features = []

        for topic in self.df['primary_topic'].unique():
            topic_data = self.df[self.df['primary_topic'] == topic]

            # Basic frequency features
            years_appeared = topic_data['Year'].unique()
            frequency = len(years_appeared) / total_years
            last_year = years_appeared.max()
            recency = max_year - last_year

            # Trend features
            recent_2y = topic_data[topic_data['Year'] >= max_year - 1]
            mid_2y = topic_data[(topic_data['Year'] >= max_year - 3) & (topic_data['Year'] < max_year - 1)]
            old_4y = topic_data[topic_data['Year'] < max_year - 3]

            recent_count = len(recent_2y)
            mid_count = len(mid_2y)
            old_count = len(old_4y)

            # Trend ratio
            if old_count > 0:
                trend_ratio = recent_count / old_count
            else:
                trend_ratio = 2.0 if recent_count > 0 else 0.0

            # Consecutive years (pattern consistency)
            years_list = sorted(years_appeared)
            consecutive_years = 0
            max_gap = 0
            if len(years_list) > 1:
                for i in range(len(years_list) - 1):
                    gap = years_list[i+1] - years_list[i]
                    if gap == 1:
                        consecutive_years += 1
                    max_gap = max(max_gap, gap)

            # Question volume features
            total_questions = len(topic_data)
            avg_questions_per_year = total_questions / len(years_appeared) if len(years_appeared) > 0 else 0

            # Marks features
            total_marks = topic_data['Marks'].sum()
            avg_marks_per_question = topic_data['Marks'].mean()
            max_marks_in_year = topic_data.groupby('Year')['Marks'].sum().max()

            # Question type diversity
            unique_qtypes = topic_data['Question type'].nunique()
            qtype_entropy = self._calculate_entropy(topic_data['Question type'].value_counts(normalize=True))

            # Temporal patterns (binary vector: appeared in each year?)
            year_vector = [1 if year in years_appeared else 0 for year in all_years]

            # Recent momentum (appeared in last 2 years?)
            appeared_recently = 1 if any(y >= max_year - 1 for y in years_appeared) else 0

            # Regularity score (std of gaps between appearances)
            if len(years_list) > 1:
                gaps = [years_list[i+1] - years_list[i] for i in range(len(years_list) - 1)]
                regularity = 1 / (np.std(gaps) + 1)  # Lower std = more regular
            else:
                regularity = 0

            feature_dict = {
                'topic': topic,
                'frequency': frequency,
                'recency': recency,
                'recent_count': recent_count,
                'mid_count': mid_count,
                'old_count': old_count,
                'trend_ratio': trend_ratio,
                'consecutive_years': consecutive_years,
                'max_gap': max_gap,
                'total_questions': total_questions,
                'avg_questions_per_year': avg_questions_per_year,
                'total_marks': total_marks,
                'avg_marks_per_question': avg_marks_per_question,
                'max_marks_in_year': max_marks_in_year,
                'unique_qtypes': unique_qtypes,
                'qtype_entropy': qtype_entropy,
                'appeared_recently': appeared_recently,
                'regularity': regularity,
                'years_appeared_count': len(years_appeared),
                'last_year': last_year
            }

            # Add year vector as separate features
            for i, year in enumerate(all_years):
                feature_dict[f'year_{year}'] = year_vector[i]

            features.append(feature_dict)

        self.topic_features = pd.DataFrame(features)

        # Define feature columns (exclude topic, last_year)
        self.feature_names = [col for col in self.topic_features.columns
                             if col not in ['topic', 'last_year']]

        return self.topic_features

    def _calculate_entropy(self, probs):
        """Calculate Shannon entropy"""
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def train_model(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        """
        Train XGBoost model

        Target: Will topic appear in next year? (binary classification)
        """
        if self.topic_features is None:
            self.engineer_features()

        print("Training XGBoost model...")

        # For training, we'll use leave-one-year-out approach
        # Use all years except last to train, last year as validation

        max_year = self.df['Year'].max()

        # Create training data: predict if each topic appeared in most recent year
        X = self.topic_features[self.feature_names]
        y = (self.topic_features['last_year'] == max_year).astype(int)

        print(f"  Features: {len(self.feature_names)}")
        print(f"  Topics: {len(X)}")
        print(f"  Positive samples (appeared in {max_year}): {y.sum()}")
        print(f"  Negative samples: {(1-y).sum()}")

        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            eval_metric='logloss'
        )

        self.model.fit(X, y)

        print("✓ Model trained")

        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Important Features:")
        print(importance.head(10).to_string(index=False))

        return self.model

    def predict(self):
        """Predict topic probabilities for next year"""
        if self.model is None:
            self.train_model()

        X = self.topic_features[self.feature_names]

        # Get probability of appearing
        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of class 1 (will appear)

        self.predictions = pd.DataFrame({
            'topic': self.topic_features['topic'],
            'probability': probabilities
        }).sort_values('probability', ascending=False)

        return self.predictions

    def cross_validate_temporal(self):
        """
        Temporal cross-validation: train on older years, test on recent years
        """
        print("\n" + "="*70)
        print("TEMPORAL CROSS-VALIDATION")
        print("="*70)

        years = sorted(self.df['Year'].unique())

        # Use last 2 years for validation
        results = []

        for test_year in years[-2:]:
            print(f"\nTest Year: {test_year}")
            print("-"*70)

            # Split data
            train_data = self.df[self.df['Year'] < test_year]
            test_data = self.df[self.df['Year'] == test_year]

            # Build features from training data
            temp_predictor = TopicXGBoostPredictor(train_data)
            temp_predictor.engineer_features()
            temp_predictor.train_model()

            # Get predictions
            predictions = temp_predictor.predict()

            # Evaluate
            predicted_topics = set(predictions.head(30)['topic'])  # Top 30
            actual_topics = set(test_data['primary_topic'].unique())

            hits = len(predicted_topics & actual_topics)
            precision = hits / len(predicted_topics) if predicted_topics else 0
            recall = hits / len(actual_topics) if actual_topics else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"Predicted (Top 30): {len(predicted_topics)}")
            print(f"Actual: {len(actual_topics)}")
            print(f"Hits: {hits}")
            print(f"Precision: {precision:.2%}")
            print(f"Recall: {recall:.2%}")
            print(f"F1 Score: {f1:.2%}")

            results.append({
                'test_year': test_year,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

        # Average metrics
        avg_results = {
            'avg_precision': np.mean([r['precision'] for r in results]),
            'avg_recall': np.mean([r['recall'] for r in results]),
            'avg_f1': np.mean([r['f1'] for r in results])
        }

        print("\n" + "="*70)
        print("AVERAGE PERFORMANCE:")
        print(f"Precision: {avg_results['avg_precision']:.2%}")
        print(f"Recall: {avg_results['avg_recall']:.2%}")
        print(f"F1 Score: {avg_results['avg_f1']:.2%}")

        return results
