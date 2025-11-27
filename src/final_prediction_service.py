"""
Final Prediction Service for Agent Integration

Architecture:
1. Predict topics using baseline model (77% F1)
2. For each topic, provide historical question type information
3. Return structured data for RAG/agent queries
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
from collections import Counter

class FinalPredictionService:
    """
    Main service for topic predictions with historical question types
    """

    def __init__(self, processed_data_path: str):
        """
        Args:
            processed_data_path: Path to processed CSV with topics
        """
        self.df = pd.read_csv(processed_data_path)
        self.topic_predictions = None
        self.qtype_history = None
        self._build_service()

    def _build_service(self):
        """Build topic predictions and question type history"""
        print("Building Final Prediction Service...")

        # 1. Build topic predictions (baseline model)
        self._predict_topics()

        # 2. Build question type history for each topic
        self._build_qtype_history()

        print("✓ Service ready")

    def _predict_topics(self):
        """Predict topics using baseline model"""
        max_year = self.df['Year'].max()
        total_years = self.df['Year'].nunique()

        features = []
        for topic in self.df['primary_topic'].unique():
            topic_data = self.df[self.df['primary_topic'] == topic]

            # Basic features
            years_appeared = topic_data['Year'].unique()
            frequency = len(years_appeared) / total_years
            last_year = years_appeared.max()
            recency = max_year - last_year

            # Trend
            recent_years = topic_data[topic_data['Year'] >= max_year - 1]
            older_years = topic_data[topic_data['Year'] < max_year - 1]
            if len(older_years) > 0:
                trend = len(recent_years) / len(older_years)
            else:
                trend = 2.0

            # Heat score
            max_recency = 8  # Known max
            norm_recency = recency / max_recency if max_recency > 0 else 0

            score = (
                frequency *
                (1 + 0.3 * norm_recency) *
                np.sqrt(trend)
            )

            features.append({
                'topic': topic,
                'probability': score,
                'frequency': frequency,
                'recency': recency,
                'trend': trend,
                'years_appeared': len(years_appeared),
                'last_year': last_year,
                'total_questions': len(topic_data),
                'total_marks': topic_data['Marks'].sum()
            })

        self.topic_predictions = pd.DataFrame(features)

        # Normalize probabilities
        max_score = self.topic_predictions['probability'].max()
        if max_score > 0:
            self.topic_predictions['probability'] = (
                self.topic_predictions['probability'] / max_score
            )

        self.topic_predictions = self.topic_predictions.sort_values(
            'probability', ascending=False
        )

    def _build_qtype_history(self):
        """Build historical question type information for each topic"""
        qtype_history = {}

        for topic in self.df['primary_topic'].unique():
            topic_data = self.df[self.df['primary_topic'] == topic]

            # Count question types
            qtype_counts = topic_data['Question type'].value_counts()

            # Get years for each type
            qtype_years = {}
            for qtype in qtype_counts.index:
                years = topic_data[topic_data['Question type'] == qtype]['Year'].unique()
                qtype_years[qtype] = sorted(years.tolist())

            # Most common type
            most_common_type = qtype_counts.index[0]
            most_common_count = qtype_counts.iloc[0]

            # Average marks per type
            avg_marks_per_type = topic_data.groupby('Question type')['Marks'].mean().to_dict()

            qtype_history[topic] = {
                'all_types': qtype_counts.to_dict(),
                'most_common_type': most_common_type,
                'most_common_count': int(most_common_count),
                'type_years': qtype_years,
                'avg_marks_per_type': {k: round(v, 1) for k, v in avg_marks_per_type.items()}
            }

        self.qtype_history = qtype_history

    def get_top_predictions(self, top_n: int = 25, min_probability: float = 0.0) -> List[Dict]:
        """
        Get top N topic predictions with question type history

        Args:
            top_n: Number of topics to return
            min_probability: Minimum probability threshold

        Returns:
            List of prediction dictionaries
        """
        filtered = self.topic_predictions[
            self.topic_predictions['probability'] >= min_probability
        ]

        results = []
        for idx, row in filtered.head(top_n).iterrows():
            topic = row['topic']
            qtype_info = self.qtype_history.get(topic, {})

            # Confidence level
            prob = row['probability']
            if prob > 0.7:
                confidence = 'Very High'
            elif prob > 0.5:
                confidence = 'High'
            elif prob > 0.3:
                confidence = 'Moderate'
            else:
                confidence = 'Low'

            # Reasoning
            reasons = []
            if row['frequency'] >= 0.7:
                reasons.append(f"appears very frequently ({row['years_appeared']}/8 years)")
            elif row['frequency'] >= 0.4:
                reasons.append(f"appears regularly ({row['years_appeared']}/8 years)")

            if row['recency'] >= 2:
                reasons.append(f"hasn't appeared in {int(row['recency'])} years (due)")
            elif row['recency'] == 0:
                reasons.append("appeared in most recent year")

            if row['trend'] > 1.2:
                reasons.append("showing increasing trend")

            # Sample questions (get a few)
            topic_questions = self.df[self.df['primary_topic'] == topic]
            sample_qs = topic_questions.head(3)['Question'].tolist()

            result = {
                'rank': len(results) + 1,
                'topic': topic,
                'probability_score': round(prob, 4),
                'confidence': confidence,
                'frequency': f"{row['years_appeared']}/8 years",
                'last_appeared': int(row['last_year']),
                'total_questions_historical': int(row['total_questions']),
                'total_marks_historical': int(row['total_marks']),
                'reasoning': "; ".join(reasons) if reasons else "appeared historically",

                # Question type history
                'historical_question_types': list(qtype_info.get('all_types', {}).keys()),
                'most_common_question_type': qtype_info.get('most_common_type', 'Unknown'),
                'question_type_distribution': qtype_info.get('all_types', {}),
                'question_type_years': qtype_info.get('type_years', {}),
                'avg_marks_per_type': qtype_info.get('avg_marks_per_type', {}),

                # Sample questions
                'sample_questions': sample_qs[:2]  # Return 2 samples
            }

            results.append(result)

        return results

    def check_topic_probability(self, topic_query: str) -> Optional[Dict]:
        """
        Check probability for a specific topic (fuzzy match)

        Args:
            topic_query: Topic name or partial match

        Returns:
            Prediction dict or None
        """
        # Fuzzy search
        matches = self.topic_predictions[
            self.topic_predictions['topic'].str.contains(topic_query, case=False, na=False)
        ]

        if len(matches) == 0:
            return None

        # Get best match
        top_match = matches.iloc[0]
        full_predictions = self.get_top_predictions(top_n=len(self.topic_predictions))

        # Find in full predictions
        for pred in full_predictions:
            if pred['topic'] == top_match['topic']:
                return pred

        return None

    def get_chapter_topics(self, chapter_name: str, top_n: int = 10) -> List[Dict]:
        """
        Get top topics from a specific chapter

        Args:
            chapter_name: Chapter name (partial match)
            top_n: Number of topics to return

        Returns:
            List of predictions for that chapter
        """
        # Find topics from this chapter
        chapter_data = self.df[
            self.df['Chapter_Name'].str.contains(chapter_name, case=False, na=False)
        ]

        if len(chapter_data) == 0:
            return []

        chapter_topics = set(chapter_data['primary_topic'].unique())

        # Filter predictions
        all_preds = self.get_top_predictions(top_n=100)
        chapter_preds = [p for p in all_preds if p['topic'] in chapter_topics]

        return chapter_preds[:top_n]

    def export_for_rag(self, output_path: str, top_n: int = 50):
        """
        Export predictions in JSON format for RAG system

        Args:
            output_path: Path to save JSON
            top_n: Number of predictions to export
        """
        predictions = self.get_top_predictions(top_n=top_n)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        print(f"✓ Exported {len(predictions)} predictions to {output_path}")

    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        return {
            'total_topics': len(self.topic_predictions),
            'high_confidence_topics': len(self.topic_predictions[self.topic_predictions['probability'] > 0.7]),
            'medium_confidence_topics': len(self.topic_predictions[
                (self.topic_predictions['probability'] > 0.5) &
                (self.topic_predictions['probability'] <= 0.7)
            ]),
            'total_questions_analyzed': len(self.df),
            'years_covered': self.df['Year'].nunique(),
            'unique_question_types': self.df['Question type'].nunique()
        }
