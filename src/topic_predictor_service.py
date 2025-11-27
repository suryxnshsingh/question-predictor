"""
Topic-level prediction service
Main queryable service for the agent
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
from pathlib import Path

class TopicPredictorService:
    def __init__(self, processed_data_path: str):
        """
        Initialize the prediction service

        Args:
            processed_data_path: Path to processed data with topics
        """
        self.df = pd.read_csv(processed_data_path)
        self.predictions = None
        self._build_predictions()

    def _build_predictions(self):
        """Build topic-level predictions"""
        print("Building topic-level predictions...")

        # Create topic + question_type combinations
        topic_question_combos = self.df.groupby(['primary_topic', 'Question type']).agg({
            'Year': ['count', 'nunique', lambda x: list(x.unique())],
            'Marks': ['mean', 'sum'],
            'Question': lambda x: list(x)
        }).reset_index()

        topic_question_combos.columns = [
            'topic', 'question_type', 'total_count', 'years_appeared',
            'years_list', 'avg_marks', 'total_marks', 'questions'
        ]

        # Calculate features
        max_year = self.df['Year'].max()
        total_years = self.df['Year'].nunique()

        topic_question_combos['frequency'] = topic_question_combos['years_appeared'] / total_years
        topic_question_combos['last_year'] = topic_question_combos['years_list'].apply(
            lambda x: max(x) if x else 0
        )
        topic_question_combos['recency'] = max_year - topic_question_combos['last_year']

        # Calculate trend (appeared more in recent 2 years?)
        def calculate_trend(years_list):
            recent_years = [y for y in years_list if y >= max_year - 1]
            older_years = [y for y in years_list if y < max_year - 1]
            if len(older_years) == 0:
                return 2.0  # New pattern
            return len(recent_years) / len(older_years) if older_years else 1.0

        topic_question_combos['trend'] = topic_question_combos['years_list'].apply(calculate_trend)

        # Calculate prediction score (similar to baseline model)
        topic_question_combos['prediction_score'] = (
            topic_question_combos['frequency'] *
            (1 + 0.2 * (topic_question_combos['recency'] / max(topic_question_combos['recency']))) *
            np.sqrt(topic_question_combos['trend'])
        )

        # Normalize scores
        max_score = topic_question_combos['prediction_score'].max()
        if max_score > 0:
            topic_question_combos['prediction_score'] = (
                topic_question_combos['prediction_score'] / max_score
            )

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

        topic_question_combos['confidence'] = topic_question_combos['prediction_score'].apply(get_confidence)

        # Sort by score
        topic_question_combos = topic_question_combos.sort_values('prediction_score', ascending=False)

        self.predictions = topic_question_combos
        print(f"✓ Built predictions for {len(self.predictions)} topic+type combinations")

    def get_important_questions(self,
                               chapter: Optional[str] = None,
                               topic: Optional[str] = None,
                               question_type: Optional[str] = None,
                               top_n: int = 10,
                               min_confidence: str = 'Low') -> List[Dict]:
        """
        Get important questions based on filters

        Args:
            chapter: Filter by chapter name (optional)
            topic: Filter by topic (optional)
            question_type: Filter by question type (optional)
            top_n: Number of results to return
            min_confidence: Minimum confidence level

        Returns:
            List of dictionaries with question details
        """
        filtered = self.predictions.copy()

        # Apply filters
        if chapter:
            # Get topics from this chapter
            chapter_topics = self.df[self.df['Chapter_Name'].str.contains(chapter, case=False, na=False)]['primary_topic'].unique()
            filtered = filtered[filtered['topic'].isin(chapter_topics)]

        if topic:
            filtered = filtered[filtered['topic'].str.contains(topic, case=False, na=False)]

        if question_type:
            filtered = filtered[filtered['question_type'].str.contains(question_type, case=False, na=False)]

        # Filter by confidence
        conf_order = ['Very High', 'High', 'Moderate', 'Low']
        min_conf_idx = conf_order.index(min_confidence)
        allowed_confs = conf_order[:min_conf_idx + 1]
        filtered = filtered[filtered['confidence'].isin(allowed_confs)]

        # Get top N
        results = []
        for idx, row in filtered.head(top_n).iterrows():
            # Sample questions (take up to 3)
            sample_questions = row['questions'][:3] if isinstance(row['questions'], list) else []

            result = {
                'topic': row['topic'],
                'question_type': row['question_type'],
                'probability_score': round(row['prediction_score'], 4),
                'confidence': row['confidence'],
                'frequency': f"{row['years_appeared']}/{self.df['Year'].nunique()} years",
                'years_appeared': row['years_list'],
                'last_appeared': int(row['last_year']),
                'avg_marks': round(row['avg_marks'], 1),
                'total_questions': int(row['total_count']),
                'sample_questions': sample_questions,
                'reasoning': self._generate_reasoning(row)
            }
            results.append(result)

        return results

    def check_question_probability(self,
                                   question_text: Optional[str] = None,
                                   topic: Optional[str] = None,
                                   question_type: Optional[str] = None) -> Dict:
        """
        Check if a specific question/topic is likely to appear

        Args:
            question_text: Question text to analyze (optional)
            topic: Topic name (optional)
            question_type: Question type (optional)

        Returns:
            Dictionary with probability and details
        """
        from src.topic_extraction import TopicExtractor

        # If question text provided, extract topic
        if question_text:
            extractor = TopicExtractor()
            topics = extractor.extract_topics(question_text, "")
            if topics:
                topic = topics[0]

        if not topic:
            return {
                'error': 'Could not extract topic from question',
                'probability': 0.0
            }

        # Find matching predictions
        matches = self.predictions[
            self.predictions['topic'].str.contains(topic, case=False, na=False)
        ]

        if question_type:
            matches = matches[
                matches['question_type'].str.contains(question_type, case=False, na=False)
            ]

        if len(matches) == 0:
            return {
                'topic': topic,
                'question_type': question_type or 'Any',
                'probability': 0.2,
                'confidence': 'Low',
                'message': 'Topic not found in historical data or very rare'
            }

        # Get best match
        best_match = matches.iloc[0]

        return {
            'topic': best_match['topic'],
            'question_type': best_match['question_type'],
            'probability_score': round(best_match['prediction_score'], 4),
            'confidence': best_match['confidence'],
            'frequency': f"{best_match['years_appeared']}/{self.df['Year'].nunique()} years",
            'years_appeared': best_match['years_list'],
            'last_appeared': int(best_match['last_year']),
            'reasoning': self._generate_reasoning(best_match),
            'similar_questions_count': int(best_match['total_count'])
        }

    def get_chapter_summary(self, chapter: str) -> Dict:
        """
        Get complete summary for a chapter

        Args:
            chapter: Chapter name

        Returns:
            Dictionary with chapter analysis
        """
        # Get all topics from this chapter
        chapter_data = self.df[self.df['Chapter_Name'].str.contains(chapter, case=False, na=False)]

        if len(chapter_data) == 0:
            return {'error': 'Chapter not found'}

        topics = chapter_data['primary_topic'].value_counts()

        return {
            'chapter': chapter,
            'total_questions': len(chapter_data),
            'unique_topics': len(topics),
            'top_topics': topics.head(10).to_dict(),
            'important_questions': self.get_important_questions(chapter=chapter, top_n=10)
        }

    def _generate_reasoning(self, row: pd.Series) -> str:
        """Generate human-readable reasoning"""
        reasons = []

        if row['frequency'] >= 0.7:
            reasons.append(f"appears very frequently ({row['years_appeared']}/{self.df['Year'].nunique()} years)")
        elif row['frequency'] >= 0.4:
            reasons.append(f"appears regularly ({row['years_appeared']}/{self.df['Year'].nunique()} years)")

        if row['recency'] >= 2:
            reasons.append(f"hasn't appeared in {int(row['recency'])} years (due for return)")
        elif row['recency'] == 0:
            reasons.append("appeared in most recent year")

        if row['trend'] > 1.2:
            reasons.append("showing increasing trend")

        if not reasons:
            reasons.append(f"appeared {int(row['total_count'])} times historically")

        return "; ".join(reasons)

    def export_to_json(self, output_path: str, top_n: int = 100):
        """
        Export predictions to JSON for RAG

        Args:
            output_path: Path to save JSON
            top_n: Number of top predictions to export
        """
        results = self.get_important_questions(top_n=top_n)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✓ Exported {len(results)} predictions to {output_path}")
