"""
Unified Prediction Service - Supports Multiple Subjects

Usage:
    from src.prediction_service import PredictionService

    # Initialize for a subject
    service = PredictionService(subject='physics')

    # Get predictions
    top_topics = service.get_top_predictions(top_n=10)
    chapter_topics = service.get_chapter_topics("Electric Charges", top_n=5)
    topic_prob = service.check_topic_probability("Ohms Law")
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
import importlib
import re
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.subjects import get_subject_config, list_subjects


def clean_question_text(text: str) -> str:
    """Remove HTML tags and clean question text for display"""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove Q. prefix
    text = re.sub(r'^Q\.\s*', '', text.strip())
    # Clean LaTeX-style formatting
    text = re.sub(r'\\\(', '', text)
    text = re.sub(r'\\\)', '', text)
    text = re.sub(r'\\', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class TopicExtractor:
    """Subject-agnostic topic extractor"""

    def __init__(self, topic_keywords: Dict[str, List[str]]):
        self.topic_keywords = topic_keywords
        self.derivation_keywords = ['derive', 'establish', 'prove', 'deduce', 'obtain expression']
        self.definition_keywords = ['define', 'what is', 'what are', 'explain']
        self.numerical_keywords = ['calculate', 'find', 'determine', 'compute']

    def clean_text(self, text: str) -> str:
        """Remove HTML tags and clean text"""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()

    def extract_topics(self, question_text: str, chapter: str) -> List[str]:
        """Extract topics from question text"""
        clean_text = self.clean_text(question_text)
        detected_topics = []

        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword.lower() in clean_text:
                    detected_topics.append(topic)
                    break

        if not detected_topics:
            detected_topics.append(chapter)

        return detected_topics

    def classify_pattern(self, question_text: str) -> str:
        """Classify question pattern"""
        clean_text = self.clean_text(question_text)

        if any(kw in clean_text for kw in self.derivation_keywords):
            return "Derivation"
        if any(kw in clean_text for kw in self.definition_keywords):
            return "Definition"
        if any(kw in clean_text for kw in self.numerical_keywords):
            return "Numerical"
        return "Conceptual"

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process dataframe to extract topics"""
        df = df.copy()
        df['extracted_topics'] = df.apply(
            lambda row: self.extract_topics(row['Question'], row['Chapter_Name']),
            axis=1
        )
        df['question_pattern'] = df['Question'].apply(self.classify_pattern)
        df['primary_topic'] = df['extracted_topics'].apply(lambda x: x[0] if x else 'Unknown')
        return df


class PredictionService:
    """
    Main prediction service for any subject
    """

    def __init__(self, subject: str = 'physics', auto_load: bool = True):
        """
        Initialize prediction service for a subject

        Args:
            subject: Subject key (e.g., 'physics', 'chemistry')
            auto_load: If True, automatically load processed data if exists
        """
        self.subject = subject
        self.config = get_subject_config(subject)
        self.df = None
        self.topic_predictions = None
        self.qtype_history = None
        self.extractor = None

        # Load topic keywords
        self._load_topic_keywords()

        if auto_load:
            self._load_or_process_data()

    def _load_topic_keywords(self):
        """Load topic keywords for this subject"""
        module = importlib.import_module(self.config['topics_module'])
        topic_keywords = getattr(module, self.config['topics_var'])
        self.extractor = TopicExtractor(topic_keywords)

    def _load_or_process_data(self):
        """Load processed data or process raw data"""
        processed_path = self.config['processed_data']

        if os.path.exists(processed_path):
            print(f"Loading processed {self.config['name']} data...")
            self.df = pd.read_csv(processed_path)
        else:
            print(f"Processing raw {self.config['name']} data...")
            self.process_raw_data()

        self._build_predictions()

    def process_raw_data(self):
        """Process raw data and save"""
        raw_path = self.config['train_data']
        processed_path = self.config['processed_data']

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data not found: {raw_path}")

        df = pd.read_csv(raw_path)
        self.df = self.extractor.process_dataframe(df)

        # Ensure directory exists
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        self.df.to_csv(processed_path, index=False)
        print(f"✓ Saved processed data to {processed_path}")

    def _build_predictions(self):
        """Build topic predictions and question type history"""
        print(f"Building {self.config['name']} predictions...")
        self._predict_topics()
        self._build_qtype_history()
        print("✓ Service ready")

    def _predict_topics(self):
        """Predict topics using optimized baseline model"""
        max_year = self.df['Year'].max()
        total_years = self.df['Year'].nunique()

        features = []
        for topic in self.df['primary_topic'].unique():
            topic_data = self.df[self.df['primary_topic'] == topic]

            years_appeared = topic_data['Year'].unique()
            frequency = len(years_appeared) / total_years
            last_year = years_appeared.max()
            recency = max_year - last_year

            # Trend calculation
            recent_years = topic_data[topic_data['Year'] >= max_year - 1]
            older_years = topic_data[topic_data['Year'] < max_year - 1]
            trend = len(recent_years) / len(older_years) if len(older_years) > 0 else 2.0

            # Optimized scoring (handles high-frequency topics correctly)
            if frequency >= 0.75:
                score = frequency * (1.0 + 0.1 * (1 / (recency + 1)))
            else:
                max_recency = total_years
                norm_recency = recency / max_recency if max_recency > 0 else 0
                score = frequency * (1 + 0.4 * norm_recency) * np.sqrt(max(trend, 0.5))

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

        # Normalize
        max_score = self.topic_predictions['probability'].max()
        if max_score > 0:
            self.topic_predictions['probability'] = self.topic_predictions['probability'] / max_score

        self.topic_predictions = self.topic_predictions.sort_values('probability', ascending=False)

    def _build_qtype_history(self):
        """Build question type history for each topic"""
        self.qtype_history = {}

        for topic in self.df['primary_topic'].unique():
            topic_data = self.df[self.df['primary_topic'] == topic]
            qtype_counts = topic_data['Question type'].value_counts()

            qtype_years = {}
            for qtype in qtype_counts.index:
                years = topic_data[topic_data['Question type'] == qtype]['Year'].unique()
                qtype_years[qtype] = sorted(years.tolist())

            avg_marks = topic_data.groupby('Question type')['Marks'].mean().to_dict()

            # Get ALL questions grouped by type (for predictions)
            questions_by_type = {}
            for qtype in qtype_counts.index:
                type_questions = topic_data[topic_data['Question type'] == qtype].sort_values('Year', ascending=False)
                questions_by_type[qtype] = []
                for _, q_row in type_questions.iterrows():  # ALL questions, not just head(3)
                    questions_by_type[qtype].append({
                        'question': clean_question_text(q_row['Question']),
                        'year': int(q_row['Year']),
                        'marks': int(q_row['Marks'])
                    })

            self.qtype_history[topic] = {
                'all_types': qtype_counts.to_dict(),
                'most_common_type': qtype_counts.index[0],
                'most_common_count': int(qtype_counts.iloc[0]),
                'type_years': qtype_years,
                'avg_marks_per_type': {k: round(v, 1) for k, v in avg_marks.items()},
                'questions_by_type': questions_by_type
            }

    def get_top_predictions(self, top_n: int = 25, min_probability: float = 0.0) -> List[Dict]:
        """Get top N topic predictions with question type history"""
        filtered = self.topic_predictions[self.topic_predictions['probability'] >= min_probability]

        results = []
        for idx, row in filtered.head(top_n).iterrows():
            topic = row['topic']
            qtype_info = self.qtype_history.get(topic, {})

            prob = row['probability']
            if prob > 0.7:
                confidence = 'Very High'
            elif prob > 0.5:
                confidence = 'High'
            elif prob > 0.3:
                confidence = 'Moderate'
            else:
                confidence = 'Low'

            reasons = []
            total_years = self.df['Year'].nunique()
            if row['frequency'] >= 0.7:
                reasons.append(f"appears very frequently ({row['years_appeared']}/{total_years} years)")
            elif row['frequency'] >= 0.4:
                reasons.append(f"appears regularly ({row['years_appeared']}/{total_years} years)")

            if row['recency'] >= 2:
                reasons.append(f"hasn't appeared in {int(row['recency'])} years (due)")
            elif row['recency'] == 0:
                reasons.append("appeared in most recent year")

            if row['trend'] > 1.2:
                reasons.append("showing increasing trend")

            result = {
                'rank': len(results) + 1,
                'topic': topic,
                'probability_score': round(prob, 4),
                'confidence': confidence,
                'frequency': f"{row['years_appeared']}/{total_years} years",
                'last_appeared': int(row['last_year']),
                'total_questions_historical': int(row['total_questions']),
                'total_marks_historical': int(row['total_marks']),
                'reasoning': "; ".join(reasons) if reasons else "appeared historically",
                'historical_question_types': list(qtype_info.get('all_types', {}).keys()),
                'most_common_question_type': qtype_info.get('most_common_type', 'Unknown'),
                'question_type_distribution': qtype_info.get('all_types', {}),
                'question_type_years': qtype_info.get('type_years', {}),
                'avg_marks_per_type': qtype_info.get('avg_marks_per_type', {}),
                'questions_by_type': qtype_info.get('questions_by_type', {})
            }
            results.append(result)

        return results

    def check_topic_probability(self, topic_query: str) -> Optional[Dict]:
        """Check probability for a specific topic (fuzzy match)"""
        matches = self.topic_predictions[
            self.topic_predictions['topic'].str.contains(topic_query, case=False, na=False)
        ]

        if len(matches) == 0:
            return None

        top_match = matches.iloc[0]
        full_predictions = self.get_top_predictions(top_n=len(self.topic_predictions))

        for pred in full_predictions:
            if pred['topic'] == top_match['topic']:
                return pred
        return None

    def get_chapter_topics(self, chapter_name: str, top_n: int = 10) -> List[Dict]:
        """Get top topics from a specific chapter"""
        chapter_data = self.df[
            self.df['Chapter_Name'].str.contains(chapter_name, case=False, na=False)
        ]

        if len(chapter_data) == 0:
            return []

        chapter_topics = set(chapter_data['primary_topic'].unique())
        all_preds = self.get_top_predictions(top_n=100)
        chapter_preds = [p for p in all_preds if p['topic'] in chapter_topics]

        return chapter_preds[:top_n]

    def export_for_rag(self, output_path: str = None, top_n: int = 50):
        """Export predictions in JSON format for RAG system"""
        if output_path is None:
            output_path = self.config['output_file']

        predictions = self.get_top_predictions(top_n=top_n)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)

        print(f"✓ Exported {len(predictions)} predictions to {output_path}")

    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        return {
            'subject': self.config['name'],
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

    def validate_on_holdout(self, top_n: int = 25) -> Dict:
        """Validate predictions against holdout data"""
        holdout_path = self.config['holdout_data']

        if not os.path.exists(holdout_path):
            return {'error': f'Holdout data not found: {holdout_path}'}

        holdout_df = pd.read_csv(holdout_path)
        holdout_df = self.extractor.process_dataframe(holdout_df)

        actual_topics = set(holdout_df['primary_topic'].unique())
        predicted_topics = set(self.topic_predictions.head(top_n)['topic'])

        hits = len(predicted_topics & actual_topics)
        precision = hits / len(predicted_topics) if predicted_topics else 0
        recall = hits / len(actual_topics) if actual_topics else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Question-level coverage
        holdout_df['predicted'] = holdout_df['primary_topic'].isin(predicted_topics)
        question_coverage = holdout_df['predicted'].sum() / len(holdout_df)

        return {
            'top_n': top_n,
            'predicted_topics': len(predicted_topics),
            'actual_topics': len(actual_topics),
            'matched_topics': hits,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'question_coverage': round(question_coverage, 4),
            'total_holdout_questions': len(holdout_df)
        }


# Convenience function
def get_service(subject: str = 'physics') -> PredictionService:
    """Get prediction service for a subject"""
    return PredictionService(subject=subject)
