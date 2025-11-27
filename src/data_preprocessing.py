"""
Data preprocessing module for board exam data
Handles loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os

class ExamDataPreprocessor:
    def __init__(self, data_path: str):
        """
        Initialize with path to raw exam data CSV
        Expected columns: Year, Subject, Chapter_Name, Question type, Marks
        """
        self.data_path = data_path
        self.df = None
        self.topic_matrix = None
        self.features = None

    def load_data(self) -> pd.DataFrame:
        """Load and validate raw exam data"""
        self.df = pd.read_csv(self.data_path)

        # Validate required columns
        required_cols = ['Year', 'Subject', 'Chapter_Name', 'Question type', 'Marks']
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Data cleaning
        self.df['Chapter_Name'] = self.df['Chapter_Name'].str.strip()
        self.df['Year'] = pd.to_numeric(self.df['Year'], errors='coerce')

        # Remove any rows with invalid years
        self.df = self.df.dropna(subset=['Year'])
        self.df['Year'] = self.df['Year'].astype(int)

        print(f"Loaded {len(self.df)} questions from {self.df['Year'].nunique()} years")
        print(f"Years: {sorted(self.df['Year'].unique())}")
        print(f"Chapters: {self.df['Chapter_Name'].nunique()}")
        return self.df

    def create_topic_matrix(self) -> pd.DataFrame:
        """
        Create binary matrix: rows=years, columns=chapters/topics
        1 if topic appeared that year, 0 otherwise
        """
        # Group by year and chapter, sum the marks
        topic_year_marks = self.df.groupby(['Year', 'Chapter_Name'])['Marks'].sum().reset_index()

        # Create pivot table
        self.topic_matrix = topic_year_marks.pivot(
            index='Year',
            columns='Chapter_Name',
            values='Marks'
        ).fillna(0)

        # Convert to binary (1 if chapter appeared, 0 otherwise)
        self.topic_matrix = (self.topic_matrix > 0).astype(int)

        print(f"\nCreated topic matrix: {self.topic_matrix.shape}")
        print(f"Years: {list(self.topic_matrix.index)}")
        print(f"Topics: {len(self.topic_matrix.columns)}")
        return self.topic_matrix

    def engineer_features(self) -> pd.DataFrame:
        """
        Create features for each chapter/topic:
        - frequency: How often it appears across years
        - recency: Years since last appearance
        - average_marks: Average marks when it appears
        - trend: Increasing/decreasing over time
        - total_appearances: Total number of years appeared
        """
        features = []

        max_year = self.df['Year'].max()
        total_years = self.df['Year'].nunique()

        for chapter in self.df['Chapter_Name'].unique():
            chapter_data = self.df[self.df['Chapter_Name'] == chapter]

            # Basic statistics
            years_appeared = chapter_data['Year'].unique()
            frequency = len(years_appeared) / total_years
            last_year = years_appeared.max()
            recency = max_year - last_year
            avg_marks = chapter_data['Marks'].mean()
            total_marks = chapter_data['Marks'].sum()

            # Trend calculation (simple: appeared more in recent years?)
            recent_years = chapter_data[chapter_data['Year'] >= max_year - 2]
            older_years = chapter_data[chapter_data['Year'] < max_year - 2]

            if len(older_years) > 0:
                trend = len(recent_years) / len(older_years)
            else:
                trend = 1.0

            features.append({
                'chapter': chapter,
                'frequency': frequency,
                'recency': recency,
                'avg_marks': avg_marks,
                'total_marks': total_marks,
                'trend': trend,
                'total_appearances': len(years_appeared),
                'last_year': last_year
            })

        self.features = pd.DataFrame(features)
        print(f"\nEngineered features for {len(features)} chapters")
        return self.features

    def get_question_type_distribution(self) -> pd.DataFrame:
        """Get distribution of question types over the years"""
        type_dist = self.df.groupby(['Year', 'Question type']).size().reset_index(name='count')
        return type_dist

    def save_processed_data(self, output_dir: str):
        """Save all processed data"""
        os.makedirs(output_dir, exist_ok=True)

        if self.topic_matrix is not None:
            self.topic_matrix.to_csv(f"{output_dir}/topic_matrix.csv")
            print(f"✓ Saved topic matrix")

        if self.features is not None:
            self.features.to_csv(f"{output_dir}/features.csv", index=False)
            print(f"✓ Saved features")

        print(f"\nAll processed data saved to {output_dir}")

    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the dataset"""
        stats = {
            'total_questions': len(self.df),
            'years': sorted(self.df['Year'].unique()),
            'num_chapters': self.df['Chapter_Name'].nunique(),
            'avg_questions_per_year': len(self.df) / self.df['Year'].nunique(),
            'question_types': self.df['Question type'].value_counts().to_dict()
        }
        return stats
