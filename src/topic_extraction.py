"""
Topic extraction from Physics questions
Extracts specific topics/concepts from question text
"""

import pandas as pd
import re
from typing import List, Dict, Set
from collections import defaultdict

class TopicExtractor:
    def __init__(self):
        """Initialize with physics topic keywords"""
        # Physics topic keywords - can be expanded
        self.topic_keywords = {
            # Electric Charges and Fields
            'Coulombs Law': ['coulomb', 'coulombs law', 'electrostatic force'],
            'Electric Field': ['electric field', 'field intensity', 'field lines'],
            'Gauss Theorem': ['gauss', 'gaussian surface', 'gauss theorem'],
            'Electric Dipole': ['dipole', 'dipole moment'],
            'Electric Flux': ['electric flux', 'flux'],

            # Current Electricity
            'Ohms Law': ['ohm', 'resistance', 'resistivity', 'specific resistance'],
            'Kirchhoff Laws': ['kirchhoff', 'junction rule', 'loop rule'],
            'Wheatstone Bridge': ['wheatstone', 'bridge'],
            'Potentiometer': ['potentiometer', 'e.m.f'],
            'Meter Bridge': ['meter bridge'],
            'Ammeter Voltmeter': ['ammeter', 'voltmeter', 'galvanometer'],
            'Current Drift Velocity': ['current', 'drift velocity', 'current density'],
            'Thermistor': ['thermistor'],

            # Electromagnetic
            'Biot Savart Law': ['biot', 'savart'],
            'Ampere Law': ['ampere', 'amperes law'],
            'Magnetic Field': ['magnetic field', 'magnetic intensity'],
            'Bar Magnet': ['bar magnet', 'magnetic dipole'],
            'Faraday Law': ['faraday', 'electromagnetic induction', 'induced emf'],
            'Lenz Law': ['lenz', "lenz's law"],
            'Self Inductance': ['self inductance', 'inductance', 'inductor'],
            'Mutual Inductance': ['mutual inductance'],
            'Eddy Currents': ['eddy current'],
            'AC Circuit': ['alternating current', 'a.c', 'ac circuit'],
            'Transformer': ['transformer'],
            'Choke Coil': ['choke coil'],

            # Optics
            'Reflection': ['reflection', 'mirror', 'spherical mirror'],
            'Refraction': ['refraction', 'refractive index', 'snell'],
            'Total Internal Reflection': ['total internal reflection', 'critical angle'],
            'Lens': ['lens', 'convex lens', 'concave lens', 'lens formula'],
            'Prism': ['prism', 'dispersion', 'deviation'],
            'Optical Instruments': ['microscope', 'telescope'],
            'Optical Fibre': ['optical fibre', 'fiber optics'],
            'Interference': ['interference', 'young', 'double slit'],
            'Diffraction': ['diffraction'],
            'Polarization': ['polarization', 'polaroid', 'malus'],
            'Resolving Power': ['resolving power'],
            'LASER': ['laser'],

            # Modern Physics
            'Photoelectric Effect': ['photoelectric', 'photon', 'work function'],
            'De Broglie': ['de broglie', 'matter wave'],
            'Bohr Model': ['bohr', 'atomic model', 'hydrogen spectrum'],
            'X-rays': ['x-ray', 'x ray'],
            'Radioactivity': ['radioactive', 'radioactivity', 'decay'],
            'Nuclear': ['nuclear', 'nucleus', 'binding energy'],

            # Semiconductor
            'PN Junction': ['p-n junction', 'pn junction', 'diode'],
            'Rectifier': ['rectifier', 'half wave', 'full wave'],
            'Transistor': ['transistor', 'amplifier'],
            'Logic Gates': ['logic gate', 'and gate', 'or gate', 'not gate', 'nand', 'nor'],
            'LED': ['led', 'light emitting diode'],
            'Zener Diode': ['zener', 'voltage regulator'],

            # Electromagnetic Waves
            'EM Spectrum': ['electromagnetic wave', 'em spectrum', 'electromagnetic spectrum'],
            'TV Transmission': ['television', 'antenna', 'transmission'],
        }

        # Question type patterns
        self.derivation_keywords = ['derive', 'establish', 'prove', 'deduce', 'obtain expression']
        self.definition_keywords = ['define', 'what is', 'what are', 'explain']
        self.numerical_keywords = ['calculate', 'find', 'determine', 'compute']

    def clean_question_text(self, text: str) -> str:
        """Remove HTML tags and clean text"""
        if not isinstance(text, str):
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Convert to lowercase
        text = text.lower()
        return text.strip()

    def extract_topics(self, question_text: str, chapter: str) -> List[str]:
        """
        Extract topics from question text

        Args:
            question_text: The question text
            chapter: Chapter name for context

        Returns:
            List of detected topics
        """
        clean_text = self.clean_question_text(question_text)
        detected_topics = []

        # Match against topic keywords
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword.lower() in clean_text:
                    detected_topics.append(topic)
                    break

        # If no specific topic found, use chapter as topic
        if not detected_topics:
            detected_topics.append(chapter)

        return detected_topics

    def classify_question_pattern(self, question_text: str) -> str:
        """
        Classify the question pattern (derivation, definition, numerical, etc.)

        Args:
            question_text: The question text

        Returns:
            Pattern type
        """
        clean_text = self.clean_question_text(question_text)

        # Check for derivation
        if any(keyword in clean_text for keyword in self.derivation_keywords):
            return "Derivation"

        # Check for definition
        if any(keyword in clean_text for keyword in self.definition_keywords):
            return "Definition"

        # Check for numerical
        if any(keyword in clean_text for keyword in self.numerical_keywords):
            return "Numerical"

        return "Conceptual"

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire dataframe to extract topics

        Args:
            df: DataFrame with questions

        Returns:
            DataFrame with added 'extracted_topics' and 'question_pattern' columns
        """
        print("Extracting topics from questions...")

        # Extract topics for each question
        df['extracted_topics'] = df.apply(
            lambda row: self.extract_topics(row['Question'], row['Chapter_Name']),
            axis=1
        )

        # Classify question patterns
        df['question_pattern'] = df['Question'].apply(self.classify_question_pattern)

        # Create a primary topic (first detected topic)
        df['primary_topic'] = df['extracted_topics'].apply(lambda x: x[0] if x else 'Unknown')

        print(f"✓ Extracted topics for {len(df)} questions")
        print(f"  Unique topics found: {df['primary_topic'].nunique()}")

        return df

    def get_topic_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get statistics about topics"""
        topic_stats = df.groupby('primary_topic').agg({
            'Year': ['count', 'nunique'],
            'Marks': 'sum',
            'Question type': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
        }).reset_index()

        topic_stats.columns = ['topic', 'total_questions', 'years_appeared',
                               'total_marks', 'most_common_type']

        return topic_stats.sort_values('total_questions', ascending=False)
