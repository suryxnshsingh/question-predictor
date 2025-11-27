"""
Enhanced Topic Extraction with Sub-Topic Granularity
Breaks down broad chapters into specific sub-topics
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List

class EnhancedTopicExtractor:
    """Extract specific sub-topics from questions"""

    def __init__(self):
        """Initialize with sub-topic patterns"""
        self.subtopic_patterns = {
            'Ray Optics and Optical Instruments': {
                'Mirror_Formula': [r'mirror\s+formula', r'focal\s+length.*mirror', r'1/v.*1/u', r'concave\s+mirror', r'convex\s+mirror', r'spherical\s+mirror'],
                'Lens_Formula': [r'lens\s+formula', r'thin\s+lens', r'convex\s+lens', r'concave\s+lens', r'focal\s+length.*lens'],
                'Total_Internal_Reflection': [r'total\s+internal', r'critical\s+angle', r'\bTIR\b', r'optical\s+fib'],
                'Telescope': [r'telescope', r'astronomical', r'galilean', r'terrestrial'],
                'Microscope': [r'microscope', r'magnifying\s+power', r'compound\s+microscope'],
                'Prism': [r'prism', r'angle\s+of\s+deviation', r'dispersive\s+power', r'dispersion'],
                'Refraction': [r'refraction', r'refractive\s+index', r'snell'],
                'Reflection': [r'reflection', r'law.*reflection']
            },

            'Current Electricity': {
                'Ohms_Law': [r'ohm', r'V\s*=\s*IR', r'resistance.*current', r'specific\s+resistance', r'resistivity'],
                'Kirchhoff': [r'kirchhoff', r'junction', r'loop\s+law', r'kvl', r'kcl'],
                'Wheatstone': [r'wheatstone', r'meter\s+bridge', r'balanced\s+bridge'],
                'Potentiometer': [r'potentiometer', r'emf\s+comparison', r'null\s+point', r'e\.m\.f'],
                'Drift_Velocity': [r'drift\s+velocity', r'current\s+density', r'mobility'],
                'Ammeter_Voltmeter': [r'ammeter', r'voltmeter', r'galvanometer', r'measuring.*current']
            },

            'Electrostatic Potential and Capacitance': {
                'Electric_Potential': [r'electric\s+potential', r'potential\s+difference', r'voltage'],
                'Capacitance': [r'capacitance', r'capacitor', r'dielectric'],
                'Van_de_Graaff': [r'van.*graaff', r'electrostatic\s+generator']
            },

            'Electric Charges and Fields': {
                'Coulombs_Law': [r'coulomb', r'electrostatic\s+force'],
                'Electric_Field': [r'electric\s+field', r'field\s+intensity', r'field\s+lines'],
                'Gauss_Theorem': [r'gauss', r'gaussian\s+surface', r'electric\s+flux'],
                'Electric_Dipole': [r'dipole', r'dipole\s+moment']
            },

            'Electromagnetic Induction': {
                'Faraday_Law': [r'faraday', r'induced\s+emf', r'electromagnetic\s+induction'],
                'Lenz_Law': [r'lenz', r"lenz's\s+law"],
                'Self_Inductance': [r'self\s+inductance', r'inductance', r'inductor', r'solenoid.*inductance'],
                'Mutual_Inductance': [r'mutual\s+inductance'],
                'Eddy_Currents': [r'eddy\s+current']
            },

            'Alternating Current': {
                'AC_Circuit': [r'alternating\s+current', r'a\.c', r'ac\s+circuit', r'rms\s+value', r'impedance'],
                'Transformer': [r'transformer', r'step\s+up', r'step\s+down'],
                'Choke_Coil': [r'choke\s+coil', r'inductive\s+reactance']
            },

            'Moving Charges and Magnetism': {
                'Biot_Savart': [r'biot', r'savart'],
                'Ampere_Law': [r'ampere', r"ampere's\s+law"],
                'Magnetic_Field': [r'magnetic\s+field', r'magnetic\s+intensity', r'magnetic\s+force'],
                'Bar_Magnet': [r'bar\s+magnet', r'magnetic\s+dipole']
            },

            'Dual Nature of Radiation and Matter': {
                'Photoelectric_Effect': [r'photoelectric', r'photon', r'work\s+function', r'einstein.*photoelectric'],
                'De_Broglie': [r'de\s+broglie', r'matter\s+wave', r'wavelength.*particle']
            },

            'Atoms': {
                'Bohr_Model': [r'bohr', r'atomic\s+model', r'hydrogen\s+spectrum', r'energy\s+level'],
                'X_rays': [r'x-ray', r'x\s+ray']
            },

            'Wave Optics': {
                'Interference': [r'interference', r'young', r'double\s+slit', r'fringe'],
                'Diffraction': [r'diffraction', r'single\s+slit'],
                'Polarization': [r'polarization', r'polaroid', r'malus'],
                'Resolving_Power': [r'resolving\s+power']
            },

            'Semiconductor Electronics: Materials, Devices and Simple Circuits': {
                'PN_Junction': [r'p-n\s+junction', r'pn\s+junction', r'diode', r'forward\s+bias', r'reverse\s+bias'],
                'Rectifier': [r'rectifier', r'half\s+wave', r'full\s+wave'],
                'Transistor': [r'transistor', r'amplifier', r'current\s+gain'],
                'Logic_Gates': [r'logic\s+gate', r'and\s+gate', r'or\s+gate', r'not\s+gate', r'nand', r'nor'],
                'LED': [r'led', r'light\s+emitting\s+diode'],
                'Zener': [r'zener', r'voltage\s+regulator']
            }
        }

    def extract_subtopic(self, question: str, main_topic: str) -> str:
        """
        Extract specific sub-topic from question text

        Args:
            question: Question text
            main_topic: Main chapter/topic

        Returns:
            Sub-topic string (e.g., "Ray Optics/Mirror_Formula")
        """
        if not isinstance(question, str):
            return f"{main_topic}/General"

        # Clean question text
        question_clean = re.sub(r'<[^>]+>', ' ', question).lower()

        # Check if main topic has subtopic patterns
        if main_topic not in self.subtopic_patterns:
            return main_topic  # Return as-is if no patterns defined

        # Try to match subtopic patterns
        for subtopic, patterns in self.subtopic_patterns[main_topic].items():
            for pattern in patterns:
                if re.search(pattern, question_clean, re.IGNORECASE):
                    return f"{main_topic}/{subtopic}"

        # If no match, return as general
        return f"{main_topic}/General"

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire dataframe to add enhanced topics

        Args:
            df: DataFrame with 'Question' and 'primary_topic' columns

        Returns:
            DataFrame with added 'enhanced_topic' column
        """
        print("Extracting enhanced sub-topics...")

        df['enhanced_topic'] = df.apply(
            lambda row: self.extract_subtopic(row['Question'], row['primary_topic']),
            axis=1
        )

        # Count enhancement
        unique_main = df['primary_topic'].nunique()
        unique_enhanced = df['enhanced_topic'].nunique()

        print(f"✓ Enhanced from {unique_main} topics → {unique_enhanced} sub-topics")
        print(f"  Granularity increase: {(unique_enhanced/unique_main - 1)*100:.1f}%")

        return df


class HistoricalPatternAnalyzer:
    """Add historical pattern features"""

    def __init__(self, df: pd.DataFrame, current_year: int = 2025):
        """
        Args:
            df: DataFrame with enhanced topics
            current_year: Year to predict for
        """
        self.df = df
        self.current_year = current_year
        self.pattern_features = None

    def analyze_patterns(self) -> pd.DataFrame:
        """Analyze historical patterns for each enhanced topic"""
        print("\nAnalyzing historical patterns...")

        patterns = []

        for topic in self.df['enhanced_topic'].unique():
            topic_data = self.df[self.df['enhanced_topic'] == topic]
            years_appeared = sorted(topic_data['Year'].unique())

            if len(years_appeared) == 0:
                continue

            # Basic stats
            frequency = len(years_appeared) / self.df['Year'].nunique()
            last_seen = years_appeared[-1]
            years_since_last = self.current_year - last_seen
            appearance_count = len(years_appeared)

            # Calculate gaps
            if len(years_appeared) > 1:
                gaps = [years_appeared[i+1] - years_appeared[i]
                       for i in range(len(years_appeared)-1)]
                avg_gap = np.mean(gaps)
                std_gap = np.std(gaps)
                max_gap = max(gaps)
                min_gap = min(gaps)

                # Is it overdue?
                is_overdue = years_since_last > avg_gap

                # Cycle position (modulo average gap)
                cycle_position = years_since_last % avg_gap if avg_gap > 0 else 0

                # Regularity score (lower std = more regular)
                regularity = 1 / (std_gap + 0.5)  # Add 0.5 to avoid division by zero
            else:
                avg_gap = 0
                std_gap = 0
                max_gap = 0
                min_gap = 0
                is_overdue = False
                cycle_position = 0
                regularity = 0

            # Recent trend (appeared more in recent years?)
            max_year = self.df['Year'].max()
            recent_years = topic_data[topic_data['Year'] >= max_year - 1]
            older_years = topic_data[topic_data['Year'] < max_year - 1]

            if len(older_years) > 0:
                trend = len(recent_years) / len(older_years)
            else:
                trend = 2.0 if len(recent_years) > 0 else 0.0

            patterns.append({
                'enhanced_topic': topic,
                'frequency': frequency,
                'last_seen': last_seen,
                'years_since_last': years_since_last,
                'appearance_count': appearance_count,
                'avg_gap': avg_gap,
                'std_gap': std_gap,
                'max_gap': max_gap,
                'min_gap': min_gap,
                'is_overdue': is_overdue,
                'cycle_position': cycle_position,
                'regularity': regularity,
                'trend': trend,
                'years_appeared': years_appeared
            })

        self.pattern_features = pd.DataFrame(patterns)
        print(f"✓ Analyzed patterns for {len(patterns)} enhanced topics")

        # Show some interesting patterns
        print("\nMost Regular Topics (predictable cycles):")
        regular = self.pattern_features.nlargest(5, 'regularity')[['enhanced_topic', 'avg_gap', 'std_gap', 'regularity']]
        for _, row in regular.iterrows():
            print(f"  • {row['enhanced_topic'][:50]:50} Gap: {row['avg_gap']:.1f}±{row['std_gap']:.1f}")

        print("\nOverdue Topics (likely to appear):")
        overdue = self.pattern_features[self.pattern_features['is_overdue']].nlargest(5, 'years_since_last')[
            ['enhanced_topic', 'years_since_last', 'avg_gap']
        ]
        for _, row in overdue.iterrows():
            print(f"  • {row['enhanced_topic'][:50]:50} Last: {int(row['years_since_last'])}y ago, Avg gap: {row['avg_gap']:.1f}y")

        return self.pattern_features
