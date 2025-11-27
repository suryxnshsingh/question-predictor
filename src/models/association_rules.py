"""
Association rules mining for topic co-occurrence patterns
Finds patterns like "If chapter A appears, then chapter B likely appears too"
"""

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class AssociationRulesPredictor:
    def __init__(self, exam_df: pd.DataFrame):
        """
        Initialize association rules predictor

        Args:
            exam_df: DataFrame with columns [Year, Chapter_Name, ...]
        """
        self.df = exam_df
        self.rules = None
        self.topic_scores = {}

    def prepare_transactions(self) -> pd.DataFrame:
        """
        Convert exam data to transaction format
        Each year is a transaction containing the chapters that appeared

        Returns:
            One-hot encoded DataFrame
        """
        # Group chapters by year (each year is a transaction)
        transactions = self.df.groupby('Year')['Chapter_Name'].apply(list).tolist()

        # One-hot encode
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        return pd.DataFrame(te_array, columns=te.columns_)

    def mine_rules(self, min_support: float = 0.2,
                   min_confidence: float = 0.5) -> pd.DataFrame:
        """
        Find association rules between chapters

        Args:
            min_support: Minimum support threshold (0-1)
            min_confidence: Minimum confidence threshold (0-1)

        Returns:
            DataFrame of association rules
        """
        # Get transaction data
        df_encoded = self.prepare_transactions()

        # Find frequent itemsets
        try:
            frequent_itemsets = apriori(
                df_encoded,
                min_support=min_support,
                use_colnames=True
            )

            if len(frequent_itemsets) == 0:
                print(f"No frequent itemsets found with min_support={min_support}")
                print("Trying with lower threshold...")
                # Try with lower support
                frequent_itemsets = apriori(
                    df_encoded,
                    min_support=0.1,
                    use_colnames=True
                )

            if len(frequent_itemsets) == 0:
                print("Still no frequent itemsets found.")
                return pd.DataFrame()

            # Generate rules
            self.rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_confidence
            )

            # Filter for strong rules (high lift)
            if len(self.rules) > 0:
                self.rules = self.rules[self.rules['lift'] > 1.0]

            print(f"Found {len(self.rules)} association rules")
            return self.rules

        except Exception as e:
            print(f"Error mining rules: {e}")
            return pd.DataFrame()

    def score_topics(self) -> pd.Series:
        """
        Score chapters based on association rules
        Chapters in consequents of strong rules get higher scores

        Returns:
            Series with chapter scores
        """
        if self.rules is None or len(self.rules) == 0:
            # Fallback: return uniform scores for all topics
            all_topics = self.df['Chapter_Name'].unique()
            return pd.Series({topic: 0.5 for topic in all_topics})

        topic_scores = {}

        # Get all unique topics
        all_topics = set()
        for _, rule in self.rules.iterrows():
            all_topics.update(rule['antecedents'])
            all_topics.update(rule['consequents'])

        # Score based on rule strength
        for topic in all_topics:
            score = 0
            for _, rule in self.rules.iterrows():
                if topic in rule['consequents']:
                    # Higher weight if topic is the consequence
                    score += rule['confidence'] * rule['lift']
                elif topic in rule['antecedents']:
                    # Lower weight if topic is just in antecedent
                    score += rule['confidence'] * 0.3

            topic_scores[topic] = score

        # Add remaining topics with zero score
        for topic in self.df['Chapter_Name'].unique():
            if topic not in topic_scores:
                topic_scores[topic] = 0.1  # Small baseline score

        # Normalize
        scores_series = pd.Series(topic_scores)
        if scores_series.max() > 0:
            scores_series = scores_series / scores_series.max()

        return scores_series

    def get_predictions(self, top_n: int = 25) -> pd.DataFrame:
        """
        Get top chapters based on association rules

        Args:
            top_n: Number of top predictions to return

        Returns:
            DataFrame with chapters and association scores
        """
        self.mine_rules()
        scores = self.score_topics()

        if len(scores) == 0:
            return pd.DataFrame()

        top = scores.nlargest(min(top_n, len(scores))).to_frame()
        top.columns = ['association_score']
        top['association_score'] = top['association_score'].round(4)
        return top

    def get_related_chapters(self, chapter: str, min_confidence: float = 0.6) -> list:
        """
        Find chapters that are strongly associated with a given chapter

        Args:
            chapter: Chapter name
            min_confidence: Minimum confidence threshold

        Returns:
            List of related chapters with their confidence scores
        """
        if self.rules is None or len(self.rules) == 0:
            return []

        related = []
        for _, rule in self.rules.iterrows():
            if chapter in rule['antecedents'] and rule['confidence'] >= min_confidence:
                for consequent in rule['consequents']:
                    related.append({
                        'chapter': consequent,
                        'confidence': round(rule['confidence'], 3),
                        'lift': round(rule['lift'], 3)
                    })

        return sorted(related, key=lambda x: x['confidence'], reverse=True)

    def get_rule_summary(self) -> dict:
        """Get summary of association rules"""
        if self.rules is None or len(self.rules) == 0:
            return {
                'total_rules': 0,
                'message': 'No rules found'
            }

        return {
            'total_rules': len(self.rules),
            'avg_confidence': round(self.rules['confidence'].mean(), 3),
            'avg_lift': round(self.rules['lift'].mean(), 3),
            'strongest_rule': self._get_strongest_rule()
        }

    def _get_strongest_rule(self) -> dict:
        """Get the strongest association rule"""
        if len(self.rules) == 0:
            return {}

        strongest = self.rules.nlargest(1, 'confidence').iloc[0]
        return {
            'antecedents': list(strongest['antecedents']),
            'consequents': list(strongest['consequents']),
            'confidence': round(strongest['confidence'], 3),
            'lift': round(strongest['lift'], 3)
        }
