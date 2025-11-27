"""
Validation of topic-level predictions against 2025 holdout data
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.topic_extraction import TopicExtractor
from src.topic_predictor_service import TopicPredictorService


def validate_topic_predictions(holdout_path: str, service: TopicPredictorService):
    """
    Validate topic predictions against 2025 holdout data

    Args:
        holdout_path: Path to 2025 holdout CSV
        service: Initialized TopicPredictorService

    Returns:
        Dictionary with validation metrics
    """
    print("\n" + "="*70)
    print("VALIDATING TOPIC-LEVEL PREDICTIONS AGAINST 2025 DATA")
    print("="*70)

    # Load and process 2025 data
    holdout_df = pd.read_csv(holdout_path)
    extractor = TopicExtractor()
    holdout_df = extractor.process_dataframe(holdout_df)

    # Get actual topic+type combinations in 2025
    actual_combos = set()
    for _, row in holdout_df.iterrows():
        combo = (row['primary_topic'], row['Question type'])
        actual_combos.add(combo)

    print(f"\nActual in 2025:")
    print(f"  Total questions: {len(holdout_df)}")
    print(f"  Unique topic+type combinations: {len(actual_combos)}")

    # Get top N predictions
    results = {}
    for top_n in [10, 20, 30, 50, 75]:
        predicted = service.get_important_questions(top_n=top_n)
        predicted_combos = set()
        for pred in predicted:
            combo = (pred['topic'], pred['question_type'])
            predicted_combos.add(combo)

        # Calculate metrics
        hits = len(predicted_combos & actual_combos)
        precision = hits / len(predicted_combos) if predicted_combos else 0
        recall = hits / len(actual_combos) if actual_combos else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results[f'top_{top_n}'] = {
            'hits': hits,
            'predicted_count': len(predicted_combos),
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        print(f"\n--- Top {top_n} Predictions ---")
        print(f"Precision@{top_n}: {precision:.2%} ({hits}/{len(predicted_combos)} predicted appeared)")
        print(f"Recall@{top_n}:    {recall:.2%} ({hits}/{len(actual_combos)} actual were predicted)")
        print(f"F1 Score:      {f1:.2%}")

    # Detailed breakdown
    print("\n" + "="*70)
    print("DETAILED BREAKDOWN (Top 50)")
    print("="*70)

    top_50 = service.get_important_questions(top_n=50)
    predicted_combos_50 = {(p['topic'], p['question_type']) for p in top_50}

    hits = predicted_combos_50 & actual_combos
    misses = actual_combos - predicted_combos_50
    false_positives = predicted_combos_50 - actual_combos

    print(f"\n✓ CORRECTLY PREDICTED ({len(hits)}):")
    for topic, q_type in sorted(hits)[:15]:  # Show first 15
        # Find in predictions
        for pred in top_50:
            if pred['topic'] == topic and pred['question_type'] == q_type:
                print(f"  • {topic[:35]:35} [{q_type[:15]:15}] Score: {pred['probability_score']:.3f}")
                break

    if len(hits) > 15:
        print(f"  ... and {len(hits) - 15} more")

    if misses:
        print(f"\n✗ MISSED ({len(misses)}):")
        for topic, q_type in sorted(misses)[:10]:  # Show first 10
            # Check if topic exists but different question type
            prob = service.check_question_probability(topic=topic, question_type=q_type)
            score = prob.get('probability_score', prob.get('probability', 0))
            print(f"  • {topic[:35]:35} [{q_type[:15]:15}] Score: {score:.3f}")

        if len(misses) > 10:
            print(f"  ... and {len(misses) - 10} more")

    # Topic-level accuracy (ignoring question type)
    print("\n" + "="*70)
    print("TOPIC-LEVEL ACCURACY (Ignoring Question Type)")
    print("="*70)

    actual_topics = set(holdout_df['primary_topic'].unique())
    predicted_topics = {p['topic'] for p in top_50}

    topic_hits = len(predicted_topics & actual_topics)
    topic_precision = topic_hits / len(predicted_topics) if predicted_topics else 0
    topic_recall = topic_hits / len(actual_topics) if actual_topics else 0
    topic_f1 = 2 * (topic_precision * topic_recall) / (topic_precision + topic_recall) if (topic_precision + topic_recall) > 0 else 0

    print(f"Topics in 2025: {len(actual_topics)}")
    print(f"Topics predicted (top 50): {len(predicted_topics)}")
    print(f"Precision: {topic_precision:.2%}")
    print(f"Recall: {topic_recall:.2%}")
    print(f"F1 Score: {topic_f1:.2%}")

    return results


if __name__ == "__main__":
    # Initialize service
    print("Initializing prediction service...")
    service = TopicPredictorService("data/processed/physics_with_topics.csv")

    # Run validation
    results = validate_topic_predictions("data/raw/physics_holdout_2025.csv", service)

    print("\n" + "="*70)
    print("✓ Validation completed!")
    print("="*70)
