"""
Validation of decoupled prediction system
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.topic_extraction import TopicExtractor
from src.models.decoupled_predictor import (
    TopicOnlyPredictor,
    QuestionTypeDistributionPredictor,
    DecoupledCombiner
)

def calculate_metrics(predicted_combos, actual_combos, name=""):
    """Calculate precision, recall, F1"""
    hits = len(predicted_combos & actual_combos)
    precision = hits / len(predicted_combos) if predicted_combos else 0
    recall = hits / len(actual_combos) if actual_combos else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'name': name,
        'hits': hits,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predicted_count': len(predicted_combos),
        'actual_count': len(actual_combos)
    }

def validate_decoupled_system():
    """Full validation"""
    print("="*70)
    print("VALIDATING DECOUPLED SYSTEM ON 2025 DATA")
    print("="*70)

    # Load data
    train_df = pd.read_csv("data/processed/physics_with_topics.csv")
    holdout_df = pd.read_csv("data/raw/physics_holdout_2025.csv")

    # Process holdout
    extractor = TopicExtractor()
    holdout_df = extractor.process_dataframe(holdout_df)

    # Get actual 2025 combinations
    actual_combos = set()
    for _, row in holdout_df.iterrows():
        combo = (row['primary_topic'], row['Question type'])
        actual_combos.add(combo)

    print(f"\nActual 2025: {len(actual_combos)} unique topic+type combinations")

    # Build decoupled system
    print("\nBuilding Decoupled Models...")
    print("-"*70)

    topic_model = TopicOnlyPredictor(train_df)
    topic_model.predict(method='baseline')

    qtype_model = QuestionTypeDistributionPredictor(train_df)

    # Test both baseline and trend methods for qtype
    results = {}

    for qtype_method in ['baseline', 'trend']:
        print(f"\n{qtype_method.upper()} Method:")
        print("-"*70)

        qtype_model.predict(method=qtype_method)
        combiner = DecoupledCombiner(topic_model, qtype_model)

        # Evaluate at different top_n
        for top_n in [10, 20, 30, 50, 75]:
            top_pred = combiner.get_top_predictions(top_n=top_n)
            predicted_combos = set(zip(top_pred['topic'], top_pred['question_type']))

            metrics = calculate_metrics(
                predicted_combos,
                actual_combos,
                name=f"{qtype_method}_top{top_n}"
            )
            results[f"{qtype_method}_top{top_n}"] = metrics

            print(f"  Top {top_n:2}: P={metrics['precision']:.2%} R={metrics['recall']:.2%} F1={metrics['f1_score']:.2%}")

    # Detailed breakdown for best performing
    print("\n" + "="*70)
    print("DETAILED BREAKDOWN (Trend method, Top 50)")
    print("="*70)

    qtype_model.predict(method='trend')
    combiner = DecoupledCombiner(topic_model, qtype_model)
    top_50 = combiner.get_top_predictions(top_n=50)
    predicted_combos_50 = set(zip(top_50['topic'], top_50['question_type']))

    hits = predicted_combos_50 & actual_combos
    misses = actual_combos - predicted_combos_50

    print(f"\n✓ CORRECTLY PREDICTED ({len(hits)}):")
    for topic, qtype in sorted(hits)[:15]:
        print(f"  • {topic[:40]:40} [{qtype}]")
    if len(hits) > 15:
        print(f"  ... and {len(hits) - 15} more")

    print(f"\n✗ MISSED ({len(misses)}):")
    for topic, qtype in sorted(misses)[:10]:
        print(f"  • {topic[:40]:40} [{qtype}]")
    if len(misses) > 10:
        print(f"  ... and {len(misses) - 10} more")

    # Compare with previous coupled approach
    print("\n" + "="*70)
    print("COMPARISON: Coupled vs Decoupled")
    print("="*70)

    print("\nPrevious Coupled Approach (from earlier):")
    print(f"  Top 50: P=38.00% R=29.69% F1=33.33%")

    print("\nNew Decoupled Approach (baseline):")
    baseline_50 = results['baseline_top50']
    print(f"  Top 50: P={baseline_50['precision']:.2%} R={baseline_50['recall']:.2%} F1={baseline_50['f1_score']:.2%}")

    print("\nNew Decoupled Approach (trend):")
    trend_50 = results['trend_top50']
    print(f"  Top 50: P={trend_50['precision']:.2%} R={trend_50['recall']:.2%} F1={trend_50['f1_score']:.2%}")

    improvement = (trend_50['f1_score'] - 0.3333) / 0.3333 * 100
    print(f"\nImprovement: {improvement:+.1f}%")

    # Topic-level accuracy (ignoring question type)
    print("\n" + "="*70)
    print("TOPIC-LEVEL ACCURACY (Just Topics, No QType)")
    print("="*70)

    actual_topics = set(holdout_df['primary_topic'].unique())
    predicted_topics = set(top_50['topic'].unique())

    topic_metrics = calculate_metrics(predicted_topics, actual_topics, "topics")
    print(f"Precision: {topic_metrics['precision']:.2%}")
    print(f"Recall: {topic_metrics['recall']:.2%}")
    print(f"F1 Score: {topic_metrics['f1_score']:.2%}")

    return results

if __name__ == "__main__":
    results = validate_decoupled_system()

    print("\n" + "="*70)
    print("✓ Validation Complete!")
    print("="*70)
