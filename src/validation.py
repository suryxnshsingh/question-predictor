"""
Validation module to test predictions against holdout data
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config


def load_holdout_data() -> pd.DataFrame:
    """Load 2025 holdout data"""
    return pd.read_csv(config.HOLDOUT_PATH)


def get_actual_chapters(holdout_df: pd.DataFrame) -> set:
    """Get set of chapters that actually appeared in 2025"""
    return set(holdout_df['Chapter_Name'].unique())


def calculate_metrics(predicted_chapters: List[str],
                     actual_chapters: set,
                     top_n: int = 25) -> Dict:
    """
    Calculate precision, recall, and F1 score

    Args:
        predicted_chapters: List of predicted chapter names
        actual_chapters: Set of actual chapter names that appeared
        top_n: Number of top predictions considered

    Returns:
        Dictionary with metrics
    """
    # Take only top N predictions
    predicted_set = set(predicted_chapters[:top_n])

    # Calculate hits
    hits = len(predicted_set & actual_chapters)

    # Metrics
    precision = hits / len(predicted_set) if predicted_set else 0
    recall = hits / len(actual_chapters) if actual_chapters else 0

    # F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    return {
        'top_n': top_n,
        'predicted_count': len(predicted_set),
        'actual_count': len(actual_chapters),
        'hits': hits,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def validate_predictions(predictions_df: pd.DataFrame,
                        holdout_df: pd.DataFrame,
                        verbose: bool = True) -> Dict:
    """
    Validate predictions against holdout data

    Args:
        predictions_df: DataFrame with predicted chapters
        holdout_df: DataFrame with actual 2025 exam data
        verbose: Print detailed results

    Returns:
        Dictionary with validation results
    """
    if verbose:
        print("\n" + "="*70)
        print("VALIDATION AGAINST 2025 HOLDOUT DATA")
        print("="*70)

    # Get predicted and actual chapters
    predicted_chapters = predictions_df.index.tolist()
    actual_chapters = get_actual_chapters(holdout_df)

    if verbose:
        print(f"\nPredicted chapters: {len(predicted_chapters)}")
        print(f"Actual chapters in 2025: {len(actual_chapters)}")

    # Calculate metrics for different top-N values
    results = {}
    top_n_values = [10, 15, 20, 25]

    for n in top_n_values:
        if n <= len(predicted_chapters):
            metrics = calculate_metrics(predicted_chapters, actual_chapters, top_n=n)
            results[f'top_{n}'] = metrics

            if verbose:
                print(f"\n--- Top {n} Predictions ---")
                print(f"Precision@{n}: {metrics['precision']:.2%} ({metrics['hits']}/{metrics['predicted_count']} predicted chapters appeared)")
                print(f"Recall@{n}:    {metrics['recall']:.2%} ({metrics['hits']}/{metrics['actual_count']} actual chapters were predicted)")
                print(f"F1 Score:      {metrics['f1_score']:.2%}")

    # Detailed breakdown
    if verbose:
        print("\n" + "="*70)
        print("DETAILED BREAKDOWN")
        print("="*70)

        # Correctly predicted (hits)
        predicted_set = set(predicted_chapters[:25])
        hits = predicted_set & actual_chapters
        misses = actual_chapters - predicted_set
        false_positives = predicted_set - actual_chapters

        print(f"\n✓ CORRECTLY PREDICTED ({len(hits)}):")
        for i, chapter in enumerate(sorted(hits), 1):
            # Find rank in predictions
            try:
                rank = predicted_chapters.index(chapter) + 1
                score = predictions_df.loc[chapter, 'ensemble_score']
                print(f"  {i}. {chapter[:50]} (Rank: {rank}, Score: {score:.3f})")
            except:
                print(f"  {i}. {chapter[:50]}")

        if misses:
            print(f"\n✗ MISSED ({len(misses)}):")
            for i, chapter in enumerate(sorted(misses), 1):
                # Check if it was in our full predictions but lower ranked
                if chapter in predicted_chapters:
                    rank = predicted_chapters.index(chapter) + 1
                    score = predictions_df.loc[chapter, 'ensemble_score']
                    print(f"  {i}. {chapter[:50]} (Rank: {rank}, Score: {score:.3f})")
                else:
                    print(f"  {i}. {chapter[:50]} (Not in predictions)")

        if false_positives:
            print(f"\n⚠ FALSE POSITIVES ({len(false_positives)}):")
            for i, chapter in enumerate(sorted(false_positives), 1):
                rank = predicted_chapters.index(chapter) + 1
                score = predictions_df.loc[chapter, 'ensemble_score']
                print(f"  {i}. {chapter[:50]} (Rank: {rank}, Score: {score:.3f})")

    return results


def compare_with_baseline(holdout_df: pd.DataFrame) -> Dict:
    """
    Compare model performance with simple baseline
    (predicting most frequent chapters from history)

    Args:
        holdout_df: Holdout data

    Returns:
        Baseline metrics
    """
    # Simple baseline: all chapters appeared every year, so predict all
    all_possible_chapters = set([
        'Electrostatic Potential and Capacitance',
        'Electric Charges and Fields',
        'Current Electricity',
        'Moving Charges and Magnetism',
        'Magnetism and Matter',
        'Electromagnetic Induction',
        'Alternating Current',
        'Electromagnetic Waves',
        'Ray Optics and Optical Instruments',
        'Wave Optics',
        'Dual Nature of Radiation and Matter',
        'Atoms',
        'Nuclei',
        'Semiconductor Electronics: Materials, Devices and Simple Circuits'
    ])

    actual_chapters = get_actual_chapters(holdout_df)

    baseline_metrics = calculate_metrics(
        list(all_possible_chapters),
        actual_chapters,
        top_n=len(all_possible_chapters)
    )

    return baseline_metrics


if __name__ == "__main__":
    # Load data
    print("Loading validation data...")
    holdout_df = load_holdout_data()
    predictions_df = pd.read_csv(
        f"{config.PREDICTIONS_PATH}/physics_predictions_2025.csv",
        index_col=0
    )

    # Run validation
    results = validate_predictions(predictions_df, holdout_df, verbose=True)

    # Compare with baseline
    print("\n" + "="*70)
    print("BASELINE COMPARISON")
    print("="*70)
    baseline = compare_with_baseline(holdout_df)
    print(f"Simple Baseline (predict all common chapters):")
    print(f"  Precision: {baseline['precision']:.2%}")
    print(f"  Recall:    {baseline['recall']:.2%}")
    print(f"  F1 Score:  {baseline['f1_score']:.2%}")

    print("\n" + "="*70)
    print("✓ Validation completed!")
    print("="*70)
