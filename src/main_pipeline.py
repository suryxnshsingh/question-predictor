"""
Main pipeline orchestrating all models
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing import ExamDataPreprocessor
from src.models.baseline_model import BaselinePredictor
from src.models.arima_model import ARIMAPredictor
from src.models.ensemble import EnsemblePredictor
from src import config

def run_prediction_pipeline(verbose: bool = True):
    """
    Run complete prediction pipeline for Physics 2025

    Args:
        verbose: Print detailed progress

    Returns:
        Tuple of (final_predictions DataFrame, ensemble model)
    """
    if verbose:
        print("\n" + "="*70)
        print("BOARD EXAM PREDICTION PIPELINE - PHYSICS 2025")
        print("="*70)

    # Step 1: Data Preprocessing
    if verbose:
        print("\nStep 1: Loading and preprocessing data...")
        print("-" * 70)

    preprocessor = ExamDataPreprocessor(config.DATA_PATH)
    df = preprocessor.load_data()

    topic_matrix = preprocessor.create_topic_matrix()
    features = preprocessor.engineer_features()

    # Save processed data
    preprocessor.save_processed_data(config.PROCESSED_PATH)

    # Step 2: Individual Models
    if verbose:
        print("\n" + "="*70)
        print("Step 2: Training individual prediction models...")
        print("-" * 70)

    # 2a. Baseline model
    if verbose:
        print("\n[1/2] Baseline Model (Frequency + Recency)...")

    baseline = BaselinePredictor(features, recency_weight=config.RECENCY_WEIGHT)
    baseline_scores = baseline.calculate_heat_score()

    if verbose:
        print(f"      ✓ Baseline model trained")
        print(f"      Top prediction: {baseline_scores.idxmax()} (score: {baseline_scores.max():.3f})")

    # 2b. ARIMA model
    if verbose:
        print("\n[2/2] ARIMA Time Series Model...")

    arima = ARIMAPredictor(topic_matrix)
    arima_scores = arima.predict_all_topics()

    if verbose:
        print(f"      ✓ ARIMA models fitted for {len(arima_scores)} chapters")
        print(f"      Top prediction: {arima_scores.idxmax()} (score: {arima_scores.max():.3f})")

    # Step 3: Ensemble
    if verbose:
        print("\n" + "="*70)
        print("Step 3: Creating ensemble predictions...")
        print("-" * 70)

    ensemble = EnsemblePredictor()

    # Add predictors with weights from config
    ensemble.add_predictor('baseline', baseline_scores, weight=config.WEIGHTS['baseline'])
    ensemble.add_predictor('arima', arima_scores, weight=config.WEIGHTS['arima'])

    # Combine predictions
    ensemble.combine_predictions()

    # Apply boosting for high-frequency chapters
    high_freq_chapters = features[features['frequency'] >= 0.9]['chapter'].tolist()
    ensemble.apply_boosting_factors(high_frequency_chapters=high_freq_chapters)

    # Get final predictions
    final_predictions = ensemble.get_final_predictions(top_n=config.TOP_N_PREDICTIONS)

    # Step 4: Display results
    if verbose:
        print("\n" + "="*70)
        print("FINAL PREDICTIONS FOR PHYSICS 2025")
        print("="*70)

        confidence_emoji = {
            'Very High': '🔴',
            'High': '🟡',
            'Moderate': '🟢',
            'Low': '⚪'
        }

        print(f"\nTop {min(25, len(final_predictions))} Chapters Most Likely to Appear:\n")

        for i, (chapter, row) in enumerate(final_predictions.head(25).iterrows(), 1):
            emoji = confidence_emoji.get(row['confidence'], '')
            # Get breakdown
            breakdown = ensemble.get_score_breakdown(chapter)

            print(f"{i:2}. {chapter[:50]:<50} [{row['ensemble_score']:.4f}] {emoji} {row['confidence']}")
            if verbose:
                print(f"    └─ Baseline: {breakdown.get('baseline', 0):.3f} | "
                      f"ARIMA: {breakdown.get('arima', 0):.3f}")

        # Summary statistics
        print("\n" + "="*70)
        print("CONFIDENCE DISTRIBUTION")
        print("="*70)
        conf_dist = final_predictions['confidence'].value_counts()
        for conf_level, count in conf_dist.items():
            emoji = confidence_emoji.get(conf_level, '')
            print(f"{emoji} {conf_level:12}: {count:2} chapters")

    return final_predictions, ensemble


def save_predictions(predictions: pd.DataFrame, output_path: str = None):
    """
    Save predictions to CSV

    Args:
        predictions: DataFrame with predictions
        output_path: Path to save CSV (uses config if None)
    """
    import os
    if output_path is None:
        os.makedirs(config.PREDICTIONS_PATH, exist_ok=True)
        output_path = f"{config.PREDICTIONS_PATH}/physics_predictions_2025.csv"

    predictions.to_csv(output_path)
    print(f"\n✓ Predictions saved to: {output_path}")


if __name__ == "__main__":
    # Run pipeline
    predictions, model = run_prediction_pipeline(verbose=True)

    # Save predictions
    save_predictions(predictions)

    print("\n" + "="*70)
    print("✓ Pipeline completed successfully!")
    print("="*70)
