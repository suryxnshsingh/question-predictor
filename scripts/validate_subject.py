#!/usr/bin/env python3
"""
Validate predictions for a subject against holdout data

Usage:
    python scripts/validate_subject.py --subject physics
    python scripts/validate_subject.py --subject physics --top-n 30
"""

import argparse
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.prediction_service import PredictionService
from config.subjects import list_subjects, get_subject_config


def main():
    parser = argparse.ArgumentParser(description='Validate predictions against holdout')
    parser.add_argument('--subject', '-s', required=True, help='Subject key')
    parser.add_argument('--top-n', '-n', type=int, default=25, help='Top N predictions to validate')

    args = parser.parse_args()

    # Validate subject
    available = list_subjects()
    if args.subject not in available:
        print(f"Error: Unknown subject '{args.subject}'")
        print(f"Available subjects: {', '.join(available)}")
        sys.exit(1)

    config = get_subject_config(args.subject)

    print("=" * 60)
    print(f"Validating {config['name']} Predictions")
    print("=" * 60)

    # Initialize service
    service = PredictionService(subject=args.subject)

    # Run validation
    results = service.validate_on_holdout(top_n=args.top_n)

    if 'error' in results:
        print(f"Error: {results['error']}")
        sys.exit(1)

    print(f"\nValidation Results (Top {args.top_n}):")
    print("-" * 40)
    print(f"Predicted topics:     {results['predicted_topics']}")
    print(f"Actual topics (2025): {results['actual_topics']}")
    print(f"Matched topics:       {results['matched_topics']}")
    print()
    print(f"Precision:            {results['precision']:.2%}")
    print(f"Recall:               {results['recall']:.2%}")
    print(f"F1 Score:             {results['f1_score']:.2%}")
    print()
    print(f"Question Coverage:    {results['question_coverage']:.2%}")
    print(f"                      ({int(results['question_coverage'] * results['total_holdout_questions'])}/{results['total_holdout_questions']} questions)")

    print("\nDone!")


if __name__ == '__main__':
    main()
