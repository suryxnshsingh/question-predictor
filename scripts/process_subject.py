#!/usr/bin/env python3
"""
Process a subject: extract topics, build predictions, export for RAG

Usage:
    python scripts/process_subject.py --subject physics
    python scripts/process_subject.py --subject chemistry --force
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
    parser = argparse.ArgumentParser(description='Process subject data for predictions')
    parser.add_argument('--subject', '-s', required=True, help='Subject key (e.g., physics, chemistry)')
    parser.add_argument('--force', '-f', action='store_true', help='Force reprocess even if processed data exists')
    parser.add_argument('--top-n', '-n', type=int, default=50, help='Number of predictions to export')

    args = parser.parse_args()

    # Validate subject
    available = list_subjects()
    if args.subject not in available:
        print(f"Error: Unknown subject '{args.subject}'")
        print(f"Available subjects: {', '.join(available)}")
        sys.exit(1)

    config = get_subject_config(args.subject)

    print("=" * 60)
    print(f"Processing {config['name']}")
    print("=" * 60)

    # Check if force reprocess
    if args.force and os.path.exists(config['processed_data']):
        print(f"Removing existing processed data...")
        os.remove(config['processed_data'])

    # Initialize service (this processes data if needed)
    service = PredictionService(subject=args.subject)

    # Print stats
    stats = service.get_summary_stats()
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Export for RAG
    print(f"\nExporting top {args.top_n} predictions...")
    service.export_for_rag(top_n=args.top_n)

    print("\nDone!")


if __name__ == '__main__':
    main()
