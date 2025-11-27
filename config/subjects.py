"""
Subject configuration registry

To add a new subject:
1. Create topic keywords file in config/topics/{subject}_topics.py
2. Add entry here with data paths
3. Run: python scripts/process_subject.py --subject {subject_key}
"""

SUBJECTS = {
    'physics': {
        'name': 'Physics',
        'train_data': 'data/raw/physics_train_2016_2024.csv',
        'holdout_data': 'data/raw/physics_holdout_2025.csv',
        'processed_data': 'data/processed/physics_with_topics.csv',
        'output_file': 'data/output/physics_predictions.json',
        'topics_module': 'config.topics.physics_topics',
        'topics_var': 'PHYSICS_TOPICS',
    },
    # Example for adding Chemistry:
    # 'chemistry': {
    #     'name': 'Chemistry',
    #     'train_data': 'data/raw/chemistry_train_2016_2024.csv',
    #     'holdout_data': 'data/raw/chemistry_holdout_2025.csv',
    #     'processed_data': 'data/processed/chemistry_with_topics.csv',
    #     'output_file': 'data/output/chemistry_predictions.json',
    #     'topics_module': 'config.topics.chemistry_topics',
    #     'topics_var': 'CHEMISTRY_TOPICS',
    # },
}


def get_subject_config(subject_key: str) -> dict:
    """Get configuration for a subject"""
    if subject_key not in SUBJECTS:
        available = ', '.join(SUBJECTS.keys())
        raise ValueError(f"Unknown subject: {subject_key}. Available: {available}")
    return SUBJECTS[subject_key]


def list_subjects() -> list:
    """List all configured subjects"""
    return list(SUBJECTS.keys())
