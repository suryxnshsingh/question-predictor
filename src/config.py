"""
Configuration settings for the board exam prediction system
"""

# Data paths
DATA_PATH = "data/raw/physics_train_2016_2024.csv"
HOLDOUT_PATH = "data/raw/physics_holdout_2025.csv"
PROCESSED_PATH = "data/processed/"
PREDICTIONS_PATH = "data/predictions/"

# Model parameters
YEARS_HISTORY = 9  # 2016-2024
PREDICTION_YEAR = 2025
TOP_N_PREDICTIONS = 25

# Ensemble weights
WEIGHTS = {
    'baseline': 0.30,
    'arima': 0.40,
    'association': 0.30,
}

# Board exam specific settings
BOARD = 'Board'
SUBJECT = 'Physics'
QUESTION_TYPES = ['Multiple Choice Question', 'Fill in the Blanks',
                  'Very Short Answer', 'Short Answer', 'Long Answer',
                  'One Word Answer', 'Numerical Answer']

# Validation settings
CONFIDENCE_THRESHOLDS = {
    'very_high': 0.7,
    'high': 0.5,
    'moderate': 0.3
}

# Feature engineering parameters
RECENCY_WEIGHT = 0.2
MIN_SUPPORT = 0.2  # For association rules
MIN_CONFIDENCE = 0.5  # For association rules
