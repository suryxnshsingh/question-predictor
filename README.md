# Physics Board Exam Question Predictor

A machine learning system that predicts which Physics chapters are most likely to appear in Class 12th board exams based on historical patterns.

## Results, honestly

- Trained on 716 questions from 2016-2024, tested on the 2025 paper.
- The 2025 evaluation below isn't a meaningful accuracy claim. It scores the top 15 predictions out of 15 topics, which is effectively the whole syllabus, so predicting every chapter would get the same score: 100% recall and about 93% F1.
- The potentially useful output is the ranking of chapters, which hasn't been tested against a baseline. A fair test would compare it with simple historical frequency over several held-out years.

## 📁 Project Structure

```
question-predictor/
├── data/
│   ├── raw/                          # Original and filtered data
│   │   ├── physics_train_2016_2024.csv
│   │   └── physics_holdout_2025.csv
│   ├── processed/                    # Feature matrices
│   │   ├── topic_matrix.csv
│   │   └── features.csv
│   └── predictions/                  # Model outputs
│       └── physics_predictions_2025.csv
│
├── src/
│   ├── config.py                     # Configuration settings
│   ├── data_preprocessing.py         # Data loading and feature engineering
│   ├── main_pipeline.py              # Main prediction pipeline
│   ├── validation.py                 # Model validation
│   └── models/
│       ├── baseline_model.py         # Frequency + recency heuristics
│       ├── arima_model.py            # Time series predictions
│       ├── association_rules.py      # Co-occurrence patterns
│       └── ensemble.py               # Weighted ensemble
│
├── app.py                            # Streamlit dashboard
├── requirements.txt                  # Python dependencies
├── venv/                             # Virtual environment
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source venv/bin/activate

# Install all dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. Run Predictions

```bash
# Generate predictions for 2025
python src/main_pipeline.py
```

This will:
- Load and preprocess data from 2016-2024
- Train Baseline and ARIMA models
- Combine predictions using ensemble
- Save results to `data/predictions/physics_predictions_2025.csv`

### 3. Validate Results

```bash
# Test predictions against actual 2025 exam
python src/validation.py
```

### 4. Launch Dashboard

```bash
# Start interactive web interface
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 🔬 How It Works

### Data Pipeline

1. **Data Collection**: 716 Physics questions from 2016-2024
   - Chapters: 15 different topics
   - Question types: MCQ, Short Answer, Long Answer, etc.
   - Years: 2016, 2017, 2018, 2019, 2020, 2022, 2023, 2024

2. **Feature Engineering**:
   - **Frequency**: How often each chapter appears (0-1)
   - **Recency**: Years since last appearance
   - **Trend**: Recent vs older appearance rate
   - **Total marks**: Historical weightage

3. **Prediction Models**:

   **Baseline Model (30% weight)**
   - Heat score = frequency × (1 + recency_weight × recency) × √trend
   - Simple but effective for consistent patterns

   **ARIMA Model (40% weight)**
   - Auto-ARIMA for each chapter's time series
   - Captures temporal patterns and trends
   - Predicts probability of next appearance

4. **Ensemble**:
   - Weighted average of model predictions
   - Boosting for high-frequency chapters
   - Confidence levels: Very High, High, Moderate, Low

### Model Performance

#### Metrics (Top-15 Predictions)

These numbers come from selecting all 15 topics, so they say little about the model. See "Results, honestly" above.

| Metric | Score |
|--------|-------|
| Precision@15 | 86.67% |
| Recall@15 | 100.00% |
| F1 Score | 92.86% |

#### What Actually Happened in 2025?
✅ **13/13 chapters correctly predicted** (100% recall)
⚠️ **2 false positives** (Nuclei and one duplicate chapter name)

### Correctly Predicted Chapters (2025):
1. Magnetism and Matter (Rank: 1, Score: 1.000)
2. Moving Charges and Magnetism (Rank: 2, Score: 0.787)
3. Current Electricity (Rank: 3, Score: 0.694)
4. Electric Charges and Fields (Rank: 4, Score: 0.659)
5. Electromagnetic Induction (Rank: 5, Score: 0.601)
6. Alternating Current (Rank: 6, Score: 0.574)
7. Ray Optics and Optical Instruments (Rank: 8, Score: 0.534)
8. Electromagnetic Waves (Rank: 9, Score: 0.533)
9. Semiconductor Electronics (Rank: 10, Score: 0.531)
10. Electrostatic Potential and Capacitance (Rank: 11, Score: 0.510)
11. Dual Nature of Radiation and Matter (Rank: 12, Score: 0.504)
12. Atoms (Rank: 13, Score: 0.444)
13. Wave Optics (Rank: 14, Score: 0.369)

## 📊 Using the Dashboard

The Streamlit dashboard provides:

1. **Predictions Tab**:
   - Top 15 predicted chapters with scores
   - Confidence levels (color-coded)
   - Download predictions as CSV

2. **Validation Tab**:
   - Precision, Recall, F1 metrics
   - Correctly predicted chapters
   - Missed chapters (if any)
   - False positives

3. **Visualizations Tab**:
   - Horizontal bar chart of predictions
   - Confidence distribution pie chart
   - Interactive filtering

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Model ensemble weights
WEIGHTS = {
    'baseline': 0.30,
    'arima': 0.40,
}

# Number of predictions
TOP_N_PREDICTIONS = 25

# Feature engineering
RECENCY_WEIGHT = 0.2
```

## 📈 Extending to Other Subjects

To adapt for Chemistry, Maths, or Biology:

1. Filter data for the new subject:
   ```python
   subject_df = df[df['Subject'] == 'Chemistry']
   ```

2. Update `config.py`:
   ```python
   SUBJECT = 'Chemistry'
   DATA_PATH = "data/raw/chemistry_train.csv"
   ```

3. Run the same pipeline - it will automatically adapt!

## 🎓 Study Strategy

Use predictions as a **strategic guide**:

1. **Very High Confidence (🔴)**: Deep preparation required
   - Master all concepts and problem types
   - Practice previous years thoroughly

2. **High Confidence (🟡)**: Strong preparation recommended
   - Cover major concepts and derivations
   - Practice important numericals

3. **Moderate Confidence (🟢)**: Standard preparation
   - Understand core concepts
   - Know basic formulae

4. **Low Confidence (⚪)**: Optional deep dive
   - Quick revision sufficient
   - Focus on easy marks

## ⚠️ Important Notes

1. **Not a Guarantee**: Predictions are based on historical patterns only
2. **Comprehensive Prep**: Always prepare ALL chapters thoroughly
3. **Pattern Changes**: Board may change question patterns
4. **Use as Guide**: Helps prioritize, doesn't replace studying

## 🛠 Dependencies

- pandas: Data manipulation
- numpy: Numerical operations
- scikit-learn: Feature engineering
- statsmodels: Statistical models
- pmdarima: Auto-ARIMA
- mlxtend: Association rules (optional)
- streamlit: Web dashboard
- plotly: Interactive visualizations

## 📝 Data Format

Expected CSV columns:
```csv
Year,Subject,Chapter_Name,Question type,Marks
2024,Physics,Current Electricity,Short Answer,3
```

## 🤝 Contributing

To improve the model:

1. Add more historical data (more years = better patterns)
2. Implement additional models (Random Forest, XGBoost)
3. Add cross-subject correlation analysis
4. Incorporate official syllabus weightages

## 📄 License

This project is for educational purposes only.

## 👨‍💻 Author

Built with Claude Code - AI-powered development assistant

---

**Good luck with your exams! 📚✨**
