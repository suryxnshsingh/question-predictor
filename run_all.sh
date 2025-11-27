#!/bin/bash

echo "======================================================================="
echo "Physics Board Exam Predictor - Complete Pipeline"
echo "======================================================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Step 1: Generate Predictions
echo "Step 1: Generating predictions for 2025..."
echo "-----------------------------------------------------------------------"
python src/main_pipeline.py
echo ""

# Step 2: Validate
echo "Step 2: Validating against actual 2025 exam..."
echo "-----------------------------------------------------------------------"
python src/validation.py
echo ""

# Step 3: Launch Dashboard
echo "======================================================================="
echo "✓ Pipeline completed!"
echo "======================================================================="
echo ""
echo "To launch the interactive dashboard, run:"
echo "  streamlit run app.py"
echo ""
