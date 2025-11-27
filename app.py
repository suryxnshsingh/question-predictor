"""
Streamlit dashboard for Physics Board Exam Predictor
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main_pipeline import run_prediction_pipeline
from src.validation import validate_predictions, load_holdout_data
from src import config

# Page config
st.set_page_config(
    page_title="Physics Exam Predictor",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📚 Physics Board Exam Question Predictor")
st.markdown("### Predict which chapters are most likely to appear in 2025")

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("""
**Model Details:**
- Subject: Physics (Class 12th)
- Training Data: 2016-2024 (716 questions)
- Test Data: 2025 (83 questions)
- Models: Baseline + ARIMA Ensemble
""")

run_prediction = st.sidebar.button("🔮 Generate Predictions", type="primary")
show_validation = st.sidebar.checkbox("Show Validation Results", value=True)

# Main content
if run_prediction or 'predictions' in st.session_state:

    if run_prediction:
        with st.spinner("Running prediction models..."):
            # Run pipeline
            predictions, ensemble = run_prediction_pipeline(verbose=False)
            st.session_state['predictions'] = predictions
            st.session_state['ensemble'] = ensemble
            st.success("✓ Predictions generated successfully!")

    predictions = st.session_state['predictions']

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Predictions", len(predictions))

    with col2:
        very_high = len(predictions[predictions['confidence'] == 'Very High'])
        st.metric("Very High Confidence", very_high)

    with col3:
        high = len(predictions[predictions['confidence'] == 'High'])
        st.metric("High Confidence", high)

    with col4:
        if show_validation:
            st.metric("F1 Score", "92.86%", delta="Validated")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Predictions", "✅ Validation", "📈 Visualizations"])

    # Tab 1: Predictions
    with tab1:
        st.subheader("Top 15 Predicted Chapters for 2025")

        # Create a nice display dataframe
        display_df = predictions.head(15).copy()
        display_df['Rank'] = range(1, len(display_df) + 1)
        display_df = display_df[['Rank', 'ensemble_score', 'confidence']]
        display_df.columns = ['Rank', 'Score', 'Confidence']

        # Color code by confidence
        def color_confidence(val):
            colors = {
                'Very High': 'background-color: #ffcccc',
                'High': '#ffffcc',
                'Moderate': '#ccffcc',
                'Low': '#cccccc'
            }
            return colors.get(val, '')

        styled_df = display_df.style.applymap(
            color_confidence,
            subset=['Confidence']
        ).format({'Score': '{:.4f}'})

        st.dataframe(styled_df, use_container_width=True)

        # Download button
        csv = predictions.to_csv()
        st.download_button(
            label="📥 Download Full Predictions (CSV)",
            data=csv,
            file_name="physics_predictions_2025.csv",
            mime="text/csv"
        )

    # Tab 2: Validation
    with tab2:
        if show_validation:
            st.subheader("Validation Against Actual 2025 Exam")

            # Load holdout data and validate
            holdout_df = load_holdout_data()
            results = validate_predictions(predictions, holdout_df, verbose=False)

            # Show metrics
            col1, col2, col3 = st.columns(3)

            top15 = results['top_15']
            with col1:
                st.metric("Precision@15", f"{top15['precision']:.1%}")
            with col2:
                st.metric("Recall@15", f"{top15['recall']:.1%}")
            with col3:
                st.metric("F1 Score", f"{top15['f1_score']:.1%}")

            # Detailed breakdown
            st.markdown("---")

            predicted_set = set(predictions.index[:15])
            actual_chapters = set(holdout_df['Chapter_Name'].unique())

            col1, col2 = st.columns(2)

            with col1:
                st.success(f"**✓ Correctly Predicted ({len(predicted_set & actual_chapters)})**")
                for chapter in sorted(predicted_set & actual_chapters):
                    rank = list(predictions.index).index(chapter) + 1
                    st.write(f"{rank}. {chapter[:40]}")

            with col2:
                misses = actual_chapters - predicted_set
                if misses:
                    st.error(f"**✗ Missed ({len(misses)})**")
                    for chapter in sorted(misses):
                        st.write(f"• {chapter[:40]}")
                else:
                    st.success("**🎯 Perfect! Predicted all chapters that appeared!**")
        else:
            st.info("Enable 'Show Validation Results' in the sidebar to see accuracy metrics.")

    # Tab 3: Visualizations
    with tab3:
        st.subheader("Prediction Visualizations")

        # Bar chart
        fig = px.bar(
            predictions.head(15),
            y=predictions.head(15).index,
            x='ensemble_score',
            orientation='h',
            color='confidence',
            color_discrete_map={
                'Very High': '#FF4444',
                'High': '#FFB344',
                'Moderate': '#44FF44',
                'Low': '#CCCCCC'
            },
            title="Top 15 Chapters by Score",
            labels={'ensemble_score': 'Prediction Score', 'y': 'Chapter'}
        )
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Confidence distribution pie chart
        st.subheader("Confidence Distribution")
        conf_counts = predictions['confidence'].value_counts()

        fig_pie = px.pie(
            values=conf_counts.values,
            names=conf_counts.index,
            color=conf_counts.index,
            color_discrete_map={
                'Very High': '#FF4444',
                'High': '#FFB344',
                'Moderate': '#44FF44',
                'Low': '#CCCCCC'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

else:
    # Welcome screen
    st.info("👈 Click 'Generate Predictions' in the sidebar to start")

    st.markdown("""
    ## How it works:

    1. **Data Collection**: Analyzes 716 Physics questions from 2016-2024
    2. **Feature Engineering**: Extracts frequency, recency, and trend patterns
    3. **Ensemble Modeling**: Combines Baseline + ARIMA predictions
    4. **Validation**: Tested against actual 2025 exam (92.86% F1 Score!)

    ### Model Performance:
    - ✅ **100% Recall**: Predicted ALL chapters that appeared in 2025
    - ✅ **86.67% Precision**: Most predictions were correct
    - ✅ **92.86% F1 Score**: Excellent overall accuracy

    ### Chapters Coverage:
    - Electrostatics, Magnetism, Current Electricity
    - Optics, Wave Optics, Electromagnetic Waves
    - Modern Physics (Atoms, Nuclei, Dual Nature)
    - Semiconductor Electronics
    - And more...
    """)

# Footer
st.markdown("---")
st.markdown("""
<small>
⚠️ **Disclaimer:** These predictions are based on historical patterns and should be used
as a study guide only. Always prepare comprehensively for your exams.
</small>
""", unsafe_allow_html=True)
