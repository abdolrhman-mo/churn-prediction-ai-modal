import streamlit as st

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .prediction-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
            text-align: center;
            font-size: 1.2rem;
            font-weight: bold;
        }
        .churn-yes {
            background-color: #ffebee;
            border: 2px solid #f44336;
            color: #d32f2f;
        }
        .churn-no {
            background-color: #e8f5e8;
            border: 2px solid #4caf50;
            color: #2e7d32;
        }
    </style>
    """, unsafe_allow_html=True)
