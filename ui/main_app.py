import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import our modular components
from ui.data_manager import load_and_process_data, train_svm_model
from ui.ui_pages.svm_analysis_page import render_svm_analysis_page
from ui.ui_pages.technical_implementation_page import render_technical_implementation_page

# Page config
st.set_page_config(
    page_title="SVM Recall Optimization",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .recall-highlight {
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .threshold-box {
        background-color: #f8f9fa;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .model-metrics {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎯 SVM Recall Optimization Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<div class="recall-highlight">🚀 Maximizing Recall for Churn Detection - No Customer Left Behind!</div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Choose Section:", [
    "📊 SVM Threshold Analysis",
    "⚙️ Technical Implementation"
])

# Load data and train model
with st.spinner("🔄 Loading data and training SVM model..."):
    df = load_and_process_data()
    svm_model, scaler, processing, threshold_results, svm_probs, y_test, X_test_scaled = train_svm_model(df)

# Convert results to DataFrame
results_df = pd.DataFrame(threshold_results)

# Page routing
if page == "📊 SVM Threshold Analysis":
    render_svm_analysis_page(df, results_df, y_test)

elif page == "⚙️ Technical Implementation":
    render_technical_implementation_page(df)

# Footer with key insights
st.markdown("---")
st.markdown("### 🎯 Key Takeaways for Your AI Course")

insights = [
    "**Threshold 0.1 = Maximum Recall**: Lower thresholds catch more positive cases",
    "**SMOTE Balancing**: Essential for imbalanced datasets like churn prediction", 
    "**SVM + Linear Kernel**: Good interpretability for business applications",
    "**Pipeline Architecture**: Proper ML workflow from data to deployment",
    "**Recall vs Precision Trade-off**: Understanding when to optimize for each metric"
]

for insight in insights:
    st.markdown(f"• {insight}")
