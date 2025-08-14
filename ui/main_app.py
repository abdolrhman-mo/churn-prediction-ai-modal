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

# Load and preprocess data function
@st.cache_data
def load_and_process_data():
    """Load and preprocess the churn data following your exact pipeline"""
    try:
        # Try to load your actual data
        df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    except:
        st.warning("⚠️ CSV file not found. Using demo data for showcase.")
        # Create realistic demo data
        np.random.seed(42)
        n_samples = 2000
        
        df = pd.DataFrame({
            'customerID': [f'C{i:04d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'tenure': np.random.randint(1, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
            'OnlineSecurity': np.random.choice(['Yes', 'No'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
            'MonthlyCharges': np.random.normal(65, 25, n_samples).clip(20, 120),
            'TotalCharges': np.random.normal(2000, 1500, n_samples).clip(20, 8000),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
        })
    
    # Follow your exact preprocessing pipeline
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df = df.drop(['customerID'], axis=1)
    
    return df

@st.cache_data
def train_svm_model(df):
    """Train SVM model following your exact pipeline and find optimal threshold"""
    
    # Prepare data exactly like your code
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Column classification (your logic)
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    feature_cols = categorical_cols.copy()
    binary_cols = []
    onehot_cols = []
    
    for col in feature_cols:
        if X[col].nunique() == 2:
            binary_cols.append(col)
        else:
            onehot_cols.append(col)
    
    # Your exact preprocessing pipeline
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    onehot_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot_encoding', OneHotEncoder())
    ])
    binary_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('binary_encoding', OrdinalEncoder())
    ])
    
    processing = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('onehot', onehot_pipeline, onehot_cols),
        ('ordinal', binary_pipeline, binary_cols)
    ], remainder='passthrough')
    
    # Train-test split (your settings)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Process data
    X_train_cleaned = processing.fit_transform(X_train)
    X_test_cleaned = processing.transform(X_test)
    
    # SMOTE resampling (your approach)
    smote = SMOTE(random_state=12)
    X_resampled, y_resampled = smote.fit_resample(X_train_cleaned, y_train)
    
    # Scaling (your approach)
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test_cleaned)
    
    # Train SVM (your settings)
    svm_model = SVC(kernel='linear', random_state=42, probability=True)
    svm_model.fit(X_resampled_scaled, y_resampled)
    
    # Threshold optimization (your logic)
    svm_probs = svm_model.predict_proba(X_test_scaled)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    threshold_results = []
    for threshold in thresholds:
        predictions = (svm_probs >= threshold).astype(int)
        pred_labels = ['Yes' if pred == 1 else 'No' for pred in predictions]
        
        recall = recall_score(y_test, pred_labels, pos_label='Yes')
        precision = precision_score(y_test, pred_labels, pos_label='Yes')
        f1 = f1_score(y_test, pred_labels, pos_label='Yes')
        accuracy = accuracy_score(y_test, pred_labels)
        
        threshold_results.append({
            'Threshold': threshold,
            'Recall': recall,
            'Precision': precision,
            'F1_Score': f1,
            'Accuracy': accuracy
        })
    
    return svm_model, scaler, processing, threshold_results, svm_probs, y_test, X_test_scaled

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

if page == "📊 SVM Threshold Analysis":
    st.markdown("## 📊 SVM Threshold Analysis")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Size", len(df))
    with col2:
        churn_rate = (df['Churn'] == 'Yes').mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}")
    with col3:
        st.metric("Features Used", len(df.columns)-1)
    with col4:
        st.metric("Test Set Size", len(y_test))
    
    st.markdown("### 🎯 Why Threshold 0.1 Achieves Highest Recall")
    
    # Threshold comparison chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results_df['Threshold'],
        y=results_df['Recall'],
        mode='lines+markers',
        name='Recall',
        line=dict(color='#ff6b6b', width=4),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=results_df['Threshold'],
        y=results_df['Precision'],
        mode='lines+markers',
        name='Precision',
        line=dict(color='#4ecdc4', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=results_df['Threshold'],
        y=results_df['F1_Score'],
        mode='lines+markers',
        name='F1 Score',
        line=dict(color='#45b7d1', width=3)
    ))
    
    # Highlight the 0.1 threshold
    max_recall_idx = results_df['Recall'].idxmax()
    best_threshold = results_df.loc[max_recall_idx, 'Threshold']
    best_recall = results_df.loc[max_recall_idx, 'Recall']
    
    fig.add_vline(x=best_threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"Best Recall: {best_recall:.4f}")
    
    fig.update_layout(
        title="🎯 SVM Threshold vs Performance Metrics",
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation
    st.markdown(f"""
    ### 🎯 Key Insight: Threshold {best_threshold} = Maximum Recall
    
    **Why lower threshold = higher recall?**
    - **Lower threshold (0.1)**: Model says "Yes, will churn" even with low confidence
    - **Higher threshold (0.9)**: Model only says "Yes" when very confident
    - **For churn detection**: Better to catch all potential churners (even false alarms) than miss real ones
    
    **Your SVM at threshold 0.1:**
    - Catches **{best_recall:.1%}** of all customers who will actually churn
    - This means **{(1-best_recall)*100:.1f}% missed churn rate** (very low!)
    """)

elif page == "⚙️ Technical Implementation":
    st.markdown("## ⚙️ Technical Implementation")
    
    st.markdown("### 🛠️ Your ML Pipeline Architecture")
    
    # Pipeline flow
    st.markdown("""
    ```
    Data Loading → Data Cleaning → Feature Engineering → 
    Preprocessing → SMOTE Resampling → Scaling → 
    SVM Training → Threshold Optimization → Performance Evaluation
    ```
    """)
    
    # Code explanations
    st.markdown("### 📝 Key Code Components")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Processing", "Model Training", "Threshold Tuning", "Model Saving", "Project Phases"])
    
    with tab1:
        st.markdown("#### Data Loading & Preprocessing")
        st.code('''
# Your data loading logic
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Handle TotalCharges conversion
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Column classification for preprocessing
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        ''', language='python')
    
    with tab2:
        st.markdown("#### SVM Model Training")
        st.code('''
# SMOTE for handling imbalanced data
smote = SMOTE(random_state=12)
X_resampled, y_resampled = smote.fit_resample(X_train_cleaned, y_train)

# Scaling for SVM
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# SVM with linear kernel
svm_model = SVC(kernel='linear', random_state=42, probability=True)
svm_model.fit(X_resampled_scaled, y_resampled)
        ''', language='python')
    
    with tab3:
        st.markdown("#### Threshold Optimization for Maximum Recall")
        st.code('''
# Get prediction probabilities
svm_probs = svm_model.predict_proba(X_test_scaled)[:, 1]

# Test different thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best_recall = 0

for threshold in thresholds:
    predictions = (svm_probs >= threshold).astype(int)
    pred_labels = ['Yes' if pred == 1 else 'No' for pred in predictions]
    
    recall = recall_score(y_test, pred_labels, pos_label='Yes')
    if recall > best_recall:
        best_recall = recall
        best_threshold = threshold
        
# Result: threshold = 0.1 gives highest recall!
        ''', language='python')
    
    with tab4:
        st.markdown("#### Model Persistence")
        st.code('''
# Save the trained model
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(processing, "models/preprocessing.pkl")

# Save optimal threshold
with open("models/svm_threshold.txt", 'w') as f:
    f.write(str(best_threshold))
        ''', language='python')
    
    with tab5:
        st.markdown("#### 📋 Complete Project Phases Overview")
        st.markdown("""
        **Phase 1: Data Loading & Initial Exploration**
        - Load CSV file into pandas DataFrame
        - Examine data shape, columns, and basic info
        - Understand what data you're working with
        
        **Phase 2: Data Quality Assessment**
        - Check for missing values, duplicates, wrong data types
        - Identify imbalanced target (80% stayed, 20% churned)
        - Find data problems that could break your model
        
        **Phase 3: Data Cleaning**
        - Convert 'TotalCharges' from text to numeric
        - Handle missing values and drop problematic rows
        - Ensure data is ready for machine learning
        
        **Phase 4: Feature Engineering**
        - Create new features from existing data
        - Handle categorical variables (gender, contract type, services)
        - Prepare features for model training
        
        **Phase 5: Data Visualization & Exploration**
        - Create charts to understand churn patterns
        - Identify which customer traits predict churn
        - Generate business insights from data
        
        **Phase 6: Data Preprocessing**
        - Remove useless columns (customerID)
        - Convert text to numbers for ML algorithms
        - Handle imbalanced data with SMOTE
        - Scale numerical features
        - Split data into training and testing sets
        
        **Phase 7: Model Training**
        - Train multiple algorithms (Logistic Regression, SVM, XGBoost, Random Forest)
        - Let algorithms learn patterns from customer data
        - Create models that can predict churn
        
        **Phase 8: Model Evaluation**
        - Test models on unseen data
        - Calculate accuracy, precision, recall, F1-score
        - Compare performance across different algorithms
        
        **Phase 9: Model Optimization**
        - Fine-tune models for better performance
        - Optimize thresholds for maximum recall
        - Choose best model for deployment
        """)
    
    # Technical achievements
    st.markdown("### 🏆 Your Technical Achievements")
    
    achievements = [
        "✅ **Complete ML Pipeline**: Data loading → preprocessing → training → evaluation",
        "✅ **Imbalanced Data Handling**: SMOTE for synthetic minority oversampling",
        "✅ **Feature Engineering**: Proper encoding for categorical and numerical features",
        "✅ **Model Optimization**: Threshold tuning for maximum recall",
        "✅ **Production Ready**: Model serialization with joblib",
        "✅ **Performance Analysis**: Comprehensive evaluation metrics"
    ]
    
    for achievement in achievements:
        st.markdown(achievement)
    
    # Dataset info
    st.markdown("### 📊 Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Numerical Features:**")
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for col in num_cols:
            st.write(f"• {col}")
    
    with col2:
        st.markdown("**Categorical Features:**")  
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        cat_cols.remove('Churn')  # Remove target
        for col in cat_cols:
            st.write(f"• {col}")
    
    # Success metrics
    st.markdown("### 📊 Success Metrics & Project Goals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Minimum Viable Model:**")
        st.markdown("- **Accuracy > 80%**: Better than random guessing")
        st.markdown("- **Recall > 70%**: Catch most customers who will churn")
        st.markdown("- **Model runs without errors**: Technical success")
    
    with col2:
        st.markdown("**💼 Business Success:**")
        st.markdown("- **Actionable insights**: Which factors drive churn most?")
        st.markdown("- **Cost savings**: Prevent customer loss through targeted retention")
        st.markdown("- **ROI**: Model saves more money than it costs to develop")
    
    st.markdown("### 📁 Expected Project Outputs")
    st.markdown("1. **Trained model file**: Save your model to use later")
    st.markdown("2. **Performance report**: Accuracy, precision, recall metrics")
    st.markdown("3. **Feature importance**: Which customer attributes matter most")
    st.markdown("4. **Predictions**: List of customers likely to churn")
    st.markdown("5. **Documentation**: What you learned and how to use the model")

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
