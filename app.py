import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(
    page_title="🔮 Churn Prediction Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# Title and introduction
st.markdown('<h1 class="main-header">🔮 Telecom Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Predict customer churn and discover insights to improve retention")

# Sidebar for navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "🏠 Home & Model Training",
    "🔍 Single Customer Prediction", 
    "📈 Data Insights",
    "📊 Model Performance",
    "📋 Batch Predictions"
])

# Helper functions
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the telecom churn data"""
    try:
        # For demo purposes, create sample data if file not found
        # In your actual app, replace this with your CSV file path
        df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    except:
        st.warning("Using sample data. Upload your actual CSV file.")
        # Create sample data structure (you can remove this in your actual app)
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'customerID': [f'C{i:04d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'tenure': np.random.randint(1, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
            'MonthlyCharges': np.random.normal(65, 30, n_samples).clip(20, 120),
            'TotalCharges': np.random.normal(2300, 2000, n_samples).clip(20, 8500),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
        })
        
        # Add more realistic service columns
        for service in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
            df[service] = np.random.choice(['Yes', 'No'], n_samples)
    
    # Data preprocessing
    df = df.copy()
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    
    # Drop customerID
    df = df.drop(['customerID'], axis=1)
    
    return df

@st.cache_data
def train_models(df):
    """Train all models and return them with performance metrics"""
    
    # Prepare features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Identify column types
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Further classify categorical columns
    binary_cols = []
    onehot_cols = []
    
    for col in categorical_cols:
        if X[col].nunique() == 2:
            binary_cols.append(col)
        else:
            onehot_cols.append(col)
    
    # Create preprocessing pipeline
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    onehot_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot_encoding', OneHotEncoder(drop='first'))
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
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocess data
    X_train_processed = processing.fit_transform(X_train)
    X_test_processed = processing.transform(X_test)
    
    # Handle imbalanced data with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test_processed)
    
    # Train models
    models = {}
    results = {}
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train_balanced)
    models['Logistic Regression'] = lr_model
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_train_processed, y_train)
    models['Random Forest'] = rf_model
    
    # XGBoost
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    pos_weight = (y_train == 'No').sum() / (y_train == 'Yes').sum()
    xgb_model = XGBClassifier(scale_pos_weight=pos_weight, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train_processed, y_train_encoded)
    models['XGBoost'] = xgb_model
    
    # Evaluate models
    for name, model in models.items():
        if name == 'Logistic Regression':
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        elif name == 'XGBoost':
            y_pred_encoded = model.predict(X_test_processed)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        else:  # Random Forest
            y_pred = model.predict(X_test_processed)
            y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='Yes')
        recall = recall_score(y_test, y_pred, pos_label='Yes')
        f1 = f1_score(y_test, y_pred, pos_label='Yes')
        
        if name == 'XGBoost':
            roc_auc = roc_auc_score(y_test_encoded, y_pred_proba)
        else:
            roc_auc = roc_auc_score((y_test == 'Yes').astype(int), y_pred_proba)
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    return models, results, processing, scaler, X_test, y_test, label_encoder

def predict_single_customer(models, processing, scaler, customer_data, label_encoder=None):
    """Predict churn for a single customer"""
    
    # Convert to DataFrame
    customer_df = pd.DataFrame([customer_data])
    
    # Preprocess
    customer_processed = processing.transform(customer_df)
    customer_scaled = scaler.transform(customer_processed)
    
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        if name == 'Logistic Regression':
            pred = model.predict(customer_scaled)[0]
            prob = model.predict_proba(customer_scaled)[0][1]
        elif name == 'XGBoost':
            pred_encoded = model.predict(customer_processed)[0]
            pred = label_encoder.inverse_transform([pred_encoded])[0]
            prob = model.predict_proba(customer_processed)[0][1]
        else:  # Random Forest
            pred = model.predict(customer_processed)[0]
            prob = model.predict_proba(customer_processed)[0][1]
        
        predictions[name] = pred
        probabilities[name] = prob
    
    return predictions, probabilities

# Load data and train models
with st.spinner("Loading data and training models..."):
    df = load_and_preprocess_data()
    models, results, processing, scaler, X_test, y_test, label_encoder = train_models(df)

# Page routing
if page == "🏠 Home & Model Training":
    st.markdown("## 🎯 Project Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(df), help="Total customers in dataset")
    with col2:
        churn_rate = (df['Churn'] == 'Yes').mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}", help="Percentage of customers who churned")
    with col3:
        st.metric("Features", len(df.columns)-1, help="Number of features used for prediction")
    
    st.markdown("## 🤖 Model Performance Summary")
    
    # Create performance comparison
    performance_df = pd.DataFrame(results).T
    performance_df = performance_df[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']]
    
    st.dataframe(performance_df.round(4))
    
    # Best model highlight
    best_model = performance_df['F1 Score'].idxmax()
    best_f1 = performance_df.loc[best_model, 'F1 Score']
    
    st.success(f"🏆 **Best Model**: {best_model} (F1 Score: {best_f1:.4f})")
    
    # Quick insights
    st.markdown("## 💡 Quick Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn by contract type
        fig = px.histogram(df, x='Contract', color='Churn', barmode='group',
                          title="Churn by Contract Type")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly charges distribution
        fig = px.box(df, x='Churn', y='MonthlyCharges',
                    title="Monthly Charges vs Churn")
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Single Customer Prediction":
    st.markdown("## 🔍 Single Customer Prediction")
    st.markdown("Enter customer details to predict churn probability")
    
    # Input form
    with st.form("customer_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ['Male', 'Female'])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: 'Yes' if x else 'No')
            partner = st.selectbox("Partner", ['Yes', 'No'])
            dependents = st.selectbox("Dependents", ['Yes', 'No'])
        
        with col2:
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
            internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
            contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        
        with col3:
            monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 65.0)
            total_charges = st.slider("Total Charges ($)", 20.0, 8500.0, 1500.0)
            
            # Additional services
            online_security = st.selectbox("Online Security", ['Yes', 'No'])
            tech_support = st.selectbox("Tech Support", ['Yes', 'No'])
        
        # Additional service columns (simplified)
        online_backup = 'No'
        device_protection = 'No' 
        streaming_tv = 'No'
        streaming_movies = 'No'
        
        submitted = st.form_submit_button("🔮 Predict Churn", type="primary")
        
        if submitted:
            customer_data = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Make prediction
            predictions, probabilities = predict_single_customer(models, processing, scaler, customer_data, label_encoder)
            
            # Display results
            st.markdown("## 🎯 Prediction Results")
            
            for model_name in models.keys():
                pred = predictions[model_name]
                prob = probabilities[model_name]
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if pred == 'Yes':
                        st.markdown(f'<div class="prediction-box churn-yes">❌ WILL CHURN</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="prediction-box churn-no">✅ WILL STAY</div>', 
                                  unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**{model_name}**")
                    st.progress(prob)
                    st.markdown(f"Churn Probability: {prob:.1%}")

elif page == "📈 Data Insights":
    st.markdown("## 📈 Data Insights & Exploration")
    
    # Dataset overview
    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Churned Customers", (df['Churn'] == 'Yes').sum())
    with col3:
        st.metric("Retained Customers", (df['Churn'] == 'No').sum())
    with col4:
        st.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean():.1%}")
    
    # Interactive charts
    chart_option = st.selectbox("Choose Analysis:", [
        "Churn by Demographics",
        "Churn by Services", 
        "Financial Analysis",
        "Correlation Matrix"
    ])
    
    if chart_option == "Churn by Demographics":
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='gender', color='Churn', barmode='group',
                             title="Churn by Gender")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='SeniorCitizen', color='Churn', barmode='group',
                             title="Churn by Senior Citizen Status")
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_option == "Churn by Services":
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='InternetService', color='Churn', barmode='group',
                             title="Churn by Internet Service")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='Contract', color='Churn', barmode='group',
                             title="Churn by Contract Type")
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_option == "Financial Analysis":
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df, x='Churn', y='MonthlyCharges',
                        title="Monthly Charges Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x='tenure', y='TotalCharges', color='Churn',
                           title="Tenure vs Total Charges")
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_option == "Correlation Matrix":
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance Analysis")
    
    # Performance comparison chart
    performance_df = pd.DataFrame(results).T
    performance_df = performance_df[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']]
    
    # Melt for plotting
    performance_melted = performance_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
    performance_melted.rename(columns={'index': 'Model'}, inplace=True)
    
    fig = px.bar(performance_melted, x='Model', y='Score', color='Metric', barmode='group',
                title="Model Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.markdown("### Detailed Performance Metrics")
    st.dataframe(performance_df.round(4))
    
    # Model comparison insights
    best_accuracy = performance_df['Accuracy'].idxmax()
    best_precision = performance_df['Precision'].idxmax()
    best_recall = performance_df['Recall'].idxmax()
    best_f1 = performance_df['F1 Score'].idxmax()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏆 Best Performers")
        st.write(f"**Accuracy**: {best_accuracy}")
        st.write(f"**Precision**: {best_precision}")
        st.write(f"**Recall**: {best_recall}")
        st.write(f"**F1 Score**: {best_f1}")
    
    with col2:
        st.markdown("### 💡 Recommendations")
        st.write("• **For Production**: Use the model with highest F1 Score")
        st.write("• **For High Precision**: Choose model with best Precision")
        st.write("• **To Catch All Churners**: Choose model with best Recall")

elif page == "📋 Batch Predictions":
    st.markdown("## 📋 Batch Predictions")
    st.markdown("Upload a CSV file to get churn predictions for multiple customers")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.markdown("### Uploaded Data Preview")
        st.dataframe(batch_df.head())
        
        if st.button("Generate Predictions"):
            # Use the best model for batch predictions
            best_model_name = pd.DataFrame(results).T['F1 Score'].idxmax()
            best_model = models[best_model_name]
            
            try:
                # Preprocess batch data
                batch_processed = processing.transform(batch_df)
                
                if best_model_name == 'Logistic Regression':
                    batch_scaled = scaler.transform(batch_processed)
                    predictions = best_model.predict(batch_scaled)
                    probabilities = best_model.predict_proba(batch_scaled)[:, 1]
                elif best_model_name == 'XGBoost':
                    predictions_encoded = best_model.predict(batch_processed)
                    predictions = label_encoder.inverse_transform(predictions_encoded)
                    probabilities = best_model.predict_proba(batch_processed)[:, 1]
                else:  # Random Forest
                    predictions = best_model.predict(batch_processed)
                    probabilities = best_model.predict_proba(batch_processed)[:, 1]
                
                # Add predictions to dataframe
                results_df = batch_df.copy()
                results_df['Predicted_Churn'] = predictions
                results_df['Churn_Probability'] = probabilities
                results_df['Risk_Level'] = pd.cut(probabilities, 
                                                bins=[0, 0.3, 0.7, 1.0], 
                                                labels=['Low', 'Medium', 'High'])
                
                st.markdown("### Prediction Results")
                st.dataframe(results_df)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Risk Customers", (results_df['Risk_Level'] == 'High').sum())
                with col2:
                    st.metric("Medium Risk Customers", (results_df['Risk_Level'] == 'Medium').sum())
                with col3:
                    st.metric("Low Risk Customers", (results_df['Risk_Level'] == 'Low').sum())
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name='churn_predictions.csv',
                    mime='text/csv'
                )
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.write("Please ensure your CSV has the same columns as the training data.")

# Footer
st.markdown("---")
st.markdown("### 🚀 How to Run This App")
st.markdown("""
1. **Install Streamlit**: `pip install streamlit`
2. **Save this code** as `churn_app.py`
3. **Run the app**: `streamlit run churn_app.py`
4. **Open in browser**: Usually at `http://localhost:8501`
""")
