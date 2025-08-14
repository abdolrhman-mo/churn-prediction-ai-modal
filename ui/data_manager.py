import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

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
