import streamlit as st
import pandas as pd

def render_technical_implementation_page(df):
    """Render the Technical Implementation page"""
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
        st.markdown("#### Model Persistence & Loading")
        st.code('''
# Save the trained model
joblib.dump(svm_model, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(processing, "models/preprocessing.pkl")

# Save optimal threshold
with open("models/svm_threshold.txt", 'w') as f:
    f.write(str(best_threshold))

# Load saved models (faster than training)
manager = ChurnDataManager()
if manager.load_saved_model():
    print("✅ Using saved model - Fast loading!")
else:
    print("⚠️ No saved model found")
        ''', language='python')
        
        # Add information about saved models
        st.markdown("#### 💾 Saved Models in models/ Folder")
        st.markdown("""
        The dashboard now automatically checks for saved models in the `models/` folder:
        
        - **`svm_model.pkl`**: Trained SVM classifier
        - **`scaler.pkl`**: StandardScaler for feature normalization
        - **`preprocessing.pkl`**: Complete preprocessing pipeline
        - **`svm_threshold.txt`**: Optimal threshold for maximum recall
        
        **Benefits:**
        - ⚡ **Fast loading**: No need to retrain every time
        - 🔄 **Consistent results**: Same model performance across sessions
        - 💾 **Production ready**: Models can be deployed without retraining
        - 📊 **Demo predictions**: Generated automatically from saved models
        """)
    
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
