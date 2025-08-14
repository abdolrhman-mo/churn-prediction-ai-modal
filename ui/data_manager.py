import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ChurnDataManager:
    """Manages data loading, preprocessing, and model training for churn prediction"""
    
    def __init__(self, data_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
        self.data_path = data_path
        self.df = None
        self.svm_model = None
        self.scaler = None
        self.processing = None
        self.threshold_results = None
        self.svm_probs = None
        self.y_test = None
        self.X_test_scaled = None
        
    def load_and_process_data(self):
        """Load and preprocess the churn data following the exact pipeline"""
        try:
            # Try to load actual data
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded actual data: {len(self.df)} rows, {len(self.df.columns)} columns")
        except FileNotFoundError:
            print("⚠️ CSV file not found. Creating demo data for showcase.")
            self.df = self._create_demo_data()
        
        # Follow exact preprocessing pipeline
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        self.df = self.df.dropna()
        self.df = self.df.drop(['customerID'], axis=1)
        
        print(f"✅ Data preprocessing completed: {len(self.df)} rows after cleaning")
        return self.df
    
    def _create_demo_data(self):
        """Create realistic demo data for showcase purposes"""
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
        
        return df
    
    def load_or_train_model(self):
        """Try to load saved model first, fall back to training if not available"""
        print("🔍 Checking for saved models...")
        
        if self.load_saved_model():
            print("✅ Using saved model from models/ folder")
            return self._get_training_results()
        else:
            print("⚠️ No saved model found. Training new model...")
            return self.train_svm_model()
    
    def train_svm_model(self):
        """Train SVM model following the exact pipeline and find optimal threshold"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_process_data() first.")
        
        print("🔄 Training SVM model...")
        
        # Prepare data exactly like the original code
        X = self.df.drop('Churn', axis=1)
        y = self.df['Churn']
        
        # Column classification (original logic)
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
        
        # Original preprocessing pipeline
        num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])
        onehot_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('one_hot_encoding', OneHotEncoder())
        ])
        binary_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('binary_encoding', OrdinalEncoder())
        ])
        
        self.processing = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('onehot', onehot_pipeline, onehot_cols),
            ('ordinal', binary_pipeline, binary_cols)
        ], remainder='passthrough')
        
        # Train-test split (original settings)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Process data
        X_train_cleaned = self.processing.fit_transform(X_train)
        X_test_cleaned = self.processing.transform(X_test)
        
        # SMOTE resampling (original approach)
        smote = SMOTE(random_state=12)
        X_resampled, y_resampled = smote.fit_resample(X_train_cleaned, y_train)
        
        # Scaling (original approach)
        self.scaler = StandardScaler()
        X_resampled_scaled = self.scaler.fit_transform(X_resampled)
        X_test_scaled = self.scaler.transform(X_test_cleaned)
        
        # Train SVM (original settings)
        self.svm_model = SVC(kernel='linear', random_state=42, probability=True)
        self.svm_model.fit(X_resampled_scaled, y_resampled)
        
        # Threshold optimization (original logic)
        self.svm_probs = self.svm_model.predict_proba(X_test_scaled)[:, 1]
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        self.threshold_results = []
        for threshold in thresholds:
            predictions = (self.svm_probs >= threshold).astype(int)
            pred_labels = ['Yes' if pred == 1 else 'No' for pred in predictions]
            
            recall = recall_score(y_test, pred_labels, pos_label='Yes')
            precision = precision_score(y_test, pred_labels, pos_label='Yes')
            f1 = f1_score(y_test, pred_labels, pos_label='Yes')
            accuracy = accuracy_score(y_test, pred_labels)
            
            self.threshold_results.append({
                'Threshold': threshold,
                'Recall': recall,
                'Precision': precision,
                'F1_Score': f1,
                'Accuracy': accuracy
            })
        
        self.y_test = y_test
        self.X_test_scaled = X_test_scaled
        
        print("✅ SVM model training completed!")
        return self._get_training_results()
    
    def _get_training_results(self):
        """Return all training results for the main app"""
        # If we have saved model but no test data, generate demo predictions
        if (self.svm_model is not None and 
            self.threshold_results is None and 
            self.df is not None):
            self._generate_demo_predictions()
        
        return (
            self.svm_model, 
            self.scaler, 
            self.processing, 
            self.threshold_results, 
            self.svm_probs, 
            self.y_test, 
            self.X_test_scaled
        )
    
    def _generate_demo_predictions(self):
        """Generate demo predictions when using saved models"""
        print("🔄 Generating demo predictions for saved model...")
        
        # Use a small sample of the data for demo predictions
        sample_size = min(1000, len(self.df))
        sample_df = self.df.sample(n=sample_size, random_state=42)
        
        X_sample = sample_df.drop('Churn', axis=1)
        y_sample = sample_df['Churn']
        
        # Process the sample data
        X_sample_cleaned = self.processing.transform(X_sample)
        X_sample_scaled = self.scaler.transform(X_sample_cleaned)
        
        # Get predictions
        self.svm_probs = self.svm_model.predict_proba(X_sample_scaled)[:, 1]
        self.y_test = y_sample
        self.X_test_scaled = X_sample_scaled
        
        # Generate threshold results
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        self.threshold_results = []
        for threshold in thresholds:
            predictions = (self.svm_probs >= threshold).astype(int)
            pred_labels = ['Yes' if pred == 1 else 'No' for pred in predictions]
            
            recall = recall_score(y_sample, pred_labels, pos_label='Yes')
            precision = precision_score(y_sample, pred_labels, pos_label='Yes')
            f1 = f1_score(y_sample, pred_labels, pos_label='Yes')
            accuracy = accuracy_score(y_sample, pred_labels)
            
            self.threshold_results.append({
                'Threshold': threshold,
                'Recall': recall,
                'Precision': precision,
                'F1_Score': f1,
                'Accuracy': accuracy
            })
        
        print("✅ Demo predictions generated successfully!")
    
    def get_best_threshold(self):
        """Get the threshold that achieves maximum recall"""
        if self.threshold_results is None:
            raise ValueError("Model not trained. Call train_svm_model() first.")
        
        best_idx = max(range(len(self.threshold_results)), 
                      key=lambda i: self.threshold_results[i]['Recall'])
        return self.threshold_results[best_idx]
    
    def get_data_summary(self):
        """Get summary statistics about the dataset"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_process_data() first.")
        
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'churn_rate': (self.df['Churn'] == 'Yes').mean(),
            'numerical_features': len(self.df.select_dtypes(include=['int64', 'float64']).columns),
            'categorical_features': len(self.df.select_dtypes(include=['object']).columns) - 1,  # -1 for target
            'missing_values': self.df.isnull().sum().sum()
        }
        
        return summary
    
    def save_model(self, models_dir="models"):
        """Save the trained model and related components"""
        if self.svm_model is None:
            raise ValueError("Model not trained. Call train_svm_model() first.")
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Save the SVM model
        svm_model_path = f"{models_dir}/svm_model.pkl"
        joblib.dump(self.svm_model, svm_model_path)
        print(f"✅ SVM model saved to: {svm_model_path}")
        
        # Save the scaler
        scaler_path = f"{models_dir}/scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler saved to: {scaler_path}")
        
        # Save the preprocessing pipeline
        processing_path = f"{models_dir}/preprocessing.pkl"
        joblib.dump(self.processing, processing_path)
        print(f"✅ Preprocessing pipeline saved to: {processing_path}")
        
        # Save the best threshold
        best_threshold = self.get_best_threshold()
        threshold_path = f"{models_dir}/svm_threshold.txt"
        with open(threshold_path, 'w') as f:
            f.write(str(best_threshold['Threshold']))
        print(f"✅ Best threshold saved to: {threshold_path}")
        
        return {
            'svm_model': svm_model_path,
            'scaler': scaler_path,
            'preprocessing': processing_path,
            'threshold': threshold_path
        }
    
    def load_saved_model(self, models_dir="models"):
        """Load previously saved model components"""
        try:
            # Load the saved components
            svm_model_path = f"{models_dir}/svm_model.pkl"
            scaler_path = f"{models_dir}/scaler.pkl"
            processing_path = f"{models_dir}/preprocessing.pkl"
            threshold_path = f"{models_dir}/svm_threshold.txt"
            
            if not all(os.path.exists(p) for p in [svm_model_path, scaler_path, processing_path, threshold_path]):
                raise FileNotFoundError("Some model files are missing")
            
            self.svm_model = joblib.load(svm_model_path)
            self.scaler = joblib.load(scaler_path)
            self.processing = joblib.load(processing_path)
            
            with open(threshold_path, 'r') as f:
                best_threshold = float(f.read().strip())
            
            print("✅ Saved model components loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading saved model: {e}")
            return False
    
    def predict_churn(self, customer_data):
        """Make churn predictions on new customer data"""
        if self.svm_model is None:
            raise ValueError("Model not trained. Call train_svm_model() or load_saved_model() first.")
        
        # Preprocess the customer data
        customer_processed = self.processing.transform(customer_data)
        customer_scaled = self.scaler.transform(customer_processed)
        
        # Get prediction probabilities
        churn_prob = self.svm_model.predict_proba(customer_scaled)[:, 1][0]
        
        # Get the best threshold
        best_threshold = self.get_best_threshold()['Threshold']
        
        # Make prediction
        prediction = 'Yes' if churn_prob >= best_threshold else 'No'
        
        return {
            'prediction': prediction,
            'churn_probability': churn_prob,
            'threshold_used': best_threshold,
            'risk_level': 'High' if churn_prob >= best_threshold else 'Low'
        }

# Convenience functions for the main app
def load_and_process_data(data_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """Convenience function to load and process data"""
    manager = ChurnDataManager(data_path)
    return manager.load_and_process_data()

def train_svm_model(df):
    """Convenience function to train SVM model (now tries to load saved model first)"""
    manager = ChurnDataManager()
    manager.df = df
    return manager.load_or_train_model()

def load_saved_model_only(df):
    """Convenience function to only load saved model (no training fallback)"""
    manager = ChurnDataManager()
    manager.df = df
    if manager.load_saved_model():
        return manager._get_training_results()
    else:
        raise FileNotFoundError("No saved model found in models/ folder")
