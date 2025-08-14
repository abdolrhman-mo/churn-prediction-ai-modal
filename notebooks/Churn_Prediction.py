#!/usr/bin/env python
# coding: utf-8

# ## **Phase 1: Data Loading & Initial Exploration**

# In[208]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek


# In[209]:


# Load dataset
# Note: df is short for dataframe for pandas dataframe
df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# See the first 5 rows
df.head()


# In[210]:


# See how many rows and columns
df.shape


# In[211]:


# See column names and types
df.info()

# Notes:
# Dtype => Data Type
# object = Text (strings) or mixed data. Pandas uses object for things that aren’t purely numbers.
# Non-Null Count => Number of non-null values in the column


# ## **Phase 2: Data Quality Assessment**

# In[212]:


#  Count missing values in each column
df.isnull().sum()


# In[213]:


# Check for duplicate customers
df.duplicated().sum()


# In[214]:


# See current data types
df.dtypes

# Conclusion:
# - TotalCharges is object, not float


# ## **Phase 3: Data Cleaning**

# In[215]:


# Convert 'TotalCharges' to numeric, some rows may be blank
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Note: errors='coerce' means that if the conversion fails, the value will be set to NaN

# Now check again for missing values
df.isnull().sum()

# Drop rows with missing TotalCharges
df = df.dropna()

# Check shapes
df.shape


# In[216]:


df.dtypes


# ## **Phase 4: Feature Engineering**

# In[217]:


# # CHUNK 1: Basic Data Exploration
# # Purpose: See what columns you have and their types

# print("=== YOUR DATASET OVERVIEW ===")
# print(f"Shape: {df.shape}")
# print(f"Columns: {df.columns.tolist()}")
# print()

# print("=== COLUMN TYPES ===")
# numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
# categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
# print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
# print()

# print("=== QUICK PREVIEW ===")
# print(df.head(3))


# In[218]:


# # CHUNK 2: Find What Drives Churn
# # Purpose: Discover which existing features matter most

# print("=== CHURN RATES BY CATEGORY ===")
# print("This shows you which categories have highest churn - these are GOLDMINES for features!")
# print()

# # Check categorical columns for churn patterns
# for col in categorical_cols:
#     if col in df.columns:
#         churn_rate = df.groupby(col)['Churn'].value_counts(normalize=True).unstack()
#         churn_rate['Churn_Rate'] = churn_rate['Yes'] / (churn_rate['Yes'] + churn_rate['No'])

#         print(f"\n{col} - Churn Rates:")
#         print(churn_rate['Churn_Rate'].sort_values(ascending=False).round(3))

#         # Highlight high-risk categories
#         high_risk = churn_rate['Churn_Rate'] > 0.4
#         if high_risk.any():
#             print(f"🚨 HIGH RISK categories in {col}: {high_risk[high_risk].index.tolist()}")


# In[219]:


# # CHUNK 3: See How Money Affects Churn
# # Purpose: Financial patterns often drive customer behavior

# print("=== MONEY vs CHURN ANALYSIS ===")

# # Plot 1: Monthly Charges vs Churn
# plt.figure(figsize=(12, 4))

# plt.subplot(1, 3, 1)
# sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
# plt.title('Monthly Charges vs Churn')
# plt.ylabel('Monthly Charges ($)')

# # Plot 2: Tenure vs Churn  
# plt.subplot(1, 3, 2)
# sns.boxplot(data=df, x='Churn', y='tenure')
# plt.title('Tenure vs Churn')
# plt.ylabel('Months with Company')

# # Plot 3: Total Charges vs Churn
# plt.subplot(1, 3, 3)
# sns.boxplot(data=df, x='Churn', y='TotalCharges')
# plt.title('Total Charges vs Churn')
# plt.ylabel('Total Charges ($)')

# plt.tight_layout()
# plt.show()

# # Show the actual numbers
# print("\n=== NUMERICAL INSIGHTS ===")
# for col in ['MonthlyCharges', 'tenure', 'TotalCharges']:
#     if col in df.columns:
#         churn_stats = df.groupby('Churn')[col].agg(['mean', 'median']).round(2)
#         print(f"\n{col}:")
#         print(churn_stats)

#         # Feature idea generator
#         churned_avg = churn_stats.loc['Yes', 'mean']
#         stayed_avg = churn_stats.loc['No', 'mean']

#         if churned_avg > stayed_avg:
#             print(f"💡 FEATURE IDEA: Create 'high_{col.lower()}' flag for values > {churned_avg:.0f}")
#         else:
#             print(f"💡 FEATURE IDEA: Create 'low_{col.lower()}' flag for values < {churned_avg:.0f}")


# In[220]:


# # CHUNK 4: Create Your First Feature - Service Count
# # Purpose: Customers with more services are usually stickier (less likely to churn)

# print("=== CREATING SERVICE COUNT FEATURE ===")
# print("Theory: More services = harder to leave = lower churn")
# print()

# # Step 1: Find all service columns
# service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
#                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# # Step 2: Check what values these columns have
# print("Service column values:")
# for col in service_cols[:3]:  # Just show first 3 for brevity
#     if col in df.columns:
#         print(f"{col}: {df[col].unique()}")
# print()

# # Step 3: Convert to 1/0 format for counting
# print("Converting services to 1/0...")
# df_temp = df.copy()  # Work on copy first

# service_count = 0
# for col in service_cols:
#     if col in df.columns:
#         if col == 'InternetService':
#             # Special case: 'No' = 0, any internet service = 1
#             df_temp[col + '_binary'] = (df_temp[col] != 'No').astype(int)
#         else:
#             # Regular case: 'Yes' = 1, 'No' = 0
#             df_temp[col + '_binary'] = (df_temp[col] == 'Yes').astype(int)

#         service_count += df_temp[col + '_binary']

# # Step 4: Create the feature
# df_temp['total_services'] = service_count

# # Step 5: Test if it works
# print("✅ Feature created! Sample results:")
# print(df_temp[['total_services']].head())
# print()
# print("Distribution of service counts:")
# print(df_temp['total_services'].value_counts().sort_index())
# print()

# # Step 6: Validate the theory - does it predict churn?
# print("=== TESTING THE THEORY ===")
# churn_by_services = df_temp.groupby('total_services')['Churn'].value_counts(normalize=True).unstack()
# churn_by_services['Churn_Rate'] = churn_by_services['Yes'] / (churn_by_services['Yes'] + churn_by_services['No'])

# print("Churn rate by number of services:")
# print(churn_by_services['Churn_Rate'].round(3))

# if churn_by_services['Churn_Rate'].iloc[0] > churn_by_services['Churn_Rate'].iloc[-1]:
#     print("✅ THEORY CONFIRMED: Fewer services = higher churn!")
# else:
#     print("❌ Theory not confirmed - investigate further")


# In[221]:


# # CHUNK 5: Create Risk Profile Feature
# # Purpose: Combine multiple risky factors into one powerful feature

# print("=== CREATING RISK PROFILE FEATURE ===")
# print("Theory: Month-to-month + New customer + High charges = VERY RISKY")
# print()

# # Step 1: Define what makes a customer risky
# print("Risk factors:")
# print("1. Contract = Month-to-month (no commitment)")
# print("2. tenure <= 12 months (new customer)")
# print("3. MonthlyCharges > 70 (expensive)")
# print()

# # Step 2: Check each risk factor individually
# df_temp = df.copy()

# risk1 = df_temp['Contract'] == 'Month-to-month'
# risk2 = df_temp['tenure'] <= 12
# risk3 = df_temp['MonthlyCharges'] > 70

# print("Individual risk factor churn rates:")
# print(f"Month-to-month: {df_temp[risk1]['Churn'].value_counts(normalize=True).get('Yes', 0):.3f}")
# print(f"New customer: {df_temp[risk2]['Churn'].value_counts(normalize=True).get('Yes', 0):.3f}")
# print(f"High charges: {df_temp[risk3]['Churn'].value_counts(normalize=True).get('Yes', 0):.3f}")
# print()

# # Step 3: Combine all risk factors
# df_temp['high_risk_profile'] = (risk1 & risk2 & risk3).astype(int)

# print("=== RISK PROFILE RESULTS ===")
# print(f"High risk customers: {df_temp['high_risk_profile'].sum()} out of {len(df_temp)}")
# print(f"Percentage: {df_temp['high_risk_profile'].mean()*100:.1f}%")
# print()

# # Step 4: Test the power of this feature
# risk_churn = df_temp.groupby('high_risk_profile')['Churn'].value_counts(normalize=True).unstack()
# risk_churn['Churn_Rate'] = risk_churn['Yes'] / (risk_churn['Yes'] + risk_churn['No'])

# print("Churn rates:")
# print(f"Normal customers (0): {risk_churn.loc[0, 'Churn_Rate']:.3f}")
# print(f"High risk customers (1): {risk_churn.loc[1, 'Churn_Rate']:.3f}")

# improvement = risk_churn.loc[1, 'Churn_Rate'] / risk_churn.loc[0, 'Churn_Rate']
# print(f"\n🎯 HIGH RISK CUSTOMERS ARE {improvement:.1f}x MORE LIKELY TO CHURN!")

# if improvement > 2:
#     print("✅ EXCELLENT FEATURE! This will help your model a lot!")
# elif improvement > 1.5:
#     print("✅ Good feature! This should improve your model.")
# else:
#     print("⚠️ Weak feature - try different combinations.")


# In[222]:


# # CHUNK 6: Create Value Perception Features
# # Purpose: Calculate ratios that show if customers feel they get good value

# print("=== CREATING VALUE-BASED FEATURES ===")
# print("Theory: Customers leave when they feel they're not getting good value for money")
# print()

# df_temp = df.copy()

# # Feature 1: Monthly charges per tenure month
# print("Feature 1: Average monthly spend over customer lifetime")
# df_temp['avg_monthly_spend'] = df_temp['TotalCharges'] / (df_temp['tenure'] + 1)  # +1 to avoid division by 0

# print("Sample values:")
# print(df_temp[['MonthlyCharges', 'TotalCharges', 'tenure', 'avg_monthly_spend']].head())
# print()

# # Feature 2: Charges per service (if we have service count)
# if 'total_services' not in df_temp.columns:
#     # Quick service count for this example
#     service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup']
#     service_count = 0
#     for col in service_cols:
#         if col in df.columns:
#             if col == 'InternetService':
#                 service_count += (df[col] != 'No').astype(int)
#             else:
#                 service_count += (df[col] == 'Yes').astype(int)
#     df_temp['total_services'] = service_count

# print("Feature 2: Price per service")
# df_temp['price_per_service'] = df_temp['MonthlyCharges'] / (df_temp['total_services'] + 1)

# print("Sample values:")
# print(df_temp[['MonthlyCharges', 'total_services', 'price_per_service']].head())
# print()

# # Feature 3: Value bins
# print("Feature 3: Value perception categories")
# df_temp['value_perception'] = pd.cut(df_temp['price_per_service'], 
#                                    bins=[0, 15, 25, 40, 1000],
#                                    labels=['Great_Value', 'Good_Value', 'Fair_Value', 'Poor_Value'])

# print("Value perception distribution:")
# print(df_temp['value_perception'].value_counts())
# print()

# # Test these features
# print("=== TESTING VALUE FEATURES ===")

# # Test avg_monthly_spend
# print("1. Average monthly spend vs churn:")
# spend_churn = df_temp.groupby('Churn')['avg_monthly_spend'].mean()
# print(f"Churned customers: ${spend_churn['Yes']:.2f}/month")
# print(f"Stayed customers: ${spend_churn['No']:.2f}/month")
# print()

# # Test value perception
# print("2. Value perception vs churn:")
# value_churn = df_temp.groupby('value_perception')['Churn'].value_counts(normalize=True).unstack()
# if 'Yes' in value_churn.columns:
#     value_churn['Churn_Rate'] = value_churn['Yes'] / (value_churn['Yes'] + value_churn['No'])
#     print("Churn rates by value perception:")
#     print(value_churn['Churn_Rate'].round(3))

#     if value_churn['Churn_Rate'].iloc[-1] > value_churn['Churn_Rate'].iloc[0]:
#         print("✅ CONFIRMED: Poor value perception leads to higher churn!")
#     else:
#         print("❌ Hypothesis not confirmed")


# In[223]:


# # ## **Phase 4: Feature Engineering**

# def create_churn_features(df):
#     """Create comprehensive features for churn prediction"""
#     df = df.copy()

#     # 1. Tenure-based features
#     df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
#     df['is_long_term'] = (df['tenure'] >= 24).astype(int)
#     df['monthly_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
#     df['total_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)

#     # 2. Service usage
#     # First, let's check what values we actually have
#     service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
#                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

#     # Handle different types of values in service columns
#     df_services = df[service_cols].copy()
#     for col in service_cols:
#         if col == 'InternetService':
#             # InternetService has 'DSL', 'Fiber optic', 'No' - treat 'No' as 0, others as 1
#             df_services[col] = (df_services[col] != 'No').astype(int)
#         else:
#             # Other columns should be Yes/No, but let's be safe
#             df_services[col] = df_services[col].replace({'Yes': 1, 'No': 0})
#             # Handle any unexpected values
#             df_services[col] = pd.to_numeric(df_services[col], errors='coerce').fillna(0)

#     df['total_services'] = df_services.sum(axis=1)
#     df['service_utilization'] = df['total_services'] / 8

#     # 3. Financial features
#     df['is_high_spender'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
#     df['charges_per_service'] = df['MonthlyCharges'] / (df['total_services'] + 1)

#     # 4. Risk combinations
#     df['high_risk'] = (
#         (df['Contract'] == 'Month-to-month') & 
#         (df['tenure'] <= 12) & 
#         (df['MonthlyCharges'] > 65)
#     ).astype(int)

#     # 5. Satisfaction proxies
#     df['likely_satisfied'] = (
#         (df['Contract'] != 'Month-to-month') &
#         (df['TechSupport'] == 'Yes')
#     ).astype(int)

#     # 6. Binned features
#     df['tenure_group'] = pd.cut(df['tenure'], 
#                                bins=[0, 6, 12, 24, 100], 
#                                labels=['New', 'Growing', 'Mature', 'Veteran'])

#     df['charges_group'] = pd.cut(df['MonthlyCharges'],
#                                 bins=[0, 35, 55, 75, 120],
#                                 labels=['Budget', 'Standard', 'Premium', 'Enterprise'])

#     return df

# # Apply feature engineering
# print("Original dataset shape:", df.shape)
# df = create_churn_features(df)
# print("Dataset shape after feature engineering:", df.shape)
# print("New features created!")

# # Check some of the new features
# print("\nSample of new features:")
# new_features = ['is_new_customer', 'total_services', 'high_risk', 'likely_satisfied']
# print(df[new_features].head())


# ## **Phase 5: Data Visualization & Exploration**

# In[224]:


# Plot churn count
sns.countplot(x='Churn', data=df)
plt.title("Churn Count")

# Plot churn by gender
sns.countplot(x='gender', hue='Churn', data=df)
plt.title("Churn by Gender")

# Plot churn by Contract type
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract Type")


# ## **Phase 6: Data Preprocessing**

# In[225]:


# Drop customerID (not useful)
df.drop(['customerID'], axis=1, inplace=True)


# In[226]:


num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(num_cols)
print(categorical_cols)


# In[227]:


# Dynamic column classification based on data types and unique values
feature_cols = [col for col in categorical_cols if col != 'Churn']
target_col = 'Churn'

# Binary columns: columns with only 2 unique values (Yes/No)
binary_cols = []
for col in feature_cols:
    unique_vals = df[col].nunique()
    if unique_vals == 2:
        binary_cols.append(col)

# One-hot columns: columns with more than 2 unique values
onehot_cols = []
for col in feature_cols:
    unique_vals = df[col].nunique()
    if unique_vals > 2:
        onehot_cols.append(col)

print("Binary columns (2 unique values):", binary_cols)
print("One-hot columns (>2 unique values):", onehot_cols)


# In[228]:


# Numerical pipeline
num_pipeline = Pipeline([ # pipeline means create a step-by-step process
    ('imputer', SimpleImputer(strategy='median')) # replace missing num with middle value
])

# OneHot pipeline (for nominal categorical features)
onehot_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot_encoding', OneHotEncoder())
])

# Binary/Ordinal pipeline
binary_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('binary_encoding', OrdinalEncoder())
])

# Master controller that decides which columns go to which processing station
processing = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('onehot', onehot_pipeline, onehot_cols),
    ('ordinal', binary_pipeline, binary_cols)
], remainder='passthrough')

processing


# In[229]:


# Separate What You Want to Predict
X = df.drop('Churn', axis=1)  # Everything EXCEPT 'Churn' column
y = df['Churn']               # ONLY the 'Churn' column

# Note:
# Why X and y?
# This comes from math notation: y = f(X) means "y depends on X"
# X: Independent variables (customer info)
# y: Dependent variable (churn yes/no)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Clean the data
X_train_cleaned = processing.fit_transform(X_train)
X_test_cleaned = processing.transform(X_test)

# Convert target labels to numerical for XGBoost
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# In[230]:


# Test if data is preprocessed
print("Original shape:", X_train.shape)
print("Processed shape:", X_train_cleaned.shape)

# If OneHot worked, processed data should have MORE columns
# Example: Original (7000, 20) → Processed (7000, 45)


# In[231]:


# Another way to test if data is preprocessed
# Test 1: All numbers?
processed_sample = pd.DataFrame(X_train_cleaned)
print("Any non-numeric values?", processed_sample.select_dtypes(include=['object']).shape[1] > 0)

# Test 2: Can we scale without errors?
try:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X_train_cleaned)
    print("✅ Pipeline worked! Data is ready for ML")
except ValueError as e:
    print(f"❌ Pipeline failed: {e}")


# In[232]:


# Check if your data is imbalanced
print("Churn distribution:")
print(y_train.value_counts())
print("\nPercentages:")
print(y_train.value_counts(normalize=True) * 100)


# In[233]:


# Resampling because the data is imbalanced
smote = SMOTE(random_state=12)
X_resampled, y_resampled = smote.fit_resample(X_train_cleaned, y_train)

# Notes:
# SMOTE = "Synthetic Minority Oversampling TEchnique"
# What it means: "I'm a tool that creates fake examples of rare cases"

# Scaling
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test_cleaned)


# ## **Phase 7: Model Training**

# In[234]:


# Train Logistic Regression model
print("Training Logistic Regression model...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_resampled_scaled, y_resampled)

# Train SVM model  
print("Training SVM model...")

svm_model = SVC(kernel='linear', random_state=42, probability=True)
svm_model.fit(X_resampled_scaled, y_resampled)

# Train XGBoost model
print("Training XGBoost model...")
# Calculate class weight for imbalanced data
pos_weight = (y_train == 'No').sum() / (y_train == 'Yes').sum()
xgb_model = XGBClassifier(
    scale_pos_weight=pos_weight,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_cleaned, y_train_encoded)

# Note: xgb model handle imbalanced data better internally through scale_pos_weight

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Handles imbalance
    random_state=42
)
rf_model.fit(X_train_cleaned, y_train)

# Create voting classifier with your existing models
print("Training Voting Classifier...")
voting_model = VotingClassifier([
    ('lr', lr_model),       # Logistic Regression
    ('svm', svm_model),     # SVM
    ('rf', rf_model)        # Random Forest
    # Note: Excluding XGBoost because it uses different data format
], voting='soft')  # 'soft' uses probabilities, 'hard' uses direct votes
voting_model.fit(X_resampled_scaled, y_resampled)

print("Model training completed!")


# ## **Phase 8: Model Evaluation**

# In[ ]:


# Make predictions
y_pred_lr = lr_model.predict(X_test_scaled)

# SVM predictions
y_pred_svm = svm_model.predict(X_test_scaled)

# XGBoost predictions
y_pred_xgb_encoded = xgb_model.predict(X_test_cleaned)
y_pred_xgb = label_encoder.inverse_transform(y_pred_xgb_encoded)  # Convert back to 'Yes'/'No'

# Random Forest predictions
y_pred_rf = rf_model.predict(X_test_cleaned)

# Voting predictions
y_pred_voting = voting_model.predict(X_test_scaled)


# In[246]:


# Refactored Phase 8: Model Evaluation with Clean, Reusable Code

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_predictions(model, X_test_data, use_encoded=False):
    """Get predictions and probabilities from a model"""
    try:
        predictions = model.predict(X_test_data)
        probabilities = model.predict_proba(X_test_data)[:, 1]
        return predictions, probabilities
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return None, None

def calculate_metrics(y_true, y_pred, probabilities=None):
    """Calculate all evaluation metrics for a model"""
    try:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, pos_label='Yes'),
            'recall': recall_score(y_true, y_pred, pos_label='Yes'),
            'f1': f1_score(y_true, y_pred, pos_label='Yes')
        }
        
        # Add ROC AUC if probabilities are available
        if probabilities is not None:
            try:
                # Handle encoded vs non-encoded labels
                if y_true.iloc[0] in ['Yes', 'No']:
                    # Convert to binary for ROC AUC
                    y_binary = (y_true == 'Yes').astype(int)
                else:
                    y_binary = y_true
                metrics['roc_auc'] = roc_auc_score(y_binary, probabilities)
            except:
                metrics['roc_auc'] = None
        
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

def find_optimal_threshold(probabilities, y_true, model_name):
    """Find the optimal threshold for a model based on recall"""
    print(f"\n🔍 {model_name.upper()} - Threshold Analysis")
    print("-" * 60)
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    print("Threshold | Recall  | Precision | F1-Score | Meaning")
    print("-" * 65)
    
    for threshold in thresholds:
        # Apply threshold
        predictions = (probabilities >= threshold).astype(int)
        pred_labels = ['Yes' if pred == 1 else 'No' for pred in predictions]
        
        # Calculate metrics
        recall = recall_score(y_true, pred_labels, pos_label='Yes')
        precision = precision_score(y_true, pred_labels, pos_label='Yes')
        f1 = f1_score(y_true, pred_labels, pos_label='Yes')
        
        # Explanation
        if threshold <= 0.2:
            meaning = "Extremely aggressive"
        elif threshold <= 0.4:
            meaning = "Very aggressive"
        elif threshold == 0.5:
            meaning = "Default (balanced)"
        elif threshold <= 0.7:
            meaning = "Conservative"
        else:
            meaning = "Very conservative"
        
        print(f"{threshold:^9} | {recall:.4f}  | {precision:.4f}    | {f1:.4f}   | {meaning}")
        results.append((threshold, recall, precision, f1))
    
    # Find best threshold based on recall
    best_recall_idx = max(range(len(results)), key=lambda i: results[i][1])
    best_threshold, best_recall, best_precision, best_f1 = results[best_recall_idx]
    
    print(f"\n🏆 BEST RECALL for {model_name}:")
    print(f"   Threshold: {best_threshold}")
    print(f"   Recall: {best_recall:.4f}")
    print(f"   Precision: {best_precision:.4f}")
    print(f"   F1-Score: {best_f1:.4f}")
    
    return best_threshold, best_recall, best_precision, best_f1

def evaluate_model_comprehensive(model, X_test_data, y_test, model_name, use_encoded=False):
    """Comprehensive evaluation of a single model"""
    print(f"\n{'='*60}")
    print(f"📊 {model_name.upper()} - COMPREHENSIVE EVALUATION")
    print(f"{'='*60}")
    
    # Get predictions and probabilities
    predictions, probabilities = get_model_predictions(model, X_test_data, use_encoded)
    
    if predictions is None:
        print(f"❌ Failed to get predictions for {model_name}")
        return None
    
    # Calculate basic metrics
    metrics = calculate_metrics(y_test, predictions, probabilities)
    
    if metrics is None:
        print(f"❌ Failed to calculate metrics for {model_name}")
        return None
    
    # Display metrics
    print(f"📈 Performance Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
    
    # Find optimal threshold
    if probabilities is not None:
        threshold_results = find_optimal_threshold(probabilities, y_test, model_name)
        metrics['optimal_threshold'] = threshold_results[0]
        metrics['optimal_recall'] = threshold_results[1]
        metrics['optimal_precision'] = threshold_results[2]
        metrics['optimal_f1'] = threshold_results[3]
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, predictions))
    
    return metrics

def compare_all_models(models_config):
    """Compare all models and find the best performer"""
    print(f"\n{'='*80}")
    print("🏆 COMPREHENSIVE MODEL COMPARISON")
    print(f"{'='*80}")
    
    all_results = []
    
    for model_name, config in models_config.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}...")
        
        results = evaluate_model_comprehensive(
            config['model'], 
            config['X_test'], 
            config['y_test'], 
            model_name, 
            config['use_encoded']
        )
        
        if results:
            results['model_name'] = model_name
            all_results.append(results)
    
    # Create comparison summary
    if all_results:
        print(f"\n{'='*80}")
        print("📊 MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Create comparison DataFrame
        comparison_data = []
        for result in all_results:
            comparison_data.append([
                result['model_name'],
                result['accuracy'],
                result['precision'],
                result['recall'],
                result['f1'],
                result.get('roc_auc', 'N/A'),
                result.get('optimal_threshold', 'N/A')
            ])
        
        comparison_df = pd.DataFrame(comparison_data, 
                                   columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Optimal Threshold'])
        print(comparison_df.round(4))
        
        # Find best models by different metrics
        best_f1_idx = comparison_df['F1 Score'].idxmax()
        best_recall_idx = comparison_df['Recall'].idxmax()
        best_accuracy_idx = comparison_df['Accuracy'].idxmax()
        
        print(f"\n🥇 BEST PERFORMING MODELS:")
        print(f"   Best F1 Score: {comparison_df.loc[best_f1_idx, 'Model']} ({comparison_df.loc[best_f1_idx, 'F1 Score']:.4f})")
        print(f"   Best Recall: {comparison_df.loc[best_recall_idx, 'Model']} ({comparison_df.loc[best_recall_idx, 'Recall']:.4f})")
        print(f"   Best Accuracy: {comparison_df.loc[best_accuracy_idx, 'Model']} ({comparison_df.loc[best_accuracy_idx, 'Accuracy']:.4f})")
        
        return comparison_df, all_results
    
    return None, None

def demonstrate_predictions(models_config, sample_idx=0):
    """Demonstrate predictions for a sample customer"""
    print(f"\n{'='*80}")
    print("🔮 SAMPLE PREDICTION DEMONSTRATION")
    print(f"{'='*80}")
    
    # Get sample customer
    sample_original = models_config[list(models_config.keys())[0]]['X_test'].iloc[sample_idx]
    sample_actual = models_config[list(models_config.keys())[0]]['y_test'].iloc[sample_idx]
    
    print(f"Customer Sample (Test Customer #{sample_idx + 1}):")
    print(f"  - Contract: {sample_original.get('Contract', 'N/A')}")
    print(f"  - Monthly Charges: ${sample_original.get('MonthlyCharges', 'N/A'):.2f}")
    print(f"  - Tenure: {sample_original.get('tenure', 'N/A')} months")
    print(f"  - Internet Service: {sample_original.get('InternetService', 'N/A')}")
    print(f"  - Actual Churn: {sample_actual}")
    
    # Get predictions from all models
    print(f"\n📊 Predictions from All Models:")
    print("-" * 50)
    
    for model_name, config in models_config.items():
        predictions, probabilities = get_model_predictions(config['model'], config['X_test'], config['use_encoded'])
        
        if predictions is not None and probabilities is not None:
            pred_label = predictions[sample_idx]
            prob = probabilities[sample_idx]
            
            print(f"  {model_name}:")
            print(f"    Prediction: {pred_label}")
            print(f"    Churn Probability: {prob:.3f}")
            print(f"    Stay Probability: {1-prob:.3f}")
            print()

def generate_business_recommendations(comparison_df, all_results):
    """Generate business recommendations based on model performance"""
    print(f"\n{'='*80}")
    print("💡 BUSINESS RECOMMENDATIONS & INSIGHTS")
    print(f"{'='*80}")
    
    if comparison_df is None or all_results is None:
        print("❌ No results available for recommendations")
        return
    
    # Find best overall model
    best_f1_idx = comparison_df['F1 Score'].idxmax()
    best_model = comparison_df.loc[best_f1_idx, 'Model']
    best_f1 = comparison_df.loc[best_f1_idx, 'F1 Score']
    
    print(f"🎯 RECOMMENDED PRODUCTION MODEL:")
    print(f"   {best_model} with F1 Score: {best_f1:.4f}")
    
    # Performance analysis
    print(f"\n📈 PERFORMANCE ANALYSIS:")
    print(f"   Overall accuracy range: {comparison_df['Accuracy'].min():.1%} - {comparison_df['Accuracy'].max():.1%}")
    print(f"   Best F1 Score: {best_f1:.4f} ({best_model})")
    print(f"   Performance level: {'Excellent' if best_f1 > 0.8 else 'Good' if best_f1 > 0.7 else 'Moderate' if best_f1 > 0.6 else 'Needs Improvement'}")
    
    # Threshold strategy recommendations
    print(f"\n🎯 THRESHOLD STRATEGY RECOMMENDATIONS:")
    
    # Find best threshold results
    best_threshold_results = None
    for result in all_results:
        if result['model_name'] == best_model and 'optimal_threshold' in result:
            best_threshold_results = result
            break
    
    if best_threshold_results:
        optimal_threshold = best_threshold_results['optimal_threshold']
        optimal_recall = best_threshold_results['optimal_recall']
        
        print(f"   Optimal threshold for {best_model}: {optimal_threshold:.1f}")
        print(f"   This threshold achieves {optimal_recall:.1%} recall")
        
        if optimal_threshold <= 0.3:
            print("   Strategy: Aggressive retention - contact many customers")
            print("   Use when: Cost of missing churners is very high")
        elif optimal_threshold <= 0.6:
            print("   Strategy: Balanced approach - targeted retention")
            print("   Use when: Moderate retention budget available")
        else:
            print("   Strategy: Conservative retention - focus on high-confidence cases")
            print("   Use when: Retention resources are limited")
    
    print(f"\n🚀 IMPLEMENTATION RECOMMENDATIONS:")
    print("1. Deploy the best performing model for production use")
    print("2. Use the optimal threshold for maximum recall")
    print("3. Monitor model performance over time")
    print("4. Implement targeted retention strategies based on predictions")
    print("5. Set up automated alerts for high-risk customers")

# =============================================================================
# MAIN EVALUATION EXECUTION
# =============================================================================

# Configure all models for evaluation
models_config = {
    'Logistic Regression': {
        'model': lr_model,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'use_encoded': False
    },
    'SVM': {
        'model': svm_model,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'use_encoded': False
    },
    'XGBoost': {
        'model': xgb_model,
        'X_test': X_test_cleaned,
        'y_test': y_test,
        'use_encoded': True
    },
    'Random Forest': {
        'model': rf_model,
        'X_test': X_test_cleaned,
        'y_test': y_test,
        'use_encoded': True
    },
    'Voting Classifier': {
        'model': voting_model,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'use_encoded': False
    }
}

# Execute comprehensive evaluation
print("🚀 Starting comprehensive model evaluation...")
comparison_df, all_results = compare_all_models(models_config)

# Demonstrate predictions
demonstrate_predictions(models_config)

# Generate business recommendations
generate_business_recommendations(comparison_df, all_results)

print("\n✅ Phase 8: Model Evaluation completed successfully!")
print("💡 All models have been evaluated with comprehensive metrics and business insights")

