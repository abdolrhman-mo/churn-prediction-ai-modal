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


# Thresholding
# STEP 1: Get probabilities from your model
probabilities = svm_model.predict_proba(X_test_cleaned)[:, 1]
# This gives you array like [0.25, 0.65, 0.45, 0.85, ...]

print("Sample probabilities (first 5 customers):")
for i in range(5):
    print(f"Customer {i+1}: {probabilities[i]:.3f} probability of churning")

print("\nTesting different thresholds:")
print("Threshold | Recall | What this means")
print("-" * 50)

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    # Apply threshold
    predictions = (probabilities >= threshold).astype(int)
    pred_labels = ['Yes' if pred == 1 else 'No' for pred in predictions]

    # Calculate recall
    recall = recall_score(y_test, pred_labels, pos_label='Yes')

    # Explanation
    if threshold == 0.3:
        meaning = "Very aggressive - flag lots of customers"
    elif threshold == 0.5:
        meaning = "Default threshold"
    elif threshold == 0.7:
        meaning = "Very conservative - flag few customers"
    else:
        meaning = "Balanced approach"

    print(f"{threshold:^9} | {recall:.4f} | {meaning}")

print("\n💡 Pick the threshold that gives you the best recall!")
print("Usually 0.3-0.4 works well for churn prediction")


# In[247]:


# Simple SVM threshold optimization
print("\n" + "="*60)
print("🎯 SVM THRESHOLD OPTIMIZATION - FINDING BEST RECALL")
print("="*60)

# Get SVM probabilities (using the same data format as your existing code)
svm_probs = svm_model.predict_proba(X_test_scaled)[:, 1]

# Test different thresholds and find best recall
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best_recall = 0
best_threshold = 0.5
best_metrics = {}

print("Threshold | Recall  | Precision | F1-Score")
print("-" * 45)

for threshold in thresholds:
    # Apply threshold
    predictions = (svm_probs >= threshold).astype(int)
    pred_labels = ['Yes' if pred == 1 else 'No' for pred in predictions]

    # Calculate metrics
    recall = recall_score(y_test, pred_labels, pos_label='Yes')
    precision = precision_score(y_test, pred_labels, pos_label='Yes')
    f1 = f1_score(y_test, pred_labels, pos_label='Yes')

    print(f"{threshold:^9} | {recall:.4f}  | {precision:.4f}    | {f1:.4f}")

    # Track best recall
    if recall > best_recall:
        best_recall = recall
        best_threshold = threshold
        best_metrics = {'recall': recall, 'precision': precision, 'f1': f1}

print(f"\n🏆 BEST SVM THRESHOLD: {best_threshold}")
print(f"   Recall: {best_metrics['recall']:.4f}")
print(f"   Precision: {best_metrics['precision']:.4f}")
print(f"   F1-Score: {best_metrics['f1']:.4f}")

# Apply best threshold to get final SVM predictions
final_svm_predictions = (svm_probs >= best_threshold).astype(int)
final_svm_labels = ['Yes' if pred == 1 else 'No' for pred in final_svm_predictions]

# Show final performance
final_accuracy = accuracy_score(y_test, final_svm_labels)
print(f"\n📊 FINAL SVM PERFORMANCE with threshold {best_threshold}:")
print(f"   Accuracy:  {final_accuracy:.4f}")
print(f"   Precision: {best_metrics['precision']:.4f}")
print(f"   Recall:    {best_metrics['recall']:.4f}")
print(f"   F1-Score:  {best_metrics['f1']:.4f}")

# Show high-risk customers
high_risk_count = (svm_probs >= best_threshold).sum()
print(f"\n🔴 HIGH-RISK CUSTOMERS (Probability >= {best_threshold}): {high_risk_count}")

print("✅ SVM threshold optimization completed!")
print(f"💡 Use threshold {best_threshold} for production deployment")

# Enhanced thresholding analysis for all models
print("\n" + "="*80)
print("🎯 THRESHOLD OPTIMIZATION - FINDING BEST RECALL FOR EACH MODEL")
print("="*80)

# Function to find best threshold for any model
def find_best_threshold(model, X_test_data, y_test, model_name, use_encoded=False):
    """Find the threshold that gives the best recall for a given model"""

    # Get probabilities
    if use_encoded:
        # For models that need encoded data
        y_test_encoded = label_encoder.transform(y_test)
        probabilities = model.predict_proba(X_test_data)[:, 1]
    else:
        # For models that use scaled data
        probabilities = model.predict_proba(X_test_data)[:, 1]

    print(f"\n🔍 {model_name.upper()} - Threshold Analysis")
    print("-" * 60)

    # Test a wider range of thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    print("Threshold | Recall  | Precision | F1-Score | Meaning")
    print("-" * 65)

    for threshold in thresholds:
        # Apply threshold
        predictions = (probabilities >= threshold).astype(int)
        pred_labels = ['Yes' if pred == 1 else 'No' for pred in predictions]

        # Calculate metrics
        recall = recall_score(y_test, pred_labels, pos_label='Yes')
        precision = precision_score(y_test, pred_labels, pos_label='Yes')
        f1 = f1_score(y_test, pred_labels, pos_label='Yes')

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

# Analyze all models
print("\n" + "="*80)
print("📊 COMPREHENSIVE THRESHOLD ANALYSIS")
print("="*80)

# Test each model
svm_best = find_best_threshold(svm_model, X_test_scaled, y_test, "SVM", False)
lr_best = find_best_threshold(lr_model, X_test_scaled, y_test, "Logistic Regression", False)
xgb_best = find_best_threshold(xgb_model, X_test_cleaned, y_test, "XGBoost", True)
rf_best = find_best_threshold(rf_model, X_test_cleaned, y_test, "Random Forest", True)
voting_best = find_best_threshold(voting_model, X_test_scaled, y_test, "Voting Classifier", False)

# Summary of best thresholds
print("\n" + "="*80)
print("🏆 BEST THRESHOLDS SUMMARY - ALL MODELS")
print("="*80)

best_results = [
    ("SVM", svm_best),
    ("Logistic Regression", lr_best),
    ("XGBoost", xgb_best),
    ("Random Forest", rf_best),
    ("Voting Classifier", voting_best)
]

print("Model              | Best Threshold | Best Recall | Best Precision | Best F1")
print("-" * 80)

for model_name, (threshold, recall, precision, f1) in best_results:
    print(f"{model_name:<18} | {threshold:^14} | {recall:^11.4f} | {precision:^14.4f} | {f1:^7.4f}")

# Find overall best model for recall
overall_best_idx = max(range(len(best_results)), key=lambda i: best_results[i][1][1])
overall_best_model, overall_best_metrics = best_results[overall_best_idx]
best_threshold, best_recall, best_precision, best_f1 = overall_best_metrics

print(f"\n🥇 OVERALL BEST MODEL FOR RECALL:")
print(f"   {overall_best_model} with threshold {best_threshold}")
print(f"   Recall: {best_recall:.4f}")
print(f"   Precision: {best_precision:.4f}")
print(f"   F1-Score: {best_f1:.4f}")

# Business recommendations based on threshold analysis
print("\n" + "="*80)
print("💡 BUSINESS RECOMMENDATIONS - THRESHOLD STRATEGY")
print("="*80)

print("Threshold Strategy Options:")
print("1. 🚨 HIGH RECALL (0.1-0.3): Flag many customers, catch most churners")
print("   - Use when: Cost of missing churners is very high")
print("   - Trade-off: Higher false positives, more retention effort needed")
print("")
print("2. ⚖️ BALANCED (0.4-0.6): Good balance of recall and precision")
print("   - Use when: Balanced approach needed, moderate retention budget")
print("   - Trade-off: Moderate false positives, reasonable effort")
print("")
print("3. 🎯 HIGH PRECISION (0.7-0.9): Flag few customers, high confidence")
print("   - Use when: Retention resources are limited, focus on high-value customers")
print("   - Trade-off: Lower recall, might miss some churners")

print(f"\n🎯 RECOMMENDED STRATEGY for {overall_best_model}:")
if best_threshold <= 0.3:
    print("   Use threshold {best_threshold:.1f} for MAXIMUM RECALL")
    print("   This will catch {best_recall:.1%} of churning customers")
    print("   Strategy: Aggressive retention - contact many customers")
elif best_threshold <= 0.6:
    print("   Use threshold {best_threshold:.1f} for BALANCED PERFORMANCE")
    print("   This will catch {best_recall:.1%} of churning customers")
    print("   Strategy: Moderate retention - targeted approach")
else:
    print("   Use threshold {best_threshold:.1f} for HIGH PRECISION")
    print("   This will catch {best_recall:.1%} of churning customers")
    print("   Strategy: Conservative retention - focus on high-confidence cases")

# Apply best threshold to get final predictions
print("\n" + "="*80)
print("🚀 FINAL PREDICTIONS WITH OPTIMIZED THRESHOLD")
print("="*80)

# Get probabilities from best model
if overall_best_model == "SVM":
    best_probs = svm_model.predict_proba(X_test_scaled)[:, 1]
elif overall_best_model == "Logistic Regression":
    best_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
elif overall_best_model == "XGBoost":
    best_probs = xgb_model.predict_proba(X_test_cleaned)[:, 1]
elif overall_best_model == "Random Forest":
    best_probs = rf_model.predict_proba(X_test_cleaned)[:, 1]
else:  # Voting Classifier
    best_probs = voting_model.predict_proba(X_test_scaled)[:, 1]

# Apply best threshold
final_predictions = (best_probs >= best_threshold).astype(int)
final_labels = ['Yes' if pred == 1 else 'No' for pred in final_predictions]

# Calculate final metrics
final_recall = recall_score(y_test, final_labels, pos_label='Yes')
final_precision = precision_score(y_test, final_labels, pos_label='Yes')
final_f1 = f1_score(y_test, final_labels, pos_label='Yes')
final_accuracy = accuracy_score(y_test, final_labels)

print(f"Final Performance with {best_threshold:.1f} threshold:")
print(f"   Accuracy:  {final_accuracy:.4f}")
print(f"   Precision: {final_precision:.4f}")
print(f"   Recall:    {final_recall:.4f}")
print(f"   F1-Score:  {final_f1:.4f}")

# Show sample of high-risk customers
print(f"\n🔴 HIGH-RISK CUSTOMERS (Probability >= {best_threshold:.1f}):")
high_risk_indices = np.where(best_probs >= best_threshold)[0]
print(f"Total high-risk customers: {len(high_risk_indices)}")

if len(high_risk_indices) > 0:
    print("\nSample high-risk customers:")
    for i, idx in enumerate(high_risk_indices[:5]):  # Show first 5
        customer = X_test.iloc[idx]
        prob = best_probs[idx]
        actual = y_test.iloc[idx]
        print(f"   Customer {i+1}: {prob:.3f} churn probability")
        print(f"     Contract: {customer.get('Contract', 'N/A')}")
        print(f"     Monthly Charges: ${customer.get('MonthlyCharges', 'N/A'):.2f}")
        print(f"     Actual Churn: {actual}")
        print("")

print("✅ Threshold optimization completed!")
print("💡 Use the recommended threshold for production deployment")


# In[241]:


# Evaluation function for cleaner code
def evaluate_model(model, y_true, y_pred, model_name, X_test_data, use_encoded=False):
    """Evaluate a single model and return metrics"""

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='Yes')
    recall = recall_score(y_true, y_pred, pos_label='Yes')
    f1 = f1_score(y_true, y_pred, pos_label='Yes')

    # ROC AUC (handle different data formats)
    if use_encoded:
        # For XGBoost and Random Forest (using encoded test data)
        y_true_encoded = label_encoder.transform(y_true)
        roc_auc = roc_auc_score(y_true_encoded, model.predict_proba(X_test_data)[:,1])
    else:
        # For Logistic Regression and SVM (using scaled test data)
        roc_auc = roc_auc_score(y_true, model.predict_proba(X_test_data)[:,1])

    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }

# Evaluate all models
print("\n" + "="*60)
print("📊 MODEL EVALUATION RESULTS")
print("="*60)

# Evaluate each model
lr_results = evaluate_model(lr_model, y_test, y_pred_lr, "Logistic Regression", X_test_scaled, use_encoded=False)
svm_results = evaluate_model(svm_model, y_test, y_pred_svm, "SVM", X_test_scaled, use_encoded=False)
xgb_results = evaluate_model(xgb_model, y_test, y_pred_xgb, "XGBoost", X_test_cleaned, use_encoded=True)
rf_results = evaluate_model(rf_model, y_test, y_pred_rf, "Random Forest", X_test_cleaned, use_encoded=True)
voting_results = evaluate_model(voting_model, y_test, y_pred_voting, "Voting Classifier", X_test_scaled, use_encoded=False)

# Display detailed results for each model
models_to_evaluate = [lr_results, svm_results, xgb_results, rf_results, voting_results]

for model_result in models_to_evaluate:
    print(f"\n{model_result['Model'].upper()}")
    print("-" * 40)
    print(f"Recall:    {model_result['Recall']:.4f}")
    # print(f"Accuracy:  {model_result['Accuracy']:.4f}")
    # print(f"Precision: {model_result['Precision']:.4f}")
    # print(f"F1 Score:  {model_result['F1 Score']:.4f}")
    # print(f"ROC AUC:   {model_result['ROC AUC']:.4f}")

    # Get predictions for classification report
    if model_result['Model'] == "Logistic Regression":
        y_pred_for_report = y_pred_lr
    elif model_result['Model'] == "SVM":
        y_pred_for_report = y_pred_svm
    elif model_result['Model'] == "XGBoost":
        y_pred_for_report = y_pred_xgb
    else:  # Random Forest
        y_pred_for_report = y_pred_rf

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_for_report))


# In[237]:


# Model comparison summary
print("\n" + "="*60)
print("🏆 MODEL COMPARISON SUMMARY")
print("="*60)

# Create comparison DataFrame
comparison_data = []
for model_result in models_to_evaluate:
    comparison_data.append([
        model_result['Model'],
        model_result['Accuracy'],
        model_result['Precision'],
        model_result['Recall'],
        model_result['F1 Score'],
        model_result['ROC AUC']
    ])

comparison_df = pd.DataFrame(comparison_data, 
                           columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])
print(comparison_df.round(4))


# In[238]:


# Find best performing model
best_model_idx = comparison_df['F1 Score'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_f1_score = comparison_df.loc[best_model_idx, 'F1 Score']

print(f"\n🥇 BEST PERFORMING MODEL:")
print(f"   {best_model_name} with F1 Score: {best_f1_score:.4f}")

# Sample prediction demonstration
print("\n" + "="*60)
print("🔮 SAMPLE PREDICTION DEMONSTRATION")
print("="*60)

# Use first test customer for demonstration
sample_idx = 0
sample_original = X_test.iloc[sample_idx]
sample_actual = y_test.iloc[sample_idx]

print(f"Customer Sample (first test customer):")
print(f"  - Contract: {sample_original.get('Contract', 'N/A')}")
print(f"  - Monthly Charges: ${sample_original.get('MonthlyCharges', 'N/A'):.2f}")
print(f"  - Tenure: {sample_original.get('tenure', 'N/A')} months")
print(f"  - Internet Service: {sample_original.get('InternetService', 'N/A')}")

# Get predictions from all models
sample_predictions = {
    'Logistic Regression': y_pred_lr[sample_idx],
    'SVM': y_pred_svm[sample_idx],
    'XGBoost': y_pred_xgb[sample_idx],
    'Random Forest': y_pred_rf[sample_idx],
    'Voting Classifier': y_pred_voting[sample_idx]
}

print(f"\nPredictions:")
print(f"  - Actual: {sample_actual}")
for model_name, prediction in sample_predictions.items():
    print(f"  - {model_name}: {prediction}")

# Show prediction probabilities for best model
print(f"\n🎯 Best Model ({best_model_name}) Prediction Probabilities:")
if best_model_name == "Logistic Regression":
    proba = lr_model.predict_proba(X_test_scaled[sample_idx:sample_idx+1])[0]
elif best_model_name == "SVM":
    proba = svm_model.predict_proba(X_test_scaled[sample_idx:sample_idx+1])[0]
elif best_model_name == "XGBoost":
    proba = xgb_model.predict_proba(X_test_cleaned[sample_idx:sample_idx+1])[0]
elif best_model_name == "Voting Classifier":
    proba = voting_model.predict_proba(X_test_scaled[sample_idx:sample_idx+1])[0]
else:  # Random Forest
    proba = rf_model.predict_proba(X_test_cleaned[sample_idx:sample_idx+1])[0]

print(f"  - Stay (No): {proba[0]:.3f}")
print(f"  - Churn (Yes): {proba[1]:.3f}")

# Step 8: Business insights summary
print("\n" + "="*60)
print("💡 BUSINESS INSIGHTS SUMMARY")
print("="*60)

print("Key Findings:")
print(f"1. Best performing model: {best_model_name}")
print(f"2. Overall accuracy range: {comparison_df['Accuracy'].min():.1%} - {comparison_df['Accuracy'].max():.1%}")
print(f"3. Best F1 Score: {best_f1_score:.4f} ({best_model_name})")
print(f"4. All models show {'good' if best_f1_score > 0.7 else 'moderate'} performance")

print("\nRecommendations:")
print("1. Deploy the best performing model for production use")
print("2. Monitor model performance over time")
print("3. Use predictions to identify high-risk customers")
print("4. Implement targeted retention strategies")

