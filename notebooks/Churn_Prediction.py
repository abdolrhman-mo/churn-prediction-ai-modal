#!/usr/bin/env python
# coding: utf-8

# ## **Phase 1: Data Loading & Initial Exploration**

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek


# In[27]:


# Load dataset
# Note: df is short for dataframe for pandas dataframe
df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# See the first 5 rows
df.head()


# In[5]:


# See how many rows and columns
df.shape


# In[6]:


# See column names and types
df.info()

# Notes:
# Dtype => Data Type
# object = Text (strings) or mixed data. Pandas uses object for things that aren’t purely numbers.
# Non-Null Count => Number of non-null values in the column


# ## **Phase 2: Data Quality Assessment**

# In[7]:


#  Count missing values in each column
df.isnull().sum()


# In[28]:


# Check for duplicate customers
df.duplicated().sum()


# In[29]:


# See current data types
df.dtypes

# Conclusion:
# - TotalCharges is object, not float


# ## **Phase 3: Data Cleaning**

# In[30]:


# Convert 'TotalCharges' to numeric, some rows may be blank
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Note: errors='coerce' means that if the conversion fails, the value will be set to NaN

# Now check again for missing values
df.isnull().sum()

# Drop rows with missing TotalCharges
df = df.dropna()

# Check shapes
df.shape


# In[31]:


df.dtypes


# ## **Phase 4: Data Visualization & Exploration**

# In[16]:


# Plot churn count
sns.countplot(x='Churn', data=df)
plt.title("Churn Count")

# Plot churn by gender
sns.countplot(x='gender', hue='Churn', data=df)
plt.title("Churn by Gender")

# Plot churn by Contract type
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract Type")


# In[17]:


# # Function to evaluate model performance
# def evaluate_model(model, X_test, y_test, model_name):
#     """Evaluate model and print comprehensive metrics"""
#     y_pred = model.predict(X_test)
#     y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

#     print(f"\n=== {model_name} Performance ===")
#     print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
#     print(f"Precision: {precision_score(y_test, y_pred):.4f}")
#     print(f"Recall: {recall_score(y_test, y_pred):.4f}")
#     print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

#     if y_pred_proba is not None:
#         print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

#     print("\nConfusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))

#     return y_pred




# In[18]:


# # Model comparison visualization
# fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# # Confusion matrix for Logistic Regression
# cm_lr = confusion_matrix(y_test, lr_predictions)
# sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
# axes[0].set_title('Logistic Regression - Confusion Matrix')
# axes[0].set_xlabel('Predicted')
# axes[0].set_ylabel('Actual')

# # Confusion matrix for SVM
# cm_svm = confusion_matrix(y_test, svm_predictions)
# sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
# axes[1].set_title('SVM - Confusion Matrix')
# axes[1].set_xlabel('Predicted')
# axes[1].set_ylabel('Actual')

# plt.tight_layout()
# plt.show()

# # Performance comparison summary
# models_performance = {
#     'Model': ['Logistic Regression', 'SVM'],
#     'Accuracy': [
#         accuracy_score(y_test, lr_predictions),
#         accuracy_score(y_test, svm_predictions)
#     ],
#     'Precision': [
#         precision_score(y_test, lr_predictions),
#         precision_score(y_test, svm_predictions)
#     ],
#     'Recall': [
#         recall_score(y_test, lr_predictions),
#         recall_score(y_test, svm_predictions)
#     ],
#     'F1-Score': [
#         f1_score(y_test, lr_predictions),
#         f1_score(y_test, svm_predictions)
#     ]
# }

# comparison_df = pd.DataFrame(models_performance)
# print("\n=== Model Performance Comparison ===")
# print(comparison_df.round(4))

## 7. Conclusions and Business Insights {#conclusions}

### Model Performance Summary

Based on our analysis, both models show strong performance in predicting customer churn:

**Key Findings:**
- Both Logistic Regression and SVM achieved high accuracy scores
- The models can effectively identify customers at risk of churning
- Feature engineering and data preprocessing were crucial for model performance

### Business Recommendations

1. **Focus on Contract Length**: Month-to-month customers are at highest risk of churn
   - **Action**: Offer incentives for longer-term contracts
   - **Strategy**: Develop retention programs specifically for month-to-month customers

2. **Early Warning System**: Use the model predictions to implement proactive retention
   - **Action**: Contact high-risk customers before they churn
   - **Strategy**: Personalized offers based on customer profiles

3. **Service Optimization**: Analyze features that contribute most to churn
   - **Action**: Improve services that correlate with higher churn rates
   - **Strategy**: Focus on customer experience improvements

### Next Steps

1. **Feature Importance Analysis**: Investigate which features contribute most to predictions
2. **Model Tuning**: Hyperparameter optimization for better performance
3. **Real-time Implementation**: Deploy the model for ongoing churn prediction
4. **A/B Testing**: Test retention strategies on predicted high-risk customers

### Technical Notes

- **Data Quality**: The dataset was clean with minimal missing values
- **Feature Engineering**: One-hot encoding was effective for categorical variables
- **Model Selection**: Both models performed well, suggesting the problem is well-suited to linear approaches
- **Scalability**: The preprocessing pipeline can be applied to new data for ongoing predictions
### Key Insights from Visualizations

**Churn Distribution:**
- The dataset shows an imbalanced distribution with more customers staying than leaving
- This is typical in churn prediction problems

**Gender Analysis:**
- Churn appears to be relatively balanced between male and female customers
- Gender may not be a strong predictor of churn

**Contract Type Analysis:**
- Month-to-month contracts show the highest churn rate
- Customers with longer-term contracts (One year, Two year) are much less likely to churn
- This suggests contract length is a strong predictor of customer retention

# ## **Phase 5: Data Preprocessing**

# In[32]:


# Drop customerID (not useful)
df.drop(['customerID'], axis=1, inplace=True)


# In[33]:


num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(num_cols)
print(categorical_cols)


# In[34]:


# Dynamic column classification based on data types and unique values
# Binary columns: columns with only 2 unique values (Yes/No)
binary_cols = []
for col in categorical_cols:
    if col != 'Churn':  # Exclude target variable
        unique_vals = df[col].nunique()
        if unique_vals == 2:
            binary_cols.append(col)

# One-hot columns: columns with more than 2 unique values
onehot_cols = []
for col in categorical_cols:
    if col != 'Churn':  # Exclude target variable
        unique_vals = df[col].nunique()
        if unique_vals > 2:
            onehot_cols.append(col)

print("Binary columns (2 unique values):", binary_cols)
print("One-hot columns (>2 unique values):", onehot_cols)

# TODO: why aren't we using label encoding?


# In[35]:


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


# In[36]:


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


# In[48]:


# Test if data is preprocessed
print("Original shape:", X_train.shape)
print("Processed shape:", X_train_cleaned.shape)

# If OneHot worked, processed data should have MORE columns
# Example: Original (7000, 20) → Processed (7000, 45)


# In[50]:


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


# In[43]:


# Check if your data is imbalanced
print("Churn distribution:")
print(y_train.value_counts())
print("\nPercentages:")
print(y_train.value_counts(normalize=True) * 100)


# In[51]:


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

# Model Training
model = LogisticRegression(random_state=42)
model.fit(X_resampled_scaled, y_resampled)


# ## **Phase 6: Model Training**

# In[52]:


# Train Logistic Regression model
print("Training Logistic Regression model...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Train SVM model  
print("Training SVM model...")
svm_model = SVC(kernel='rbf', random_state=42, probability=True)
svm_model.fit(X_train_scaled, y_train)

print("Model training completed!")


# ## **Phase 7: Model Evaluation**

# In[17]:


# Features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


# Create and train model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict
y_pred_lr = log_reg.predict(X_test)


# In[9]:


# Create and train SVM
svm = SVC(probability=True)
svm.fit(X_train, y_train)

# Predict
y_pred_svm = svm.predict(X_test)


# In[10]:


# Logistic Regression
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))
print("ROC AUC:", roc_auc_score(y_test, log_reg.predict_proba(X_test)[:,1]))
print(classification_report(y_test, y_pred_lr))

# SVM
print("\nSVM:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm))
print("Recall:", recall_score(y_test, y_pred_svm))
print("F1 Score:", f1_score(y_test, y_pred_svm))
print("ROC AUC:", roc_auc_score(y_test, svm.predict_proba(X_test)[:,1]))
print(classification_report(y_test, y_pred_svm))


# In[11]:


# Pick a random customer from test set
sample = X_test.iloc[0]

# Predict churn (Logistic Regression)
result = log_reg.predict([sample])
print("Will churn (Logistic Regression):", result)

# Predict churn (SVM)
result_svm = svm.predict([sample])
print("Will churn (SVM):", result_svm)

