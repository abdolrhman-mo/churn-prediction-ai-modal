# Churn Prediction Model - Project Specification

## Project Goal
Build an AI model that can predict whether a telecom customer will leave the company (churn) based on their profile and usage patterns.

---

## 📋 **Phase 1: Data Loading & Initial Exploration**

### What this means:
- Load your CSV file into a pandas DataFrame (like opening an Excel file in Python)
- Get a first look at your data to understand what you're working with

### What you'll do:
```python
import pandas as pd
df = pd.read_csv('telecom_churn.csv')  # Load the data
print(df.shape)                        # See how many rows and columns (like 7000 rows, 21 columns)
print(df.head())                       # See first 5 rows
print(df.info())                       # See column names and types
```

### Why this matters:
- You need to understand what data you have before you can build a model
- Check if the file loaded correctly
- See if data looks reasonable

---

## 🔍 **Phase 2: Data Quality Assessment**

### What this means:
- Check if your data has problems that could break your model
- Like checking if some customers have missing information or weird values

### What you'll do:
```python
print(df.isnull().sum())              # Count missing values in each column
print(df.duplicated().sum())          # Check for duplicate customers  
print(df['Churn'].value_counts())     # See how many customers left vs stayed
```

### Key data quality checks:
1. **Missing values**: Some customers don't have phone service info
2. **Duplicate records**: Check if same customer appears multiple times
3. **Wrong data types**: Monthly charges stored as text instead of numbers
4. **Weird values**: Spaces instead of "No" for some services
5. **Imbalanced target**: 80% customers stayed, only 20% left

### Why this matters:
- Bad data = bad model predictions
- You need to fix problems before training your model

---

## 🧹 **Phase 3: Data Cleaning**

### What this means:
- Fix the problems you found in Phase 2
- Make the data ready for machine learning algorithms

### What you'll do:
```python
# Convert 'TotalCharges' to numeric, some rows may be blank
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Now check again for missing values
df.isnull().sum()

# Drop rows with missing TotalCharges
df = df.dropna()

# Check shapes
df.shape
```
                                    - Need to revise from here
### Common cleaning tasks:
- **Convert text numbers to actual numbers**: "29.85" → 29.85
- **Fill missing values**: Replace empty cells with 0 or "No"
- **Standardize text**: Make sure all "No" values are actually "No", not " " or "no"
- **Remove duplicates**: Delete customers that appear twice

### Why this matters:
- Machine learning algorithms need clean, consistent data
- Garbage in = garbage out

---

## 📊 **Phase 4: Data Visualization & Exploration**

### What this means:
- Look at your data with pictures instead of just numbers
- Find out which customer traits make them more likely to leave

### What you'll create (3 simple charts):

1. **Churn Pie Chart**
   - Shows how many customers left vs stayed
   - Helps you see if the problem is big or small

2. **Monthly Charges Box Plot**
   - Compares monthly bills between customers who left vs stayed
   - Shows if expensive customers leave more often

3. **Contract Type Bar Chart**
   - Shows churn rates for different contract types
   - Reveals if month-to-month customers leave more

### Why this matters:
- **See patterns**: Pictures make it easier to spot what causes churn
- **Business decisions**: Know which customers to focus on keeping
- **Simple insights**: No complex math needed - just look at the charts

---

## ⚙️ **Phase 5: Data Preprocessing**

### What this means:
- Clean up your data so the computer can understand it
- Remove stuff that doesn't help predict churn

### Main tasks:
- **Remove useless columns**: 
  - `customerID` - just a random number, doesn't tell us anything about churn
  - Other columns that don't help predict if someone will leave
- **Convert text to numbers**:
  - "Male/Female" → 1/0 (computer only understands numbers)
  - "Month-to-month/One year/Two year" → separate columns with 1/0
- **Prepare data for training**: 
  - X = customer info (age, gender, services, etc.)
  - y = whether they churned (Yes/No)
  - Split data: 80% to train, 20% to test

### Why this matters:
- Computer needs clean, simple data to learn from
- Useless columns confuse the computer and make predictions worse

---

## 🤖 **Phase 6: Model Training**

### What this means:
- Teach a computer algorithm to recognize patterns between customer info and churn
- Like showing the algorithm: "Customers with month-to-month contracts churn more"

### What happens:
- **Algorithm learns patterns**: High monthly charges + month-to-month contract = likely to churn
- **Creates a "brain"**: Model that can make predictions on new customers
- **Uses training data**: 80% of your customers to learn from

### Popular algorithms to try:
- **Logistic Regression**: Simple, fast, good starting point
- **Random Forest**: More powerful, handles complex patterns
- **XGBoost**: Often wins competitions, very powerful

### Why this matters:
- This is where the "AI" happens - the model learns to predict churn

---

## 📊 **Phase 7: Model Evaluation**

### What this means:
- Test how well your model can predict churn on customers it has never seen before
- Like giving a student a test on material they studied

### Key metrics:
- **Accuracy**: Overall correctness (85% = correct on 85 out of 100 customers)
- **Precision**: When model says "will churn", how often is it right?
- **Recall**: Of customers who actually churned, how many did we catch?
- **F1-Score**: Balance between precision and recall

### Confusion Matrix example:
```
Predicted:    No Churn  |  Churn
Actual:
No Churn         850    |   50     (850 correctly predicted to stay)
Churn             75    |  125     (125 correctly predicted to leave)
```

### Why this matters:
- Tells you if your model is good enough to use in real business
- Shows where the model makes mistakes

---

## 🎯 **Phase 8: Model Optimization** (Optional)

### What this means:
- Fine-tune your model to get better predictions
- Like adjusting settings to get better performance

### Common optimization techniques:
- **Hyperparameter tuning**: Adjust algorithm settings for better performance
- **Feature engineering**: Create new features (e.g., "charges per month of tenure")
- **Feature selection**: Remove unimportant features that confuse the model
- **Ensemble methods**: Combine multiple models for better predictions

### Why this matters:
- Can improve accuracy from 85% to 90%+
- Better model = better business decisions

---

## 📈 **Success Metrics**

### Minimum viable model:
- **Accuracy > 80%**: Better than random guessing
- **Recall > 70%**: Catch most customers who will churn
- **Model runs without errors**: Technical success

### Business success:
- **Actionable insights**: Which factors drive churn most?
- **Cost savings**: Prevent customer loss through targeted retention
- **ROI**: Model saves more money than it costs to develop

---

## 🛠 **Tools & Libraries Needed**

```python
import pandas as pd           # Data manipulation
import numpy as np            # Math operations  
import matplotlib.pyplot as plt  # Basic plotting
import seaborn as sns         # Pretty plots
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.ensemble import RandomForestClassifier   # ML algorithm
from sklearn.metrics import accuracy_score, classification_report  # Evaluation
```

---

## 📁 **Expected Outputs**

1. **Trained model file**: Save your model to use later
2. **Performance report**: Accuracy, precision, recall metrics
3. **Feature importance**: Which customer attributes matter most
4. **Predictions**: List of customers likely to churn
5. **Documentation**: What you learned and how to use the model