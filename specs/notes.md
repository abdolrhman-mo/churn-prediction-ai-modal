## Binary Encoding
**What it is**: Yes and No to 1 and 0

## Label Encoding
**What it is**: Assigns each category a number.
**Example**:
- InternetService: DSL → 0, Fiber optic → 1, No → 2

## One-Hot Encoding (a.k.a. get_dummies in pandas)
**What it is**: Creates new columns for each category with 1/0 flags.
**Example for InternetService**:
InternetService_DSL      InternetService_Fiber optic      InternetService_No
         1                           0                             0
         0                           1                             0
         0                           0                             1

## Label Encoding vs One-Hot Encoding

Think of it like this:

| Feature Type               | Example                    | Correct Encoding                                   | Why                                                     |
|----------------------------|----------------------------|----------------------------------------------------|---------------------------------------------------------|
| **Ordinal** (ordered categories) | Small, Medium, Large       | Label Encoding (Small=0, Medium=1, Large=2)        | Order matters — 2 > 1 > 0 has meaning                   |
| **Nominal** (unordered categories) | Red, Green, Blue          | One-Hot Encoding                                   | No order — numbers like 0,1,2 would create a fake ranking |

---

### Why not just use label encoding for everything?
Because in many models, numbers imply size.

**Example:**
If you label encode:
Red → 0, Green → 1, Blue → 2
A model like Logistic Regression might think Blue is somehow "bigger" than Red, which is nonsense.


## Imbalanced Target
An imbalanced target happens when, in your dataset, the thing you’re trying to predict (the target variable) has a very uneven distribution of classes.

For example:  
Suppose you’re predicting whether a customer will buy a product:  
- Yes: 980 rows  
- No: 20 rows  

That means 98% are "Yes" and only 2% are "No".  
This is imbalanced because the model could just predict "Yes" for everything and still get 98% accuracy — but it wouldn’t actually learn how to detect the "No" cases well.


## Ressambling
if a value is so much greater than the other value
- over sambling
- under sambling

## PCA
reduce columns if they're a lot

## Supervised ML Algorithms (Simple Guide)

### **What is Supervised Learning?**
- You have data with answers (like customer info + whether they left)
- Computer learns from examples to predict new answers

### **Main Algorithm Types:**

#### **1. Simple & Fast (Good for beginners)**
- **Linear Regression**: Predicts numbers (like price, age)
- **Logistic Regression**: Predicts Yes/No (like will customer leave?)
- **KNN**: Finds similar examples to make predictions

#### **2. More Powerful (Better accuracy)**
- **SVM**: Good at finding patterns in complex data
- **Decision Tree**: Makes decisions like a flowchart
- **Random Forest**: Multiple trees working together (very accurate)

#### **3. Advanced Techniques (Best performance)**
- **Bagging**: Combines multiple models for better results
- **Boosting**: Each model learns from previous mistakes
- **Stacking**: Uses different models and combines their predictions
- **Voting**: Multiple models vote on the final prediction (majority wins)

### **For Your Churn Project:**
- **Start with**: Logistic Regression (simple, fast)
- **Try next**: Random Forest (more accurate)
- **Best option**: XGBoost (often wins competitions)

### **Important Tools:**
- **Cross Validation**: Tests model on different data splits
- **Hyperparameter Tuning**: Finds best settings for your model
- **Imbalance Handling**: Deals with few customers leaving vs many staying
## Unsupervised ML Algorithms
1 week (project)
## DeepLearning 1 week (project
## Last week (final project)


## recall
we want a high recall

we measure with recall, accuracy, percision 
classification report gets the percentage of all the previous