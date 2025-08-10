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