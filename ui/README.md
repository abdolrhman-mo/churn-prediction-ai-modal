# 🎯 SVM Recall Optimization Dashboard - UI Module Structure

This directory contains the **SVM Recall Optimization Dashboard** - a specialized Streamlit application focused on maximizing recall for churn prediction using Support Vector Machines.

## 📁 Directory Structure

```
ui/
├── __init__.py                 # Package initialization
├── main_app.py                 # Main SVM Recall Optimization dashboard
├── data_manager.py             # Data loading, preprocessing, and model training
├── styling.py                  # Custom CSS and styling functions
├── pages/                      # Individual page modules
│   ├── __init__.py            # Pages package initialization
│   ├── home_page.py           # Home page with project overview
│   ├── single_prediction_page.py  # Single customer prediction form
│   ├── data_insights_page.py      # Data exploration and analysis
│   ├── model_performance_page.py  # Model evaluation metrics
│   └── batch_predictions_page.py  # Batch CSV processing
└── README.md                   # This file
```

## 🚀 How to Run

### Option 1: Direct Streamlit Run
```bash
streamlit run ui/main_app.py
```

### Option 2: Using the Runner Script
```bash
python run_app.py
```

## 🎯 What This Dashboard Does

The **SVM Recall Optimization Dashboard** is specifically designed to:

1. **Train SVM Models**: Uses your exact preprocessing pipeline with SMOTE resampling
2. **Optimize Thresholds**: Tests multiple thresholds (0.1 to 0.9) to find maximum recall
3. **Visualize Performance**: Interactive charts showing threshold vs. metrics relationships
4. **Explain Results**: Clear explanations of why lower thresholds achieve higher recall
5. **Technical Details**: Code examples and implementation insights

## 📊 Key Features

### 🎯 **Threshold Analysis**
- Interactive line charts showing Recall, Precision, F1-Score vs. Threshold
- Automatic identification of optimal threshold for maximum recall
- Visual highlighting of best performing threshold

### 📈 **Performance Visualization**
- Confusion matrices at optimal thresholds
- Probability distribution histograms
- Risk distribution pie charts
- Customer segmentation by churn probability

### ⚙️ **Technical Implementation**
- Complete ML pipeline walkthrough
- Code examples for each component
- Data preprocessing explanations
- Model training and evaluation details

## 🔧 How It Works

1. **Data Loading**: Loads your telecom churn dataset (or creates demo data)
2. **Preprocessing**: Applies your exact pipeline (SMOTE, scaling, encoding)
3. **SVM Training**: Trains SVM with linear kernel and probability estimates
4. **Threshold Testing**: Tests 9 different thresholds (0.1 to 0.9)
5. **Performance Analysis**: Calculates recall, precision, F1-score for each threshold
6. **Visualization**: Creates interactive charts and tables
7. **Insights**: Explains why certain thresholds perform better

## 🎯 Why Threshold 0.1 = Maximum Recall

**The Key Insight:**
- **Lower threshold (0.1)**: Model predicts "Yes, will churn" even with low confidence
- **Higher threshold (0.9)**: Model only predicts "Yes" when very confident
- **For churn detection**: Better to catch all potential churners (even false alarms) than miss real ones

**Business Impact:**
- Higher recall means fewer missed churners
- Lower precision means more false alarms
- **Trade-off**: Catch everyone vs. accuracy of predictions

## 🚀 Next Steps

1. **Run the dashboard**: `streamlit run ui/main_app.py`
2. **Explore thresholds**: Navigate to "SVM Threshold Analysis" to see the key insight
3. **Understand results**: Check "Recall Optimization Results" for optimal threshold
4. **Technical details**: Visit "Technical Implementation" for code explanations
5. **Apply insights**: Use the optimal threshold in your production models

## 📝 Key Takeaways for AI Course

- **Threshold 0.1 = Maximum Recall**: Lower thresholds catch more positive cases
- **SMOTE Balancing**: Essential for imbalanced datasets like churn prediction
- **SVM + Linear Kernel**: Good interpretability for business applications
- **Pipeline Architecture**: Proper ML workflow from data to deployment
- **Recall vs Precision Trade-off**: Understanding when to optimize for each metric

## 🔄 Migration Notes

This dashboard replaces the previous multi-model churn prediction app with a focused, educational tool that demonstrates:
- **SVM threshold optimization**
- **Recall maximization strategies**
- **Interactive ML visualization**
- **Technical implementation details**

Perfect for learning and demonstrating ML concepts in your AI course!
