# 🔮 Churn Prediction Dashboard - UI Module Structure

This directory contains the refactored, modular version of the Churn Prediction Dashboard.

## 📁 Directory Structure

```
ui/
├── __init__.py                 # Package initialization
├── main_app.py                 # Main Streamlit application
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

## ✨ Benefits of the New Structure

1. **Modularity**: Each page is now a separate module, making it easier to edit specific features
2. **Maintainability**: Code is organized by functionality, reducing complexity
3. **Reusability**: Functions can be imported and reused across different pages
4. **Testing**: Individual modules can be tested independently
5. **Collaboration**: Multiple developers can work on different modules simultaneously

## 🔧 Making Edits

### To Edit a Specific Page:
- Navigate to `ui/pages/` and edit the corresponding `.py` file
- Each page module contains a `render_*_page()` function that handles all the UI logic

### To Edit Data Processing:
- Modify `ui/data_manager.py` for changes to data loading, preprocessing, or model training

### To Edit Styling:
- Modify `ui/styling.py` for changes to CSS and visual appearance

### To Edit the Main App:
- Modify `ui/main_app.py` for changes to navigation, page routing, or overall structure

## 📝 Example: Adding a New Page

1. Create a new file in `ui/pages/` (e.g., `new_page.py`)
2. Define a `render_new_page()` function
3. Import it in `ui/main_app.py`
4. Add it to the navigation sidebar
5. Add the routing logic

## 🔄 Migration from Original app.py

The original `app.py` has been completely refactored into these modules. All functionality is preserved, but now organized for better maintainability.

## 📊 Dependencies

All dependencies remain the same as in the original `requirements.txt`. The refactoring only changes the code organization, not the external libraries used.
