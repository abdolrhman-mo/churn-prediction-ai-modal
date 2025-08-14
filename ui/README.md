# 🎯 SVM Recall Optimization Dashboard

A specialized Streamlit dashboard for churn prediction using Support Vector Machines (SVM) with threshold optimization for maximum recall.

## 🏗️ Architecture

The dashboard is built with a modular architecture for better maintainability and extensibility:

```
ui/
├── main_app.py                 # Main Streamlit application orchestrator
├── data_manager.py            # Data loading, preprocessing, and model training
├── ui_pages/                  # Individual page modules
│   ├── __init__.py           # Package initialization
│   ├── svm_analysis_page.py  # SVM Threshold Analysis page
│   └── technical_implementation_page.py  # Technical details page
└── README.md                  # This file
```

## 🚀 Features

### 📊 SVM Threshold Analysis Page
- **Dataset Overview**: Size, churn rate, features, test set metrics
- **Threshold Comparison**: Interactive charts showing recall vs precision trade-offs
- **Optimal Threshold**: Automatic identification of threshold for maximum recall
- **Business Insights**: Clear explanations of why lower thresholds achieve higher recall

### ⚙️ Technical Implementation Page
- **ML Pipeline Architecture**: Complete workflow visualization
- **Code Components**: Detailed explanations of key implementation parts
- **Project Phases**: Complete 9-phase project overview
- **Technical Achievements**: Summary of implemented features
- **Dataset Information**: Feature breakdown and success metrics

## 🛠️ Data Management

The `data_manager.py` provides comprehensive functionality:

- **Data Loading**: Automatic CSV loading with fallback to demo data
- **Preprocessing**: Complete pipeline following the original implementation
- **Model Training**: SVM training with SMOTE balancing and threshold optimization
- **Model Persistence**: Save/load trained models and components
- **Prediction**: Make churn predictions on new customer data

### Key Classes

#### `ChurnDataManager`
- Manages the complete ML workflow
- Handles data loading, preprocessing, and model training
- Provides utility functions for model saving/loading

### Key Functions

- `load_and_process_data()`: Load and preprocess churn data
- `train_svm_model()`: Train SVM with threshold optimization
- `save_model()`: Persist trained model components
- `predict_churn()`: Make predictions on new data

## 📁 File Structure

- **`main_app.py`**: Main application entry point with navigation and page routing
- **`data_manager.py`**: Core data management and ML functionality
- **`ui_pages/svm_analysis_page.py`**: SVM threshold analysis visualization
- **`ui_pages/technical_implementation_page.py`**: Technical documentation and implementation details

## 🎯 Key Benefits

1. **Modular Design**: Easy to maintain and extend individual components
2. **Separation of Concerns**: Data management separate from UI rendering
3. **Reusable Components**: Page modules can be easily modified or extended
4. **Clean Architecture**: Clear separation between data, logic, and presentation
5. **Easy Testing**: Individual components can be tested independently

## 🚀 Running the Dashboard

1. **Install Dependencies**: Ensure all required packages are installed
2. **Run Main App**: Execute `python ui/main_app.py` or use Streamlit
3. **Navigate**: Use the sidebar to switch between analysis and technical pages
4. **Explore**: Interact with charts and explore the complete ML pipeline

## 🔧 Customization

- **Add New Pages**: Create new modules in `ui_pages/` and import them in `main_app.py`
- **Modify Data Pipeline**: Update `data_manager.py` for different preprocessing approaches
- **Extend Models**: Add new algorithms to the `ChurnDataManager` class
- **Custom Styling**: Modify CSS in `main_app.py` for different visual themes

## 📊 Technical Highlights

- **SVM with Linear Kernel**: Good interpretability for business applications
- **SMOTE Balancing**: Handles imbalanced churn datasets effectively
- **Threshold Optimization**: Automatically finds optimal decision boundaries
- **Pipeline Architecture**: Robust preprocessing and training workflow
- **Model Persistence**: Production-ready model saving and loading

This modular structure makes the dashboard easy to maintain, extend, and customize while preserving all the original functionality for SVM recall optimization.
