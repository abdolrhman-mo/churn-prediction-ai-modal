import streamlit as st
from ui.data_manager import load_and_preprocess_data, train_models
from ui.pages.home_page import render_home_page
from ui.pages.single_prediction_page import render_single_prediction_page
from ui.pages.data_insights_page import render_data_insights_page
from ui.pages.model_performance_page import render_model_performance_page
from ui.pages.batch_predictions_page import render_batch_predictions_page
from ui.styling import apply_custom_css

# Apply custom styling
apply_custom_css()

# Configure the page
st.set_page_config(
    page_title="🔮 Churn Prediction Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.markdown('<h1 class="main-header">🔮 Telecom Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Predict customer churn and discover insights to improve retention")

# Sidebar for navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "🏠 Home & Model Training",
    "🔍 Single Customer Prediction", 
    "📈 Data Insights",
    "📊 Model Performance",
    "📋 Batch Predictions"
])

# Load data and train models
with st.spinner("Loading data and training models..."):
    df = load_and_preprocess_data()
    models, results, processing, scaler, X_test, y_test, label_encoder = train_models(df)

# Page routing
if page == "🏠 Home & Model Training":
    render_home_page(df, models, results)
elif page == "🔍 Single Customer Prediction":
    render_single_prediction_page(models, processing, scaler, label_encoder)
elif page == "📈 Data Insights":
    render_data_insights_page(df)
elif page == "📊 Model Performance":
    render_model_performance_page(results)
elif page == "📋 Batch Predictions":
    render_batch_predictions_page(models, results, processing, scaler, label_encoder)

# Footer
st.markdown("---")
st.markdown("### 🚀 How to Run This App")
st.markdown("""
1. **Install Streamlit**: `pip install streamlit`
2. **Save this code** as `ui/main_app.py`
3. **Run the app**: `streamlit run ui/main_app.py`
4. **Open in browser**: Usually at `http://localhost:8501`
""")
