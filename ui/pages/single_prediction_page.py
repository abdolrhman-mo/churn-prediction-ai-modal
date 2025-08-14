import streamlit as st
from ui.data_manager import predict_single_customer

def render_single_prediction_page(models, processing, scaler, label_encoder):
    """Render the single customer prediction page"""
    
    st.markdown("## 🔍 Single Customer Prediction")
    st.markdown("Enter customer details to predict churn probability")
    
    # Input form
    with st.form("customer_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ['Male', 'Female'])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: 'Yes' if x else 'No')
            partner = st.selectbox("Partner", ['Yes', 'No'])
            dependents = st.selectbox("Dependents", ['Yes', 'No'])
        
        with col2:
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
            internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
            contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        
        with col3:
            monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 65.0)
            total_charges = st.slider("Total Charges ($)", 20.0, 8500.0, 1500.0)
            
            # Additional services
            online_security = st.selectbox("Online Security", ['Yes', 'No'])
            tech_support = st.selectbox("Tech Support", ['Yes', 'No'])
        
        # Additional service columns (simplified)
        online_backup = 'No'
        device_protection = 'No' 
        streaming_tv = 'No'
        streaming_movies = 'No'
        
        submitted = st.form_submit_button("🔮 Predict Churn", type="primary")
        
        if submitted:
            customer_data = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Make prediction
            predictions, probabilities = predict_single_customer(models, processing, scaler, customer_data, label_encoder)
            
            # Display results
            st.markdown("## 🎯 Prediction Results")
            
            for model_name in models.keys():
                pred = predictions[model_name]
                prob = probabilities[model_name]
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if pred == 'Yes':
                        st.markdown(f'<div class="prediction-box churn-yes">❌ WILL CHURN</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="prediction-box churn-no">✅ WILL STAY</div>', 
                                  unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**{model_name}**")
                    st.progress(prob)
                    st.markdown(f"Churn Probability: {prob:.1%}")
