import streamlit as st
import pandas as pd

def render_batch_predictions_page(models, results, processing, scaler, label_encoder):
    """Render the batch predictions page"""
    
    st.markdown("## 📋 Batch Predictions")
    st.markdown("Upload a CSV file to get churn predictions for multiple customers")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.markdown("### Uploaded Data Preview")
        st.dataframe(batch_df.head())
        
        if st.button("Generate Predictions"):
            # Use the best model for batch predictions
            best_model_name = pd.DataFrame(results).T['F1 Score'].idxmax()
            best_model = models[best_model_name]
            
            try:
                # Preprocess batch data
                batch_processed = processing.transform(batch_df)
                
                if best_model_name == 'Logistic Regression':
                    batch_scaled = scaler.transform(batch_processed)
                    predictions = best_model.predict(batch_scaled)
                    probabilities = best_model.predict_proba(batch_scaled)[:, 1]
                elif best_model_name == 'XGBoost':
                    predictions_encoded = best_model.predict(batch_processed)
                    predictions = label_encoder.inverse_transform(predictions_encoded)
                    probabilities = best_model.predict_proba(batch_processed)[:, 1]
                else:  # Random Forest
                    predictions = best_model.predict(batch_processed)
                    probabilities = best_model.predict_proba(batch_processed)[:, 1]
                
                # Add predictions to dataframe
                results_df = batch_df.copy()
                results_df['Predicted_Churn'] = predictions
                results_df['Churn_Probability'] = probabilities
                results_df['Risk_Level'] = pd.cut(probabilities, 
                                                bins=[0, 0.3, 0.7, 1.0], 
                                                labels=['Low', 'Medium', 'High'])
                
                st.markdown("### Prediction Results")
                st.dataframe(results_df)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Risk Customers", (results_df['Risk_Level'] == 'High').sum())
                with col2:
                    st.metric("Medium Risk Customers", (results_df['Risk_Level'] == 'Medium').sum())
                with col3:
                    st.metric("Low Risk Customers", (results_df['Risk_Level'] == 'Low').sum())
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name='churn_predictions.csv',
                    mime='text/csv'
                )
                
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.write("Please ensure your CSV has the same columns as the training data.")
