import streamlit as st
import plotly.express as px
import pandas as pd

def render_home_page(df, models, results):
    """Render the home page with project overview and model training results"""
    
    st.markdown("## 🎯 Project Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(df), help="Total customers in dataset")
    with col2:
        churn_rate = (df['Churn'] == 'Yes').mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}", help="Percentage of customers who churned")
    with col3:
        st.metric("Features", len(df.columns)-1, help="Number of features used for prediction")
    
    st.markdown("## 🤖 Model Performance Summary")
    
    # Create performance comparison
    performance_df = pd.DataFrame(results).T
    performance_df = performance_df[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']]
    
    st.dataframe(performance_df.round(4))
    
    # Best model highlight
    best_model = performance_df['F1 Score'].idxmax()
    best_f1 = performance_df.loc[best_model, 'F1 Score']
    
    st.success(f"🏆 **Best Model**: {best_model} (F1 Score: {best_f1:.4f})")
    
    # Quick insights
    st.markdown("## 💡 Quick Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn by contract type
        fig = px.histogram(df, x='Contract', color='Churn', barmode='group',
                          title="Churn by Contract Type")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly charges distribution
        fig = px.box(df, x='Churn', y='MonthlyCharges',
                    title="Monthly Charges vs Churn")
        st.plotly_chart(fig, use_container_width=True)
