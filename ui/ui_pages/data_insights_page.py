import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

def render_data_insights_page(df):
    """Render the data insights and exploration page"""
    
    st.markdown("## 📈 Data Insights & Exploration")
    
    # Dataset overview
    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Churned Customers", (df['Churn'] == 'Yes').sum())
    with col3:
        st.metric("Retained Customers", (df['Churn'] == 'No').sum())
    with col4:
        st.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean():.1%}")
    
    # Interactive charts
    chart_option = st.selectbox("Choose Analysis:", [
        "Churn by Demographics",
        "Churn by Services", 
        "Financial Analysis",
        "Correlation Matrix"
    ])
    
    if chart_option == "Churn by Demographics":
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='gender', color='Churn', barmode='group',
                             title="Churn by Gender")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='SeniorCitizen', color='Churn', barmode='group',
                             title="Churn by Senior Citizen Status")
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_option == "Churn by Services":
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='InternetService', color='Churn', barmode='group',
                             title="Churn by Internet Service")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='Contract', color='Churn', barmode='group',
                             title="Churn by Contract Type")
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_option == "Financial Analysis":
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df, x='Churn', y='MonthlyCharges',
                        title="Monthly Charges Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x='tenure', y='TotalCharges', color='Churn',
                           title="Tenure vs Total Charges")
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_option == "Correlation Matrix":
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
