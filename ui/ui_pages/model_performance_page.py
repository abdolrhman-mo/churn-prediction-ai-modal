import streamlit as st
import plotly.express as px
import pandas as pd

def render_model_performance_page(results):
    """Render the model performance analysis page"""
    
    st.markdown("## 📊 Model Performance Analysis")
    
    # Performance comparison chart
    performance_df = pd.DataFrame(results).T
    performance_df = performance_df[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']]
    
    # Melt for plotting
    performance_melted = performance_df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
    performance_melted.rename(columns={'index': 'Model'}, inplace=True)
    
    fig = px.bar(performance_melted, x='Model', y='Score', color='Metric', barmode='group',
                title="Model Performance Comparison")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.markdown("### Detailed Performance Metrics")
    st.dataframe(performance_df.round(4))
    
    # Model comparison insights
    best_accuracy = performance_df['Accuracy'].idxmax()
    best_precision = performance_df['Precision'].idxmax()
    best_recall = performance_df['Recall'].idxmax()
    best_f1 = performance_df['F1 Score'].idxmax()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏆 Best Performers")
        st.write(f"**Accuracy**: {best_accuracy}")
        st.write(f"**Precision**: {best_precision}")
        st.write(f"**Recall**: {best_recall}")
        st.write(f"**F1 Score**: {best_f1}")
    
    with col2:
        st.markdown("### 💡 Recommendations")
        st.write("• **For Production**: Use the model with highest F1 Score")
        st.write("• **For High Precision**: Choose model with best Precision")
        st.write("• **To Catch All Churners**: Choose model with best Recall")
