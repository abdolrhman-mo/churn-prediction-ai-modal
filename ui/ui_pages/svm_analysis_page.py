import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_svm_analysis_page(df, results_df, y_test):
    """Render the SVM Threshold Analysis page"""
    st.markdown("## 📊 SVM Threshold Analysis")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Size", len(df))
    with col2:
        churn_rate = (df['Churn'] == 'Yes').mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}")
    with col3:
        st.metric("Features Used", len(df.columns)-1)
    with col4:
        st.metric("Test Set Size", len(y_test))
    
    st.markdown("### 🎯 Why Threshold 0.1 Achieves Highest Recall")
    
    # Threshold comparison chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results_df['Threshold'],
        y=results_df['Recall'],
        mode='lines+markers',
        name='Recall',
        line=dict(color='#ff6b6b', width=4),
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=results_df['Threshold'],
        y=results_df['Precision'],
        mode='lines+markers',
        name='Precision',
        line=dict(color='#4ecdc4', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=results_df['Threshold'],
        y=results_df['F1_Score'],
        mode='lines+markers',
        name='F1 Score',
        line=dict(color='#45b7d1', width=3)
    ))
    
    # Highlight the 0.1 threshold
    max_recall_idx = results_df['Recall'].idxmax()
    best_threshold = results_df.loc[max_recall_idx, 'Threshold']
    best_recall = results_df.loc[max_recall_idx, 'Recall']
    
    fig.add_vline(x=best_threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"Best Recall: {best_recall:.4f}")
    
    fig.update_layout(
        title="🎯 SVM Threshold vs Performance Metrics",
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation
    st.markdown(f"""
    ### 🎯 Key Insight: Threshold {best_threshold} = Maximum Recall
    
    **Why lower threshold = higher recall?**
    - **Lower threshold (0.1)**: Model says "Yes, will churn" even with low confidence
    - **Higher threshold (0.9)**: Model only says "Yes" when very confident
    - **For churn detection**: Better to catch all potential churners (even false alarms) than miss real ones
    
    **Your SVM at threshold 0.1:**
    - Catches **{best_recall:.1%}** of all customers who will actually churn
    - This means **{(1-best_recall)*100:.1f}% missed churn rate** (very low!)
    """)
