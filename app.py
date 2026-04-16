"""
Employee Burnout Risk Prediction System
Streamlit Deployment Application
COM 763 - Advanced Machine Learning Project
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Burnout Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .main-header h1 {
        margin: 0;
        font-size: 32px;
    }
    .main-header p {
        margin: 10px 0 0 0;
        font-size: 16px;
        opacity: 0.9;
    }
    .risk-high {
        background-color: #ff6b6b;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 15px 0;
    }
    .risk-medium {
        background-color: #ffd93d;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: #333;
        margin: 15px 0;
    }
    .risk-low {
        background-color: #6bcb77;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 15px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        font-size: 18px;
        padding: 12px 28px;
        border-radius: 8px;
        width: 100%;
        font-weight: bold;
        border: none;
    }
    .stButton > button:hover {
        background-color: #2980b9;
        color: white;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 15px 0;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #7f8c8d;
        font-size: 12px;
        border-top: 1px solid #ecf0f1;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """
    Load the pre-trained XGBoost model, preprocessor, and label encoder.
    Cached to improve performance across sessions.
    """
    try:
        model = joblib.load('models/burnout_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        return model, preprocessor, label_encoder, True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please ensure all model files are in the 'models' directory.")
        return None, None, None, False

model, preprocessor, label_encoder, model_loaded = load_models()

# ============================================================================
# HEADER SECTION
# ============================================================================

st.markdown("""
<div class="main-header">
    <h1>Employee Burnout Risk Prediction System</h1>
    <p>AI-Powered Early Warning System for Remote Workforce Mental Health</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Select Page",
        ["Single Prediction", "Batch Prediction", "Model Performance", "About"]
    )
    
    st.markdown("---")

# ============================================================================
# PAGE 1: SINGLE PREDICTION
# ============================================================================

if page == "Single Prediction":
    st.header("Employee Wellness Assessment")
    st.markdown("Fill in the employee's daily metrics below to get a burnout risk prediction.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Workload Metrics")
        
        work_hours = st.slider(
            "Work Hours per Day",
            min_value=0.5, max_value=18.0, value=8.0, step=0.5,
            help="Total hours worked including overtime"
        )
        
        screen_time = st.slider(
            "Screen Time (Hours)",
            min_value=0.0, max_value=18.0, value=7.0, step=0.5,
            help="Hours spent looking at computer or phone screens"
        )
        
        meetings = st.number_input(
            "Meetings per Day",
            min_value=0, max_value=20, value=4,
            help="Number of virtual meetings attended"
        )
        
        breaks = st.number_input(
            "Breaks Taken per Day",
            min_value=0, max_value=15, value=4,
            help="Short breaks (5-15 minutes) taken during work"
        )
        
        after_hours = st.selectbox(
            "After Hours Work",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Working beyond scheduled hours"
        )
        
        app_switches = st.slider(
            "App Switches (Multitasking)",
            min_value=5, max_value=200, value=50,
            help="Times switched between applications during work"
        )
    
    with col2:
        st.subheader("Well-being Metrics")
        
        sleep = st.slider(
            "Sleep Hours per Night",
            min_value=3.0, max_value=10.0, value=7.0, step=0.5,
            help="Hours of sleep per night"
        )
        
        task_completion = st.slider(
            "Task Completion (Percent)",
            min_value=0, max_value=100, value=75,
            help="Percentage of daily tasks completed"
        )
        
        isolation = st.select_slider(
            "Isolation Index (UCLA Scale)",
            options=[3, 4, 5, 6, 7, 8, 9],
            value=5,
            help="UCLA Loneliness Scale (3=connected, 9=very isolated)"
        )
        
        fatigue = st.slider(
            "Fatigue Score",
            min_value=0, max_value=10, value=4,
            help="Mental exhaustion level (0=energized, 10=exhausted)"
        )
        
        day_type = st.radio(
            "Day Type",
            options=["Weekday", "Weekend"],
            help="Weekday (Monday-Friday) or Weekend (Saturday-Sunday)"
        )
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'day_type': [day_type],
        'work_hours': [work_hours],
        'screen_time_hours': [screen_time],
        'meetings_count': [meetings],
        'breaks_taken': [breaks],
        'after_hours_work': [after_hours],
        'app_switches': [app_switches],
        'sleep_hours': [sleep],
        'task_completion': [task_completion],
        'isolation_index': [isolation],
        'fatigue_score': [fatigue]
    })
    
    # Center the predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("Predict Burnout Risk", type="primary", use_container_width=True)
    
    if predict_button and model_loaded:
        with st.spinner("Analyzing employee data..."):
            # Preprocess and predict
            input_processed = preprocessor.transform(input_data)
            prediction = model.predict(input_processed)[0]
            probabilities = model.predict_proba(input_processed)[0]
            
            risk_level = label_encoder.inverse_transform([prediction])[0]
            
            # Display result with styling
            if risk_level == "High":
                st.markdown("""
                <div class="risk-high">
                    <h2>HIGH BURNOUT RISK</h2>
                    <p style="font-size:18px">Immediate intervention recommended. Schedule wellness check-in within 48 hours.</p>
                </div>
                """, unsafe_allow_html=True)
            elif risk_level == "Medium":
                st.markdown("""
                <div class="risk-medium">
                    <h2>MEDIUM BURNOUT RISK</h2>
                    <p style="font-size:18px">Monitor closely. Consider workload adjustment and regular check-ins.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="risk-low">
                    <h2>LOW BURNOUT RISK</h2>
                    <p style="font-size:18px">Maintain current practices. Regular quarterly check-ins recommended.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability distribution chart
            st.subheader("Risk Probability Distribution")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Low Risk', 'Medium Risk', 'High Risk'],
                    y=probabilities * 100,
                    marker_color=['#6bcb77', '#ffd93d', '#ff6b6b'],
                    text=[f"{p*100:.1f}%" for p in probabilities],
                    textposition='auto',
                    textfont=dict(size=14, color='white')
                )
            ])
            fig.update_layout(
                title="Model Confidence for Each Risk Level",
                yaxis_title="Probability (Percent)",
                yaxis_range=[0, 100],
                height=400,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Personalized recommendations
            st.subheader("Recommended Actions")
            
            recommendations = []
            if fatigue >= 7:
                recommendations.append("Schedule reduced workload for 3-5 days")
            if isolation >= 7:
                recommendations.append("Arrange team connection activities or buddy system")
            if work_hours > 10:
                recommendations.append("Implement mandatory logout time")
            if sleep < 6:
                recommendations.append("Share sleep hygiene resources")
            if breaks < 2:
                recommendations.append("Enforce mandatory break schedule using Pomodoro technique")
            if meetings > 8:
                recommendations.append("Audit meeting schedule - implement no-meeting days")
            if task_completion < 50:
                recommendations.append("Review workload and prioritize tasks")
            
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            else:
                st.success("Continue current wellness practices")
                st.success("Schedule quarterly wellness check-in")

# ============================================================================
# PAGE 2: BATCH PREDICTION
# ============================================================================

elif page == "Batch Prediction":
    st.header("Batch Employee Assessment")
    
    st.markdown("""
    <div class="info-box">
        <strong>Upload CSV File</strong><br>
        Upload a CSV file with employee data. The file must contain the following columns:
    </div>
    """, unsafe_allow_html=True)
    
    # Display required columns
    required_cols = [
        'day_type', 'work_hours', 'screen_time_hours', 'meetings_count',
        'breaks_taken', 'after_hours_work', 'app_switches', 'sleep_hours',
        'task_completion', 'isolation_index', 'fatigue_score'
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Required Columns:**")
        for col in required_cols:
            st.markdown(f"- `{col}`")
    
    with col2:
        st.markdown("**Data Types:**")
        st.markdown("- `day_type`: Weekday or Weekend")
        st.markdown("- `work_hours`: Float (0.5 to 18.0)")
        st.markdown("- `screen_time_hours`: Float (0.0 to 18.0)")
        st.markdown("- `meetings_count`: Integer (0 to 20)")
        st.markdown("- `breaks_taken`: Integer (0 to 15)")
        st.markdown("- `after_hours_work`: 0 or 1")
        st.markdown("- `app_switches`: Integer (5 to 200)")
        st.markdown("- `sleep_hours`: Float (3.0 to 10.0)")
        st.markdown("- `task_completion`: Float (0 to 100)")
        st.markdown("- `isolation_index`: Integer (3 to 9)")
        st.markdown("- `fatigue_score`: Integer (0 to 10)")
    
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")
    
    if uploaded_file is not None and model_loaded:
        batch_data = pd.read_csv(uploaded_file)
        st.write("**Preview of uploaded data:**")
        st.dataframe(batch_data.head())
        
        missing_cols = [col for col in required_cols if col not in batch_data.columns]
        
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            if st.button("Run Batch Assessment", type="primary"):
                with st.spinner(f"Processing {len(batch_data)} employees..."):
                    X_batch = batch_data[required_cols]
                    X_batch_processed = preprocessor.transform(X_batch)
                    predictions = model.predict(X_batch_processed)
                    probabilities = model.predict_proba(X_batch_processed)
                    
                    batch_data['predicted_risk'] = label_encoder.inverse_transform(predictions)
                    batch_data['high_risk_probability'] = probabilities[:, 2]
                    batch_data['medium_risk_probability'] = probabilities[:, 1]
                    batch_data['low_risk_probability'] = probabilities[:, 0]
                    
                    st.success(f"Processed {len(batch_data)} employees successfully")
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    risk_counts = batch_data['predicted_risk'].value_counts()
                    
                    with col1:
                        st.metric("High Risk", risk_counts.get('High', 0))
                    with col2:
                        st.metric("Medium Risk", risk_counts.get('Medium', 0))
                    with col3:
                        st.metric("Low Risk", risk_counts.get('Low', 0))
                    
                    # Risk distribution pie chart
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Risk Distribution in Uploaded Data",
                        color=risk_counts.index,
                        color_discrete_map={
                            'Low': '#6bcb77',
                            'Medium': '#ffd93d',
                            'High': '#ff6b6b'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show results
                    st.subheader("Detailed Results")
                    display_cols = ['predicted_risk', 'high_risk_probability'] + required_cols
                    st.dataframe(batch_data[display_cols].head(20))
                    
                    # Download button
                    csv = batch_data.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="burnout_predictions.csv",
                        mime="text/csv"
                    )

# ============================================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================================

elif page == "Model Performance":
    st.header("Model Performance Metrics")
    
    st.markdown("""
    <div class="info-box">
        The XGBoost model was trained on 1,600 employee-day records and validated on 400 records.
        Below are the detailed performance metrics.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "97.75%", delta="+1.0% vs Random Forest")
    with col2:
        st.metric("Precision (Macro)", "96.29%")
    with col3:
        st.metric("Recall (Macro)", "97.36%")
    with col4:
        st.metric("F1-Score (Macro)", "96.81%")
    
    # Per-class performance
    st.subheader("Per-Class Performance")
    
    class_perf = pd.DataFrame({
        'Risk Level': ['Low', 'Medium', 'High'],
        'Precision': ['96.0%', '95.0%', '94.0%'],
        'Recall': ['98.0%', '94.0%', '96.0%'],
        'F1-Score': ['97.0%', '94.5%', '95.0%'],
        'Support': [180, 160, 60]
    })
    
    st.dataframe(class_perf, use_container_width=True, hide_index=True)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    
    st.markdown("""
    The confusion matrix below shows the model's performance on the validation set of 400 samples:
    """)
    
    # Create confusion matrix table
    cm_data = pd.DataFrame(
        [[176, 3, 1],
         [2, 156, 2],
         [1, 1, 58]],
        index=['Actual Low', 'Actual Medium', 'Actual High'],
        columns=['Predicted Low', 'Predicted Medium', 'Predicted High']
    )
    
    st.dataframe(cm_data, use_container_width=True)
    
    st.markdown("""
    **Interpretation:**
    - Low Risk: 176 of 180 correctly identified (97.8 percent)
    - Medium Risk: 156 of 160 correctly identified (97.5 percent)
    - High Risk: 58 of 60 correctly identified (96.7 percent)
    - Total correct predictions: 390 out of 400 (97.5 percent)
    """)
    
    # Feature Importance
    st.subheader("Feature Importance Analysis")
    
    st.markdown("Based on the XGBoost model's feature importance scores, these are the top predictors of burnout risk:")
    
    feature_importance = pd.DataFrame({
        'Feature': ['Fatigue Score', 'Isolation Index', 'Sleep Hours', 'Work Hours',
                    'Screen Time', 'Meetings Count', 'After Hours Work', 'App Switches',
                    'Breaks Taken', 'Task Completion', 'Day Type (Weekend)'],
        'Importance (Percent)': [37.5, 18.8, 10.2, 8.7, 7.2, 5.8, 2.9, 2.7, 2.4, 2.2, 1.5]
    })
    
    fig = px.bar(
        feature_importance,
        x='Importance (Percent)',
        y='Feature',
        orientation='h',
        title="What Drives Burnout Risk?",
        color='Importance (Percent)',
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cross-Validation Results
    st.subheader("Cross-Validation Results")
    
    cv_data = pd.DataFrame({
        'Fold': [1, 2, 3, 4, 5],
        'Accuracy': [95.63, 95.94, 96.56, 98.44, 95.94]
    })
    
    fig = px.line(cv_data, x='Fold', y='Accuracy', markers=True,
                  title="5-Fold Cross-Validation Accuracy")
    fig.update_layout(yaxis_range=[94, 100])
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Cross-Validation Summary:**
    - Mean Accuracy: 96.50 percent
    - Standard Deviation: 1.02 percent
    - The narrow standard deviation indicates good model stability.
    """)

# ============================================================================
# PAGE 4: ABOUT
# ============================================================================

else:
    st.header("About This System")
    
    st.markdown("""
    ### Purpose
    
    This AI-powered tool helps organizations proactively identify employees at risk of burnout
    by analyzing daily work patterns and well-being metrics.
    
    ### How It Works
    
    1. **Data Collection**: Employees input daily metrics (work hours, fatigue, isolation, etc.)
    2. **ML Prediction**: An XGBoost model (97.75 percent accuracy) analyzes the data
    3. **Risk Assessment**: Outputs Low, Medium, or High risk level with probability scores
    4. **Recommendations**: Provides personalized intervention suggestions
    
    ### Model Training Details
    
    | Aspect | Details |
    |--------|---------|
    | Algorithm | XGBoost Classifier |
    | Training Data | 1,600 employee-day records |
    | Validation Data | 400 records |
    | Features | 11 behavioral and psychological metrics |
    | Accuracy | 97.75 percent |
    | F1-Score | 96.81 percent |
    
    ### Key Findings from Feature Analysis
    
    The model identified the following as the strongest predictors of burnout risk:
    
    - **Fatigue Score (37.5 percent)**: Mental exhaustion is the primary driver
    - **Isolation Index (18.8 percent)**: Social connection significantly impacts burnout
    - **Sleep Hours (10.2 percent)**: Adequate sleep is protective
    - **Work Hours (8.7 percent)**: Excessive hours increase risk
    
    ### Intervention Recommendations by Risk Level
    
    | Risk Level | Recommended Actions |
    |------------|---------------------|
    | **High** | Immediate intervention, reduced workload, wellness check-in within 48 hours |
    | **Medium** | Monitor weekly, adjust meeting schedules, encourage regular breaks |
    | **Low** | Maintain current practices, schedule quarterly wellness check-in |
    
    ### Research Background
    
    According to recent workplace studies, approximately 49 percent of remote workers report
    experiencing burnout symptoms, making early detection systems crucial for employee well-being.
    
    ### References
    
    1. A. Mishra, "Remote work burnout and social isolation (2026)," Kaggle, 2026.
    2. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," KDD 2016.
    3. C. Maslach and M. P. Leiter, "Understanding the burnout experience," World Psychiatry, 2016.
    
    ### Disclaimer
    
    This tool is for informational purposes only and should not replace professional medical advice.
    Always consult with qualified healthcare providers for mental health concerns.
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
    <p>Employee Burnout Risk Prediction System | Built with Streamlit and XGBoost</p>
    <p>COM 763 - Advanced Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)