import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page Configuration
st.set_page_config(
    page_title="Student Admission & Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    .result-success {
        background-color: #d1fae5;
        color: #065f46;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        border: 1px solid #34d399;
    }
    .result-fail {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        border: 1px solid #f87171;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-header">🎓 Student Admission & Success Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by AdaBoost Machine Learning Classifier</div>', unsafe_allow_html=True)

# Load Model Safely
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'adaboost.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f), None
    else:
        return None, f"Model file 'decisionc.pkl' not found at {model_path}"

model, model_error = load_model()

if model_error:
    st.error(f"**Critical Error:** {model_error}\nPlease ensure `decisionc.pkl` is in the same directory as `app.py`.")
    st.stop()

# Input Layout using Form & Columns
with st.form("prediction_form"):
    st.subheader("📋 Enter Student Profile Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=15, max_value=100, value=18)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        state = st.text_input("State (e.g., CA, NY)", value="CA")
        family_income = st.number_input("Family Income ($)", min_value=0, value=75000, step=5000)
        high_school_gpa = st.number_input("High School GPA", min_value=0.0, max_value=4.0, value=3.5, step=0.01)
        sat_score = st.number_input("SAT Score", min_value=400, max_value=1600, value=1200, step=10)

    with col2:
        act_score = st.number_input("ACT Score", min_value=1, max_value=36, value=25, step=1)
        attendance_rate = st.slider("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=95.0, step=0.5)
        ap_courses = st.number_input("AP Courses Taken", min_value=0, max_value=15, value=3)
        extracurricular_count = st.number_input("Extracurricular Activities", min_value=0, max_value=20, value=2)
        volunteer_hours = st.number_input("Volunteer Hours", min_value=0, max_value=500, value=50)
        leadership_positions = st.number_input("Leadership Positions Held", min_value=0, max_value=10, value=1)

    with col3:
        coding_projects = st.number_input("Coding Projects Completed", min_value=0, max_value=50, value=1)
        social_media_hours = st.number_input("Social Media Hours/Day", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
        online_certifications = st.number_input("Online Certifications", min_value=0, max_value=20, value=1)
        essay_score = st.number_input("Essay Score (out of 100)", min_value=0.0, max_value=100.0, value=85.0, step=0.5)
        recommendation_score = st.number_input("Recommendation Score (out of 100)", min_value=0.0, max_value=100.0, value=88.0, step=0.5)
        interview_score = st.number_input("Interview Score (out of 100)", min_value=0.0, max_value=100.0, value=90.0, step=0.5)

    st.markdown("---")
    submit_button = st.form_submit_button(label="Run Prediction Analysis")

# Prediction Execution
if submit_button:
    try:
        # Encoding mappings for categorical features if needed by model encoding
        gender_map = {"Male": 1, "Female": 0, "Other": 2}
        
        # Build input dataframe matching model expected features exactly
        input_data = pd.DataFrame([{
            'age': float(age),
            'gender': gender_map.get(gender, 1),
            'state': str(state).strip().upper(),
            'family_income': float(family_income),
            'high_school_gpa': float(high_school_gpa),
            'sat_score': float(sat_score),
            'act_score': float(act_score),
            'attendance_rate': float(attendance_rate),
            'ap_courses': int(ap_courses),
            'extracurricular_count': int(extracurricular_count),
            'volunteer_hours': float(volunteer_hours),
            'leadership_positions': int(leadership_positions),
            'coding_projects': int(coding_projects),
            'social_media_hours': float(social_media_hours),
            'online_certifications': int(online_certifications),
            'essay_score': float(essay_score),
            'recommendation_score': float(recommendation_score),
            'interview_score': float(interview_score)
        }])

        # Handle string encoding for state if model expects ordinal/label encoding
        # If state was label encoded during training, map or convert categorical columns safely:
        for col in input_data.select_dtypes(include=['object']).columns:
            input_data[col] = input_data[col].astype('category').cat.codes

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None

        st.markdown("### 📊 Analysis Results")
        
        if str(prediction) in ['1', 'True', 'Accepted', 'Passed']:
            st.markdown('<div class="result-success">🎉 Predicted Outcome: SUCCESS / ACCEPTED</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-fail">⚠️ Predicted Outcome: REVIEW REQUIRED (Class: {prediction})</div>', unsafe_allow_html=True)

        if probability is not None:
            max_prob = np.max(probability) * 100
            st.metric(label="Model Prediction Confidence", value=f"{max_prob:.2f}%")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")