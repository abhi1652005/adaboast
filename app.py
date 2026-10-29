import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration with a professional layout and theme
st.set_page_config(
    page_title="Student Performance & Admission Classifier",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a polished look
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #F9FAFB;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">🎓 Student Evaluation Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by an AdaBoost Machine Learning Model to predict outcomes based on academic and extracurricular profile data.</div>', unsafe_allow_html=True)

# Load Model with Caching and Error Handling
@st.cache_resource
def load_model():
    try:
        with open('decisionc.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, str(e)

model, model_error = load_model()

if model_error:
    st.error(f"⚠️ Failed to load the model file (`decisionc.pkl`).")
    st.error(f"**Error Details:** {model_error}")
    st.info("Ensure `decisionc.pkl` is uploaded in the root directory of your GitHub repository.")
    st.stop()

# Sidebar instructions
st.sidebar.header("📋 Instructions")
st.sidebar.info("Fill out the applicant's details across the sections on the main page, then click **Run Prediction** at the bottom.")

# --- INPUT FORM LAYOUT ---
with st.form("prediction_form"):
    
    st.subheader("👤 Demographics & Background")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=15, max_value=100, value=18, step=1)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col3:
        state = st.text_input("State", value="California")

    st.markdown("---")
    st.subheader("💰 Family & Financial Background")
    family_income = st.number_input("Family Income ($)", min_value=0, value=75000, step=5000)

    st.markdown("---")
    st.subheader("📚 Academic Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_school_gpa = st.number_input("High School GPA", min_value=0.0, max_value=4.0, value=3.5, step=0.01)
    with col2:
        sat_score = st.number_input("SAT Score", min_value=400, max_value=1600, value=1200, step=10)
    with col3:
        act_score = st.number_input("ACT Score", min_value=1, max_value=36, value=25, step=1)
    with col4:
        attendance_rate = st.slider("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=95.0, step=0.5)

    col1, col2 = st.columns(2)
    with col1:
        ap_courses = st.number_input("AP Courses Taken", min_value=0, max_value=20, value=3, step=1)
    with col2:
        essay_score = st.slider("Essay Score", min_value=0.0, max_value=100.0, value=85.0, step=1.0)

    st.markdown("---")
    st.subheader("🏆 Extracurriculars & Achievements")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        extracurricular_count = st.number_input("Extracurricular Activities", min_value=0, max_value=20, value=2, step=1)
    with col2:
        volunteer_hours = st.number_input("Volunteer Hours", min_value=0, max_value=1000, value=50, step=5)
    with col3:
        leadership_positions = st.number_input("Leadership Positions", min_value=0, max_value=10, value=1, step=1)
    with col4:
        coding_projects = st.number_input("Coding Projects", min_value=0, max_value=50, value=2, step=1)

    col
