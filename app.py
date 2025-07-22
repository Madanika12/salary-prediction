import streamlit as st
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load('best_salary_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
mlb = joblib.load('skills_mlb.pkl')

# --- CSS for white BG, clean card, and colored icons/text ---
st.markdown("""
<style>
body { background: #fff !important; }
.form-card {
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 3px 16px #f0f3fa;
    padding: 36px 36px 24px 36px;
    margin: 28px auto 0 auto;
    width: 100%;
    max-width: 720px;
    border: 1.5px solid #e5e7eb;
}
.form-label {
    font-weight: 700;
    font-size: 1.11em;
    color: #232b36;
    margin-bottom: 0.2em;
    display: flex;
    align-items: center;
    gap: 0.35em;
}
.stSelectbox>div>div { color: #232b36 !important; }
.stNumberInput>div>div { color: #232b36 !important; }
.stMultiSelect>div>div { color: #232b36 !important; }
.stButton>button {
    background: #fff;
    border: 1.5px solid #232b36;
    border-radius: 8px;
    color: #232b36;
    font-weight: 700;
    font-size: 1.1em;
    padding: 0.6em 0.5em;
    margin-top: 0.7em;
    transition: 0.2s;
}
.stButton>button:hover {
    border-color: #0f5ef7;
    color: #0f5ef7;
}
@media (max-width: 700px) {
    .form-card { padding: 12px 6px 10px 6px; }
}
</style>
""", unsafe_allow_html=True)

# --- FORM UI ---
st.markdown("""
<div class="form-card">
""", unsafe_allow_html=True)
with st.form("salary_form"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="form-label">🏢 Job Title</div>', unsafe_allow_html=True)
        job_title = st.selectbox("", label_encoders['job_title'].classes_, key='job_title')
    with col2:
        st.markdown('<div class="form-label">⏳ Years of Experience</div>', unsafe_allow_html=True)
        years_of_experience = st.number_input("", 0, 50, 2, key='yoe')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="form-label">📍 Location</div>', unsafe_allow_html=True)
        location = st.selectbox("", label_encoders['location'].classes_, key='location')
    with col2:
        st.markdown('<div class="form-label">🎓 Education Level</div>', unsafe_allow_html=True)
        education_level = st.selectbox("", label_encoders['education_level'].classes_, key='edu')
    st.markdown('<div class="form-label">🏢 Company Size</div>', unsafe_allow_html=True)
    company_size = st.selectbox("", label_encoders['company_size'].classes_, key='company')
    st.markdown('<div class="form-label" style="margin-top:8px;">&lt;/&gt; Skills & Technologies</div>', unsafe_allow_html=True)
    skills_list = st.multiselect("", options=mlb.classes_, key="skills")
    submitted = st.form_submit_button("Get Salary Prediction")
st.markdown("</div>", unsafe_allow_html=True)
