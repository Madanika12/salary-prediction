import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load('best_salary_mode2.pkl')
    scaler = joblib.load('scaler2.pkl')
    return model, scaler

model, scaler = load_model()

st.title("💼 Employee Salary Prediction App")
st.markdown("Predict salary based on personal and professional attributes")

with st.form("salary_prediction_form"):
    seniority = st.selectbox("Seniority Level", ['Junior', 'Mid', 'Senior', 'Lead', 'Other'])
    age = st.slider("Age", 18, 70, 30)
    job_title = st.selectbox("Job Title", ['Data Scientist', 'Data Engineer', 'Analyst', 'Software Engineer', 'Other'])
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    education = st.selectbox("Education Level", ['High School', "Bachelor's", "Master's", "PhD"])
    years_exp = st.slider("Years of Experience", 0, 40, 2)

    st.markdown("### Technical Skills (Select those you have):")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: python_yn = st.checkbox("Python")
    with col2: R_yn = st.checkbox("R")
    with col3: spark = st.checkbox("Spark")
    with col4: aws = st.checkbox("AWS")
    with col5: excel = st.checkbox("Excel")

    submit = st.form_submit_button("Predict Salary 💰")

def encode_inputs(seniority, age, job_title, gender, education, years_exp, python_yn, R_yn, spark, aws, excel):
    mapping = {
        'Senior': {'Junior': 0, 'Mid': 1, 'Senior': 2, 'Lead': 3, 'Other': 4},
        'Job Title': {'Data Scientist': 0, 'Data Engineer': 1, 'Analyst': 2, 'Software Engineer': 3, 'Other': 4},
        'Gender': {'Male': 0, 'Female': 1, 'Other': 2},
        'Education Level': {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
    }
    return np.array([[
        mapping['Senior'][seniority],
        age,
        mapping['Job Title'][job_title],
        mapping['Gender'][gender],
        mapping['Education Level'][education],
        years_exp,
        int(python_yn),
        int(R_yn),
        int(spark),
        int(aws),
        int(excel)
    ]])

if submit:
    try:
        input_data = encode_inputs(seniority, age, job_title, gender, education, years_exp, python_yn, R_yn, spark, aws, excel)
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        st.success(f"💵 Estimated Salary: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
