import streamlit as st
import joblib
import pandas as pd

# Load your model and encoders
model = joblib.load("best_salary_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
mlb = joblib.load("skills_mlb.pkl")

# ----------------- CSS: Light Blue Text & Font Size -----------------
st.markdown("""
    <style>
        label, .css-1c7y2kd, .css-1d391kg, .css-16huue1, h3, h4, h5 {
            color: #00AEEF !important;
            font-size: 16px !important;
        }

        .title-style {
            color: #00AEEF;
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .stButton > button {
            background-color: #00AEEF;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 10px 25px;
        }

        .stButton > button:hover {
            background-color: #0096c7;
        }

        .block-container {
            padding-top: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- App Title -----------------
st.markdown("<h1 class='title-style'>Salary Predictor</h1>", unsafe_allow_html=True)

# ----------------- Form -----------------
with st.form("salary_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Job Title**", unsafe_allow_html=True)
        job_title = st.selectbox("", label_encoders['job_title'].classes_)

    with col2:
        st.markdown("**Years of Experience**", unsafe_allow_html=True)
        years_of_experience = st.number_input("", 0, 50, 2)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Location**", unsafe_allow_html=True)
        location = st.selectbox("", label_encoders['location'].classes_)

    with col4:
        st.markdown("**Education Level**", unsafe_allow_html=True)
        education_level = st.selectbox("", label_encoders['education_level'].classes_)

    st.markdown("**Company Size**", unsafe_allow_html=True)
    company_size = st.selectbox("", label_encoders['company_size'].classes_)

    st.markdown("**Skills & Technologies**", unsafe_allow_html=True)
    skills_list = st.multiselect("", mlb.classes_)

    submitted = st.form_submit_button("Predict My Salary")

# ----------------- Prediction -----------------
if submitted:
    try:
        # Encode input
        input_data = {
            'job_title': label_encoders['job_title'].transform([job_title])[0],
            'years_of_experience': years_of_experience,
            'location': label_encoders['location'].transform([location])[0],
            'education_level': label_encoders['education_level'].transform([education_level])[0],
            'company_size': label_encoders['company_size'].transform([company_size])[0]
        }

        # Encode skills
        skills_encoded = mlb.transform([skills_list])
        skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_)

        # Merge all inputs
        input_df = pd.DataFrame([input_data])
        final_input = pd.concat([input_df, skills_df], axis=1)

        # Handle missing features
        for col in model.feature_names_in_:
            if col not in final_input.columns:
                final_input[col] = 0
        final_input = final_input[model.feature_names_in_]

        # Predict salary
        predicted_salary = model.predict(final_input)[0]
        st.success(f"💰 Estimated Salary: ₹{predicted_salary:,.2f}")

    except Exception as e:
        st.error(f"⚠️ Error in prediction: {e}")
