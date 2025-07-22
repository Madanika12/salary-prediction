import streamlit as st

# Load encoders and model (dummy placeholders here)
import joblib

# Replace with your actual model and encoders
model = joblib.load("salary_model.pkl")
label_encoders = {
    "job_title": joblib.load("job_title_encoder.pkl"),
    "location": joblib.load("location_encoder.pkl"),
    "education_level": joblib.load("education_encoder.pkl"),
    "company_size": joblib.load("company_size_encoder.pkl"),
}
mlb = joblib.load("skills_mlb.pkl")

# --------------------------- CSS STYLING --------------------------- #
st.markdown("""
    <style>
        /* Light blue labels and text */
        label, .css-1c7y2kd, .css-1d391kg, .css-16huue1, h3, h4, h5 {
            color: #00AEEF !important;
            font-size: 16px !important;
        }

        .title-style {
            color: #00AEEF;
            font-size: 30px;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .subheading {
            color: #888;
            font-size: 15px;
            margin-bottom: 30px;
        }

        .stButton > button {
            background-color: #00AEEF;
            color: white;
            font-size: 16px;
            padding: 0.6em 2em;
            border: none;
            border-radius: 5px;
        }

        .stButton > button:hover {
            background-color: #0096c7;
        }

        .card {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            max-width: 700px;
            margin: auto;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------- UI --------------------------- #
st.markdown("<h1 class='title-style'>Salary Predictor</h1>", unsafe_allow_html=True)

with st.form("salary_form"):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#00AEEF'><i class='fa fa-suitcase'></i> Salary Prediction Form</h4>", unsafe_allow_html=True)
    st.markdown("<p class='subheading'>Please fill in your details below</p>", unsafe_allow_html=True)

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

    submitted = st.form_submit_button("Get Salary Prediction")

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------- Prediction Logic --------------------------- #
if submitted:
    try:
        job_title_enc = label_encoders['job_title'].transform([job_title])[0]
        location_enc = label_encoders['location'].transform([location])[0]
        education_level_enc = label_encoders['education_level'].transform([education_level])[0]
        company_size_enc = label_encoders['company_size'].transform([company_size])[0]
        skills_encoded = mlb.transform([skills_list])[0]

        input_features = [job_title_enc, years_of_experience, location_enc, education_level_enc, company_size_enc]
        final_input = input_features + list(skills_encoded)

        predicted_salary = model.predict([final_input])[0]
        st.success(f"💰 Estimated Salary: ₹{predicted_salary:,.2f}")
    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")
