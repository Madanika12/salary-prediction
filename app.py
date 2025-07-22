import streamlit as st
import joblib
import pandas as pd

# --- Load model and encoders ---
model = joblib.load('best_salary_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
mlb = joblib.load('skills_mlb.pkl')

# --- Session State setup ---
if 'page' not in st.session_state:
    st.session_state.page = 'form'
if 'predicted_salary' not in st.session_state:
    st.session_state.predicted_salary = None
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}

def predict_salary(job_title, years_of_experience, location, education_level, company_size, skills_list):
    input_data = {
        'job_title': label_encoders['job_title'].transform([job_title])[0],
        'years_of_experience': years_of_experience,
        'location': label_encoders['location'].transform([location])[0],
        'education_level': label_encoders['education_level'].transform([education_level])[0],
        'company_size': label_encoders['company_size'].transform([company_size])[0]
    }
    skills_encoded = mlb.transform([skills_list])
    skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_)
    input_df = pd.DataFrame([input_data])
    final_input = pd.concat([input_df, skills_df], axis=1)
    for col in model.feature_names_in_:
        if col not in final_input.columns:
            final_input[col] = 0
    final_input = final_input[model.feature_names_in_]
    predicted = model.predict(final_input)[0]
    return round(predicted, 2)

# --- Custom CSS based on reference images for a simple, modern, not-large-text UI ---
st.markdown("""
    <style>
    .main-card {
        background: #fff;
        border-radius: 16px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.08);
        padding: 30px 32px 22px 32px;
        max-width: 540px;
        margin: 48px auto;
    }
    .title {
        color: #00b6ff;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 8px;
        text-align: center;
    }
    .desc {
        color: #444;
        text-align: center;
        margin-bottom: 26px;
        font-size: 1.08rem;
    }
    .section-label {
        font-weight: 500;
        margin-top: 8px;
        margin-bottom: 6px;
        color: #2574a9;
    }
    .skills-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5em 1em;
        margin-top: 8px;
        margin-bottom: 18px;
    }
    .skill-chip {
        background: #e9f4fb;
        color: #1796d2;
        border-radius: 7px;
        padding: 4px 13px;
        font-size: 1em;
        margin-bottom: 5px;
    }
    .stButton>button {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: #fff;
        border: none;
        border-radius: 7px;
        padding: 0.7em 0;
        font-weight: 500;
        font-size: 1.08em;
        width: 100%;
        margin-top: 12px;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #0072ff, #00c6ff);
    }
    </style>
""", unsafe_allow_html=T

# -------- FORM PAGE --------
if st.session_state.page == 'form':
    st.markdown('<div class="center-card">', unsafe_allow_html=True)
    st.markdown('<div class="heading-main">Salary Prediction Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="heading-sub">Enter your details to get an accurate salary prediction</div>', unsafe_allow_html=True)
    with st.form("salary_form"):
        # --- Job Title & Years of Experience ---
        st.markdown(
            """<div class="form-row">
                <div class="form-col">
                    <div class="form-label"><span class="form-icon">🏢</span>Job Title</div>
                    """, unsafe_allow_html=True)
        job_title = st.selectbox("", label_encoders['job_title'].classes_, key='job_title')
        st.markdown("""</div>
                <div class="form-col">
                    <div class="form-label"><span class="form-icon">🕒</span>Years of Experience</div>
            """, unsafe_allow_html=True)
        years_of_experience = st.number_input("", 0, 50, 2, key='yoe', placeholder="e.g. 5")
        st.markdown("""</div></div>""", unsafe_allow_html=True)
        # --- Location & Education Level ---
        st.markdown(
            """<div class="form-row">
                <div class="form-col">
                    <div class="form-label"><span class="form-icon">📍</span>Location</div>
            """, unsafe_allow_html=True)
        location = st.selectbox("", label_encoders['location'].classes_, key='location')
        st.markdown("""</div>
                <div class="form-col">
                    <div class="form-label"><span class="form-icon">🎓</span>Education Level</div>
            """, unsafe_allow_html=True)
        education_level = st.selectbox("", label_encoders['education_level'].classes_, key='edu')
        st.markdown("""</div></div>""", unsafe_allow_html=True)
        # --- Company Size ---
        st.markdown(
            """<div class="form-col" style="margin-bottom:0;">
                    <div class="form-label"><span class="form-icon">🏢</span>Company Size</div>
            """, unsafe_allow_html=True)
        company_size = st.selectbox("", label_encoders['company_size'].classes_, key='company')
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        # --- Skills ---
        st.markdown('<div class="skills-label"><span class="form-icon">⚡</span>Skills & Technologies</div>', unsafe_allow_html=True)
        skills = [
            "JavaScript", "Python", "React",
            "Node.js", "AWS", "Docker",
            "Machine Learning", "SQL", "Project Management",
            "Leadership", "Data Analysis", "Marketing",
            "Sales", "Design"
        ]
        st.markdown('<div class="skills-grid">', unsafe_allow_html=True)
        for skill in skills:
            st.markdown(f'<span class="skill-pill">{skill}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        # --- Button ---
        submitted = st.form_submit_button("Predict My Salary")
        if submitted:
            skills_list = []  # For now, empty or adapt as needed
            salary = predict_salary(job_title, years_of_experience, location, education_level, company_size, skills_list)
            st.session_state.predicted_salary = salary
            st.session_state.user_inputs = {
                'Position': job_title,
                'Experience': years_of_experience,
                'Location': location,
                'Education': education_level,
                'Company Size': company_size,
                'Skills': skills_list,
            }
            st.session_state.page = 'result'
    st.markdown('</div>', unsafe_allow_html=True)

# -------- RESULT PAGE (simple/placeholder) --------
elif st.session_state.page == 'result':
    st.button("← Back", on_click=lambda: st.session_state.update(page='form'), key="back_form", help="Back to input form", type="secondary")
    st.markdown(f"""
        <div class="center-card" style="text-align:center;">
            <div class="heading-main" style="margin-bottom:12px;">Your Predicted Salary</div>
            <div style="font-size:2em; color:#1abcfe; font-weight:800;">${st.session_state.predicted_salary:,.0f}</div>
            <div style="color:#7b8998;font-size:1em; margin:8px 0 0 0;">Estimated Annual Salary</div>
        </div>
    """, unsafe_allow_html=True)
