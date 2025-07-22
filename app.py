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
body, .stApp {
    background: linear-gradient(135deg, #f7fafd 0%, #e9f1f7 100%) !important;
}
.center-card {
    background: #fff;
    border-radius: 14px;
    box-shadow: 0 6px 24px 0 rgba(20,40,120,0.10);
    padding: 28px 28px 18px 28px;
    margin: 40px auto 0 auto;
    width: 100%;
    max-width: 520px;
}
.heading-main {
    color: #1abcfe;
    text-align: center;
    font-size: 1.38em;
    font-weight: 800;
    margin-bottom: 2px;
    letter-spacing: 0.02em;
}
.heading-sub {
    color: #7b8998;
    text-align: center;
    margin-bottom: 22px;
    margin-top: 0;
    font-size: 0.98em;
}
.form-row {
    display: flex;
    gap: 12px;
    margin-bottom: 13px;
}
.form-col {
    flex: 1;
}
.form-label {
    font-weight: 600;
    color: #1abcfe;
    font-size: 0.98em;
    display: flex;
    align-items: center;
    margin-bottom: 4px;
    gap: 6px;
}
.form-icon {
    font-size: 1.05em;
    margin-right: 4px;
}
.stSelectbox, .stNumberInput, .stMultiSelect, .stTextInput input {
    background: #fff !important;
    color: #222 !important;
    border-radius: 7px !important;
    border: 1.2px solid #e3e9f0 !important;
    font-size: 0.98em !important;
    min-height: 38px;
}
.stNumberInput input { color: #222 !important; }
.stMultiSelect>div>div {
    background: #fff !important;
    color: #222 !important;
}
hr {
    border: none;
    border-top: 1.3px solid #e3e9f0;
    margin: 18px 0 14px 0;
}
.skills-label {
    font-weight: 700;
    color: #1abcfe;
    margin-bottom: 5px;
    margin-top: 5px;
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.96em;
}
.skills-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 9px 13px;
    margin-bottom: 8px;
}
.skill-pill {
    background: #eaf4fa;
    color: #19a8e8;
    border-radius: 14px;
    padding: 5px 15px;
    font-size: 0.96em;
    font-weight: 600;
    border: none;
    pointer-events: none;
}
.predict-btn button {
    width: 100%;
    background: linear-gradient(90deg,#1abcfe 60%,#15e0ff 100%);
    color: #fff;
    font-weight: 700;
    font-size: 1em;
    border: none;
    border-radius: 7px;
    padding: 0.7em 0;
    margin-top: 12px;
    transition: 0.16s;
    box-shadow: 0 3px 12px rgba(0,180,255,0.09);
}
.predict-btn button:hover {
    background: linear-gradient(90deg,#15e0ff 60%,#1abcfe 100%);
    color: #fff;
}
@media (max-width: 700px) {
    .center-card { padding: 12px 2vw 12px 2vw; }
    .form-row { flex-direction: column; gap: 6px;}
}
</style>
""", unsafe_allow_html=True)

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
