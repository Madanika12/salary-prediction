import streamlit as st
import joblib
import pandas as pd

# --- Load model and encoders (your code here) ---
model = joblib.load('best_salary_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
mlb = joblib.load('skills_mlb.pkl')

if 'page' not in st.session_state:
    st.session_state.page = 'form'
if 'predicted_salary' not in st.session_state:
    st.session_state.predicted_salary = None
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}

def go_to_result():
    st.session_state.page = 'result'

def go_back_to_form():
    st.session_state.page = 'form'

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

# --- Custom CSS for look & feel inspired by your references ---
st.markdown("""
<style>
body { background: #f9fbfd; }
header, footer { visibility: hidden; }
.stApp { background: #f9fbfd; }
h1, h2, h3, h4, h5, h6 { color: #13b7ff; font-weight: 700; }
.stButton>button, .stDownloadButton>button {
    background: #38c6ff !important;
    color: white !important;
    border-radius: 8px !important;
    font-size: 1.1em !important;
    border: none !important;
    padding: 0.75em 0 !important;
    margin-top: 18px;
    transition: 0.2s;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: #13b7ff !important;
}
.input-card, .result-card {
    background: #fff;
    border-radius: 14px;
    box-shadow: 0 4px 20px rgba(22, 170, 255, 0.09);
    padding: 0px 0 30px 0;
    margin: 0 auto 30px auto;
    max-width: 620px;
}
.input-card { padding: 0px 0 30px 0 !important; }
.form-section-title {
    background: #f4faff;
    color: #13b7ff;
    font-weight: 700;
    font-size: 1.3em;
    border-radius: 14px 14px 0 0;
    padding: 22px 36px 12px 36px;
    margin-bottom: 8px;
    letter-spacing: 0.02em;
}
.form-section-subtitle {
    color: #9bbbd4;
    font-size: 1em;
    margin-bottom: 18px;
    margin-left: 36px;
}
.form-fields {
    padding: 0 36px;
}
.stSelectbox>div, .stMultiSelect>div, .stNumberInput>div {
    margin-bottom: 18px;
}
.skill-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 18px;
    margin-bottom: 8px;
}
.skill-badge {
    background: #f4faff;
    color: #13b7ff;
    padding: 8px 22px;
    border-radius: 18px;
    font-size: 1em;
    font-weight: 600;
    margin-bottom: 2px;
    margin-right: 0px;
    border: none;
    cursor: pointer;
    transition: 0.18s;
    outline: none;
}
.skill-badge.selected, .skill-badge:hover {
    background: #13b7ff;
    color: #fff;
}
.result-header {
    background: #13b7ff;
    color: white;
    font-size: 1.22em;
    padding: 16px 36px;
    border-radius: 14px 14px 0 0;
    font-weight: 600;
    margin-bottom: 0;
    display: flex; gap: 10px; align-items: center;
}
.result-main {
    padding: 24px 36px 10px 36px;
    text-align: center;
}
.result-main .salary {
    color: #13b7ff;
    font-size: 2.9em;
    font-weight: 700;
    margin-bottom: 0.1em;
}
.result-range {
    display: flex; justify-content: center; gap: 42px;
    margin-bottom: 8px;
    margin-top: 0.9em;
}
.result-range .low, .result-range .high {
    font-size: 1.13em;
    color: #333;
}
.result-range .low-label, .result-range .high-label {
    color: #9bbbd4;
    font-size: 0.97em;
}
.profile-market-cols {
    display: flex;
    gap: 20px;
    justify-content: center;
    margin-top: 30px;
    margin-bottom: 18px;
}
.profile-box, .market-box {
    background: #fff;
    border: 1.5px solid #f0f6fa;
    border-radius: 12px;
    padding: 22px 27px 16px 27px;
    min-width: 220px;
    flex: 1 1 0;
}
.profile-box h4, .market-box h4 {
    color: #13b7ff;
    font-size: 1.14em;
    font-weight: 700;
    margin-bottom: 12px;
}
.profile-rows, .market-rows {
    font-size: 1.04em;
    color: #333;
    margin-bottom: 7px;
}
.icon { font-size: 1.1em; margin-right: 7px; }
.inline-badge {
    background: #e6fbe8;
    color: #23b055;
    padding: 2.5px 13px;
    border-radius: 10px;
    font-size: 0.96em;
    margin-left: 7px;
    font-weight: 600;
}
.inline-badge.blue {
    background: #e6f2ff;
    color: #287cf7;
}
@media (max-width: 700px) {
    .input-card, .result-card { padding: 0 0 10px 0 !important; }
    .form-section-title, .result-header, .form-fields, .result-main { padding: 13px 12px !important; }
    .profile-market-cols { flex-direction: column; gap: 10px; }
    .profile-box, .market-box { min-width: 0; padding: 17px 7px 7px 7px; }
}
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown("<h1 style='margin-bottom: 34px;'>Salary Predictor</h1>", unsafe_allow_html=True)

# ---- FORM PAGE ----
if st.session_state.page == 'form':
    st.markdown("""
    <div class="input-card">
        <div class="form-section-title">💼 Salary Prediction Form</div>
        <div class="form-section-subtitle">
            Please fill in your details below
        </div>
        <div class="form-fields">
    """, unsafe_allow_html=True)
    with st.form("salary_form"):
        cols = st.columns(2)
        job_title = cols[0].selectbox("Job Title", label_encoders['job_title'].classes_, key="job_title")
        years_of_experience = cols[1].number_input("Years of Experience", 0, 50, 2, key="years_exp", placeholder="e.g. 5")
        cols2 = st.columns(2)
        location = cols2[0].selectbox("Location", label_encoders['location'].classes_, key="loc")
        education_level = cols2[1].selectbox("Education Level", label_encoders['education_level'].classes_, key="edu")
        company_size = st.selectbox("Company Size", label_encoders['company_size'].classes_, key="comp_size")

        # --- Skills & Technologies as badges ---
        st.markdown('<div class="form-section-title" style="font-size:1.18em;margin-bottom:7px;margin-top:20px;"><span class="icon">&lt;/&gt;</span>Skills & Technologies</div>', unsafe_allow_html=True)
        skills_selected = st.multiselect("", mlb.classes_, key="skills")
        # Show badges (visual only)
        st.markdown('<div class="skill-badges">' + ''.join(
            [f'<span class="skill-badge{" selected" if skill in skills_selected else ""}">{skill}</span>' for skill in mlb.classes_]
        ) + '</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Get Salary Prediction")
        if submitted:
            salary = predict_salary(job_title, years_of_experience, location, education_level, company_size, skills_selected)
            st.session_state.predicted_salary = salary
            st.session_state.user_inputs = {
                'Position': job_title,
                'Experience': years_of_experience,
                'Location': location,
                'Education': education_level,
                'Skills': skills_selected
            }
            go_to_result()
    st.markdown("</div>", unsafe_allow_html=True)

# ---- RESULT PAGE ----
elif st.session_state.page == 'result':
    st.markdown("""
        <a href="#" onclick="window.location.reload(); return false;" style="text-decoration:none;">
            <button style="margin-bottom:18px;padding:7px 18px;border-radius:7px;background:#fff;border:1px solid #e6f2ff;cursor:pointer;font-size:1em;">
                ← Back to Form
            </button>
        </a>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="result-card">
        <div class="result-header">
            <span style="font-size:1.22em;">🪙</span>
            Salary Prediction Result
        </div>
        <div class="result-main">
            <div class="salary">${:,.0f}</div>
            <div style="color:#9bbbd4;font-size:1.14em;margin-bottom:7px;">Annual Salary Estimate</div>
            <div class="result-range">
                <div>
                    <div class="low">${:,.0f}</div>
                    <div class="low-label">Low Range</div>
                </div>
                <div>
                    <div class="high">${:,.0f}</div>
                    <div class="high-label">High Range</div>
                </div>
            </div>
        </div>
        <div class="profile-market-cols">
            <div class="profile-box">
                <h4><span class="icon">👤</span>Your Profile</h4>
                <div class="profile-rows"><span class="icon">💼</span>Position <span style="float:right;font-weight:600;">{}</span></div>
                <div class="profile-rows"><span class="icon">⏱️</span>Experience <span style="float:right;font-weight:600;">{} years</span></div>
                <div class="profile-rows"><span class="icon">📍</span>Location <span style="float:right;font-weight:600;">{}</span></div>
                <div class="profile-rows"><span class="icon">🎓</span>Education <span style="float:right;font-weight:600;">{}</span></div>
                <div class="profile-rows" style="margin-top:10px;">
                    <div style="font-size:0.96em;color:#9bbbd4;margin-bottom:3px;">Skills</div>
                    <div>""" + ''.join([f'<span class="skill-badge selected" style="margin:0 6px 6px 0;">{s}</span>' for s in st.session_state.user_inputs.get('Skills', [])]) + """</div>
                </div>
            </div>
            <div class="market-box">
                <h4><span class="icon">📈</span>Market Insights</h4>
                <div class="market-rows"><b>Industry Average:</b> <span style="float:right;">$97,888</span></div>
                <div class="market-rows"><b>Top 10% Earners:</b> <span style="float:right;">$154,280</span></div>
                <div class="market-rows"><b>Growth Potential:</b> <span class="inline-badge">High</span></div>
                <div class="market-rows"><b>Demand Level:</b> <span class="inline-badge blue">Very High</span></div>
            </div>
        </div>
        <form action="" method="post">
            <button onclick="window.location.reload(); return false;" type="button" style="background:#38c6ff;color:white;padding:13px 0;font-size:1.1em;border:none;border-radius:8px;width:250px;margin:30px auto 0 auto;display:block;cursor:pointer;">
                Try Another Prediction
            </button>
        </form>
    </div>
    """.format(
        st.session_state.predicted_salary,
        st.session_state.predicted_salary * 0.85,
        st.session_state.predicted_salary * 1.15,
        st.session_state.user_inputs.get('Position', ''),
        st.session_state.user_inputs.get('Experience', ''),
        st.session_state.user_inputs.get('Location', ''),
        st.session_state.user_inputs.get('Education', ''),
    ), unsafe_allow_html=True)
