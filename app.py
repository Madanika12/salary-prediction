import streamlit as st
import joblib
import pandas as pd

# --- Model/Encoders Loading ---
model = joblib.load('best_salary_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
mlb = joblib.load('skills_mlb.pkl')

# --- Session State ---
if 'page' not in st.session_state:
    st.session_state.page = 'form'
if 'predicted_salary' not in st.session_state:
    st.session_state.predicted_salary = None
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}
if 'selected_skills' not in st.session_state:
    st.session_state.selected_skills = []

# --- Salary Prediction Logic ---
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

# --- Custom CSS for Blue Modern UI ---
st.markdown("""
<style>
body { background: #f8fbff; }
h1, h2, h3, h4, h5, h6 { color: #1abcfe !important; }
.blue-card {
    background: #fff;
    border-radius: 16px;
    box-shadow: 0 3px 16px #e7f3fd;
    padding: 34px 32px 24px 32px;
    margin: 30px auto;
    width: 100%;
    max-width: 540px;
}
.form-label {
    font-weight: 600;
    color: #1abcfe;
    margin-bottom: 0.2em;
}
.form-section {
    margin-bottom: 18px;
}
.stButton>button {
    background: #1abcfe;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1.1em;
    padding: 0.7em 0.5em;
    margin-top: 0.5em;
    margin-bottom: 0.3em;
    transition: 0.2s;
}
.stButton>button:hover {
    background: #0fa4da;
}
.skill-badge {
    display: inline-block;
    background: #f2f8fd;
    color: #1abcfe;
    border-radius: 17px;
    padding: 5px 18px;
    margin: 3px 8px 8px 0;
    font-weight: 500;
    font-size: 1em;
    border: 1px solid #e2f0fb;
}
.header-card {
    background: #e8f6fe;
    border-radius: 14px 14px 0 0;
    padding: 18px 32px;
    color: #1abcfe;
    font-weight: 700;
    font-size: 1.25em;
    display: flex;
    align-items: center;
    margin-bottom: 0;
}
.result-main {
    color: #1abcfe;
    font-size: 2.4em;
    font-weight: 800;
    margin: 0.2em 0 0.1em 0;
}
.range-label {
    color: #222;
    font-size: 1em;
    margin-bottom: 0.3em;
}
.range-value {
    color: #1abcfe;
    font-weight: 600;
}
.profile-card, .insight-card {
    background: #fcfdfe;
    border-radius: 12px;
    border: 1px solid #e7f3fd;
    padding: 18px 18px 10px 18px;
    margin: 15px 6px 10px 0;
    font-size: 1em;
}
.back-btn {
    background: #fff !important;
    color: #1abcfe !important;
    border: 1.5px solid #d3eafd !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    margin-bottom: 1.5em !important;
}
@media (max-width: 650px) {
    .blue-card { padding: 16px 6px 16px 6px; }
    .header-card { padding: 10px 12px; font-size: 1.1em; }
}
</style>
""", unsafe_allow_html=True)

# --- Main Title (top left) ---
st.markdown("<h2 style='color:#1abcfe;font-weight:800;margin-bottom:12px;'>Salary Predictor</h2>", unsafe_allow_html=True)

# --- FORM PAGE ---
if st.session_state.page == 'form':
    st.markdown("""
    <div class="blue-card">
        <div class="header-card">
            <span style="font-size:1.4em;margin-right:10px;">🧾</span>
            Salary Prediction Form
        </div>
        <div style="color:#8a99ad;font-size:1.03em;margin-bottom:13px;">
            Please fill in your details below
        </div>
    """, unsafe_allow_html=True)
    with st.form("salary_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="form-label">🏢 Job Title</div>', unsafe_allow_html=True)
            job_title = st.selectbox("", label_encoders['job_title'].classes_, key='job_title')
        with col2:
            st.markdown('<div class="form-label">⌛ Years of Experience</div>', unsafe_allow_html=True)
            years_of_experience = st.number_input("", 0, 50, 2, key='yoe', placeholder="e.g. 5")
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
        # --- Custom Grid Style for Skills ---
        skills_list = st.multiselect(
            "",
            options=mlb.classes_,
            default=st.session_state.selected_skills,
            key="skills"
        )
        st.session_state.selected_skills = skills_list
        st.markdown(
            "".join([f'<span class="skill-badge">{skill}</span>' for skill in skills_list]),
            unsafe_allow_html=True
        )
        submitted = st.form_submit_button("Get Salary Prediction")
        if submitted:
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
    st.markdown("</div>", unsafe_allow_html=True)

# --- RESULT PAGE ---
elif st.session_state.page == 'result':
    st.button("← Back to Form", on_click=lambda: st.session_state.update(page='form'), key="back_form", help="Back to input form", type="secondary")
    st.markdown("""
    <div class="blue-card" style="max-width:600px;margin-top:0;">
        <div class="header-card" style="background:#1abcfe;color:#fff;">
            <span style="font-size:1.4em;margin-right:10px;">💰</span>
            Salary Prediction Result
        </div>
    """, unsafe_allow_html=True)
    st.markdown(
        f"""<div style="text-align:center;margin:18px 0 12px 0;">
            <div class="result-main">${st.session_state.predicted_salary:,.0f}</div>
            <div style="color:#8a99ad;font-size:1.12em;margin-bottom:0.4em;">Annual Salary Estimate</div>
            <span class="range-value">${st.session_state.predicted_salary*0.85:,.0f}</span>
            <span class="range-label">Low Range</span> &nbsp;&nbsp;
            <span class="range-value">${st.session_state.predicted_salary*1.15:,.0f}</span>
            <span class="range-label">High Range</span>
        </div>""",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        st.markdown('<b>👤 Your Profile</b><br>', unsafe_allow_html=True)
        ui = st.session_state.user_inputs
        st.markdown(f"""<ul style="margin-bottom:0.7em;">
            <li><b>🏢 Position:</b> {ui['Position']}</li>
            <li><b>⌛ Experience:</b> {ui['Experience']} years</li>
            <li><b>📍 Location:</b> {ui['Location']}</li>
            <li><b>🎓 Education:</b> {ui['Education']}</li>
            <li><b>🏢 Company Size:</b> {ui['Company Size']}</li>
        </ul>""", unsafe_allow_html=True)
        if ui['Skills']:
            st.markdown('<b>Skills</b><br>', unsafe_allow_html=True)
            st.markdown(
                "".join([f'<span class="skill-badge">{skill}</span>' for skill in ui['Skills']]),
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.markdown('<b>📈 Market Insights</b><br>', unsafe_allow_html=True)
        st.markdown("""
            <ul style="margin-bottom:0.7em;">
                <li><b>Industry Average:</b> $55,200</li>
                <li><b>Top 10% Earners:</b> $87,000</li>
                <li><b>Growth Potential:</b> <span style='background:#d5f6e3;color:#18aa4b;border-radius:8px;padding:2px 10px;'>High</span></li>
                <li><b>Demand Level:</b> <span style='background:#e1eafd;color:#3a87f2;border-radius:8px;padding:2px 10px;'>Very High</span></li>
            </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;margin-top:20px;"><button class="stButton" style="width:280px;max-width:90%;" onclick="window.location.reload()">Try Another Prediction</button></div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
