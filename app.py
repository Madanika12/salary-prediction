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
# --- Custom CSS for Dark Large Header Box ---
st.markdown("""
<style>
body { background: #11141a; }
.header-main-box {
    background: linear-gradient(90deg, #002642 0%, #005bea 100%);
    border-radius: 22px;
    padding: 40px 0 40px 0;
    margin: 0 0 38px 0;
    width: 100%;
    text-align: center;
    box-shadow: 0 8px 40px 0 rgba(0,40,120,0.11);
}
.header-title {
    color: #1abcfe;
    font-size: 3em;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 0;
    margin-top: 0;
    text-shadow: 0 2px 18px rgba(0,170,255,0.13);
}
@media (max-width: 650px) {
    .header-main-box { padding: 22px 0; }
    .header-title { font-size: 2em; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-main-box">
    <div class="header-title">
        Salary Predictor
    </div>
</div>
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
