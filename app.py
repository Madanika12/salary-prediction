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

# --- Custom CSS for header and dark form card (designs from your references) ---
st.markdown("""
<style>
/* Whole background white */
body, .stApp { background: #fff !important; }

/* Large blue gradient header box */
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

/* Form main card - dark */
.form-dark-card {
    background: #181b20;
    border: 1.2px solid #292c34;
    border-radius: 12px;
    padding: 34px 24px 28px 24px;
    margin: 18px auto 0 auto;
    width: 100%;
    max-width: 670px;
    box-shadow: 0 4px 24px rgba(20,20,20,0.08);
}

/* Label row flex */
.form-label-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 36px;
    margin-bottom: 7px;
    margin-top: 8px;
}
.form-label {
    color: #fff;
    font-size: 1.06em;
    font-weight: 600;
    margin-bottom: 2px;
    display: flex;
    align-items: center;
    gap: 7px;
}
.form-icon {
    font-size: 1.2em;
    margin-right: 5px;
    vertical-align: middle;
}
.form-input-row {
    display: flex;
    gap: 36px;
    margin-bottom: 17px;
}
.form-input-col {
    flex: 1;
}

/* Streamlit input tweaks for dark card */
.form-dark-card .stSelectbox, 
.form-dark-card .stNumberInput, 
.form-dark-card .stMultiSelect {
    background: #23242a !important;
    color: #fff !important;
    border-radius: 8px !important;
    border: none !important;
}
.form-dark-card .stSelectbox label,
.form-dark-card .stNumberInput label,
.form-dark-card .stMultiSelect label {
    color: #fff !important;
}
.form-dark-card .stTextInput input, 
.form-dark-card .stNumberInput input {
    background: #23242a !important;
    color: #fff !important;
    border-radius: 8px !important;
}
.form-dark-card .stMultiSelect>div>div {
    background: #23242a !important;
    color: #fff !important;
}
.stButton>button {
    background: none !important;
    color: #fff !important;
    border: 1.5px solid #888;
    border-radius: 7px;
    padding: 0.6em 0.9em;
    font-size: 1.07em;
    margin-top: 0.7em;
    transition: 0.2s;
}
.stButton>button:hover {
    border: 1.5px solid #1abcfe;
    color: #1abcfe !important;
}

/* Result card - blue/white */
.result-card {
    background: linear-gradient(90deg,#005bea 60%,#00c6fb 100%);
    color: white;
    padding: 34px 20px 26px 20px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 6px 20px rgba(0, 80, 255, 0.15);
}
.profile-card, .insight-card {
    background: #fcfdfe;
    border-radius: 12px;
    border: 1px solid #e7f3fd;
    padding: 18px 18px 10px 18px;
    margin: 15px 6px 10px 0;
    font-size: 1em;
    color: #222;
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
.range-label {
    color: #eee;
    font-size: 1em;
    margin-bottom: 0.3em;
}
.range-value {
    color: #fff;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --- Main Header ---
st.markdown("""
<div class="header-main-box">
    <div class="header-title">
        Salary Predictor
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------- FORM PAGE ---------------------- #
if st.session_state.page == 'form':
    st.markdown('<div class="form-dark-card">', unsafe_allow_html=True)
    with st.form("salary_form"):
        # --- Job Title & Years of Experience ---
        st.markdown(
            """
            <div class="form-label-row">
                <div class="form-label"><span class="form-icon">🏢</span>Job Title</div>
                <div class="form-label"><span class="form-icon">⏳</span>Years of Experience</div>
            </div>
            """, unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            job_title = st.selectbox("", label_encoders['job_title'].classes_, key='job_title')
        with col2:
            years_of_experience = st.number_input("", 0, 50, 2, key='yoe')

        # --- Location & Education Level ---
        st.markdown(
            """
            <div class="form-label-row">
                <div class="form-label"><span class="form-icon">📍</span>Location</div>
                <div class="form-label"><span class="form-icon">🎓</span>Education Level</div>
            </div>
            """, unsafe_allow_html=True,
        )
        col3, col4 = st.columns(2)
        with col3:
            location = st.selectbox("", label_encoders['location'].classes_, key='location')
        with col4:
            education_level = st.selectbox("", label_encoders['education_level'].classes_, key='edu')

        # --- Company Size ---
        st.markdown('<div class="form-label" style="margin-top:10px;"><span class="form-icon">🏢</span>Company Size</div>', unsafe_allow_html=True)
        company_size = st.selectbox("", label_encoders['company_size'].classes_, key='company')

        # --- Skills ---
        st.markdown('<div class="form-label" style="margin-top:10px;"><span class="form-icon">&lt;/&gt;</span>Skills & Technologies</div>', unsafe_allow_html=True)
        skills_list = st.multiselect("", options=mlb.classes_, key="skills")

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
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- RESULT PAGE ---------------------- #
elif st.session_state.page == 'result':
    st.button("← Back to Form", on_click=lambda: st.session_state.update(page='form'), key="back_form", help="Back to input form", type="secondary")
    st.markdown(f"""
        <div class="result-card">
            <h2>💲 Salary Prediction</h2>
            <h1 style="margin-bottom: 0.2em;">${st.session_state.predicted_salary:,.0f}</h1>
            <p style="font-size:1.15em;margin-top:0.2em;">Estimated Annual Salary</p>
            <span class="range-value">${st.session_state.predicted_salary*0.85:,.0f}</span>
            <span class="range-label">Low Range</span> &nbsp;&nbsp;
            <span class="range-value">${st.session_state.predicted_salary*1.15:,.0f}</span>
            <span class="range-label">High Range</span>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="profile-card">', unsafe_allow_html=True)
        st.markdown('<b>👤 Your Profile</b><br>', unsafe_allow_html=True)
        ui = st.session_state.user_inputs
        st.markdown(f"""<ul style="margin-bottom:0.7em;">
            <li><b>🏢 Position:</b> {ui['Position']}</li>
            <li><b>⏳ Experience:</b> {ui['Experience']} years</li>
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
