import streamlit as st
import joblib
import pandas as pd

# ----------- Load model and encoders -----------
model = joblib.load('best_salary_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
mlb = joblib.load('skills_mlb.pkl')

# ----------- Session State Setup -----------
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

# ----------- Custom CSS -----------
st.markdown("""
<style>
body { background: #101217; }
.stButton>button {
    background: linear-gradient(90deg,#005bea,#00c6fb);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.65em 1.7em;
    font-weight: bold;
    font-size: 1.06em;
    margin-top: 0.8em;
    transition: 0.3s;
    box-shadow: 0 2px 8px rgba(0,40,120,0.05);
}
.stButton>button:hover {
    background: linear-gradient(90deg,#00c6fb,#005bea);
    box-shadow: 0 4px 16px rgba(0,55,255,0.13);
}
.card, .profile-card, .insight-card {
    background: #fff;
    border-radius: 20px;
    padding: 24px 24px 18px 24px;
    box-shadow: 0 4px 16px rgba(0, 80, 255, 0.07);
    margin-bottom: 18px;
}
.result-card {
    background: linear-gradient(90deg,#005bea 60%,#00c6fb 100%);
    color: white;
    padding: 34px 20px 26px 20px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 6px 20px rgba(0, 80, 255, 0.15);
}
input, select, textarea {
    border-radius: 8px !important;
}
.profile-details {
    margin-top: 20px;
    color: #fff;
    font-size: 1.08em;
}
.profile-details b {
    color: #fff;
}
.insight-list {
    margin-top: 18px;
    color: #222;
    font-size: 1.07em;
}
.insight-list .insight-label {
    color: #7a7a7a;
    font-weight: 500;
}
.insight-list .insight-value {
    color: #111;
    font-weight: 600;
}
.insight-list .high { color: #1aa260; font-weight: bold;}
.insight-list .veryhigh { color: #1a56e0; font-weight: bold;}
/* Icon and heading in cards */
.card-head {
    display: flex;
    align-items: center;
    font-size: 1.08em;
    font-weight: 700;
    color: #1abcfe;
    margin-bottom: 10px;
}
.card-head .icon {
    font-size: 1.25em;
    margin-right: 10px;
}
@media (max-width: 900px) {
    .card, .profile-card, .insight-card { padding: 14px 8px 10px 8px; }
}
</style>
""", unsafe_allow_html=True)

# ----------- Sidebar -----------
with st.sidebar:
    st.image("https://img.icons8.com/external-flatart-icons-outline-flatarticons/64/000000/external-salary-global-business-flatart-icons-outline-flatarticons.png", width=50)
    st.title("Salary Predictor")
    st.caption("Estimate your market value instantly. Enter your profile, skills and get an AI-powered salary prediction.")
    st.markdown("---")
    st.info("**Pro tip:** Select your actual skills for the best results.")
    st.markdown(
        "<div style='font-size: 0.95em; color: #888;'>Made with ❤️ using Streamlit.</div>",
        unsafe_allow_html=True
    )

# ---------------------- FORM PAGE ---------------------- #
if st.session_state.page == 'form':
    st.markdown("<div class='result-card'><h2>💼 Salary Prediction Tool</h2><p style='margin-top: -12px;'>Get your personalized salary estimate</p></div>", unsafe_allow_html=True)
    with st.form("salary_form"):
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            job_title = st.selectbox("Job Title", label_encoders['job_title'].classes_)
            years_of_experience = st.number_input("Years of Experience", 0, 50, 2, help="Enter total years of professional experience.")
            location = st.selectbox("Location", label_encoders['location'].classes_)
            education_level = st.selectbox("Education Level", label_encoders['education_level'].classes_)
            company_size = st.selectbox("Company Size", label_encoders['company_size'].classes_)
            skills_list = st.multiselect(
                "Select Your Skills",
                mlb.classes_,
                help="Hold Ctrl (Windows) or Cmd (Mac) to select multiple."
            )
            st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔎 Predict My Salary")
        if submitted:
            salary = predict_salary(job_title, years_of_experience, location, education_level, company_size, skills_list)
            st.session_state.predicted_salary = salary
            st.session_state.user_inputs = {
                'Position': job_title,
                'Experience': years_of_experience,
                'Location': location,
                'Education': education_level
            }
            go_to_result()

# ---------------------- RESULT PAGE ---------------------- #
elif st.session_state.page == 'result':
    st.button("← Back to Form", on_click=go_back_to_form)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="profile-card">
            <div class="card-head"><span class="icon">🎯</span>Your Profile</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            f"""<div class='profile-details'>
<b>Position:</b> {st.session_state.user_inputs.get('Position', '')}<br>
<b>Experience:</b> {st.session_state.user_inputs.get('Experience', '')}<br>
<b>Location:</b> {st.session_state.user_inputs.get('Location', '')}<br>
<b>Education:</b> {st.session_state.user_inputs.get('Education', '')}
</div>
""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-card">
            <div class="card-head"><span class="icon">📊</span>Market Insights</div>
            <div class="insight-list">
                <div><span class="insight-label">Industry Average:</span> <span class="insight-value">$55,200</span></div>
                <div><span class="insight-label">Top 10% Earners:</span> <span class="insight-value">$87,000</span></div>
                <div><span class="insight-label">Growth Potential:</span> <span class="insight-value high">High</span></div>
                <div><span class="insight-label">Demand Level:</span> <span class="insight-value veryhigh">Very High</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="result-card">
            <h2>💲 Salary Prediction</h2>
            <h1 style="margin-bottom: 0.2em;">${st.session_state.predicted_salary:,.0f}</h1>
            <p style="font-size:1.15em;margin-top:0.2em;">Estimated Annual Salary</p>
            <p><b>${st.session_state.predicted_salary * 0.85:,.0f}</b> <span style="color:#d4ffea;">Low</span> — 
               <b>${st.session_state.predicted_salary * 1.15:,.0f}</b> <span style="color:#ffe4c2;">High</span></p>
        </div>
    """, unsafe_allow_html=True)
