# ... (your other imports and code above remain unchanged)
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
.header-main {
    max-width: 600px;
    margin: 42px auto 30px auto;
    padding: 0;
    border-radius: 22px;
    text-align: center;
}
.header-box {
    background: linear-gradient(90deg, #005bea 0%, #00c6fb 100%);
    border-radius: 22px;
    padding: 32px 0 25px 0;
    box-shadow: 0 8px 40px 0 rgba(0,40,120,0.13);
    margin-bottom: 0;
}
.header-title {
    color: #fff;
    font-size: 2.2em;
    font-weight: 800;
    letter-spacing: 0.02em;
    margin-bottom: 12px;
    margin-top: 0;
    text-shadow: 0 2px 18px rgba(0,170,255,0.12);
}
.header-desc {
    color: #e6f4ff;
    font-size: 1.08em;
    font-weight: 400;
    margin-top: 0;
    margin-bottom: 0;
    letter-spacing: 0.003em;
}
@media (max-width: 700px) {
    .header-main { max-width: 96vw; }
    .header-box { padding: 16px 0 12px 0; }
    .header-title { font-size: 1.25em; }
    .header-desc { font-size: 0.99em; }
}
</style>
""", unsafe_allow_html=True)

# ----------- Sidebar (remains unchanged)
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
    # New header: professional, in a colored box, no empty white box.
    st.markdown("""
    <div class="header-main">
      <div class="header-box">
        <div class="header-title">Salary Prediction Tool</div>
        <div class="header-desc">
            Get your personalized salary estimate instantly.<br>
            Enter your details below to see your market value!
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("salary_form"):
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            job_title = st.selectbox("🏢Job Title", label_encoders['job_title'].classes_)
            years_of_experience = st.number_input("⏳Years of Experience", 0, 50, 2, help="Enter total years of professional experience.")
            location = st.selectbox("📍Location", label_encoders['location'].classes_)
            education_level = st.selectbox("🎓Education Level", label_encoders['education_level'].classes_)
            company_size = st.selectbox("🏢Company Size", label_encoders['company_size'].classes_)
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

# ---------------------- RESULT PAGE (unchanged) ---------------------- #
elif st.session_state.page == 'result':
    st.button("← Back to Form", on_click=go_back_to_form)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="profile-card">
            <div class="card-head"><span class="icon">🎯</span>Your Profile</div>
            <div class='profile-details'>
                <b>Position:</b> {position}<br>
                <b>Experience:</b> {experience}<br>
                <b>Location:</b> {location}<br>
                <b>Education:</b> {education}
            </div>
        </div>
        """.format(
            position=st.session_state.user_inputs.get('Position', ''),
            experience=st.session_state.user_inputs.get('Experience', ''),
            location=st.session_state.user_inputs.get('Location', ''),
            education=st.session_state.user_inputs.get('Education', '')
        ), unsafe_allow_html=True)

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
