import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# Load model and encoders
model = joblib.load('best_salary_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
mlb = joblib.load('skills_mlb.pkl')

# Initialize session state
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

# ------------------- CSS Styling -------------------
import streamlit as st

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #ffffff;
        }
        .form-container {
            background-color: #f8fbfd;
            border-radius: 10px;
            padding: 2rem;
            width: 70%;
            margin: auto;
            box-shadow: 0 0 20px rgba(0, 123, 255, 0.05);
        }
        .form-title {
            font-size: 24px;
            font-weight: 600;
            color: #00AEEF;
            margin-bottom: 0.3rem;
        }
        .form-subtitle {
            font-size: 14px;
            color: #888;
            margin-bottom: 1.5rem;
        }
        label {
            font-size: 14px !important;
            font-weight: 600 !important;
            color: #00AEEF !important;
            margin-bottom: 0.2rem !important;
        }
        .stButton>button {
            background-color: #82d8ff;
            color: white;
            font-weight: 600;
            width: 100%;
            border-radius: 8px;
            padding: 10px 0px;
            margin-top: 1.5rem;
        }
        .stButton>button:hover {
            background-color: #59c6f9;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:#00AEEF;'>Salary Predictor</h1>", unsafe_allow_html=True)

with st.form("salary_form"):
    st.markdown("<div class='form-container'>", unsafe_allow_html=True)
    st.markdown("<div class='form-title'>💼 Salary Prediction Form</div>", unsafe_allow_html=True)
    st.markdown("<div class='form-subtitle'>Please fill in your details below</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Job Title**", unsafe_allow_html=True)
        job_title = st.selectbox("", ["Select your job title", "Software Engineer", "Data Scientist", "Product Manager"])

    with col2:
        st.markdown("**Years of Experience**", unsafe_allow_html=True)
        years_of_experience = st.number_input("", 0, 50, 2)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Location**", unsafe_allow_html=True)
        location = st.selectbox("", ["Select your location", "Bangalore", "Hyderabad", "Pune"])

    with col4:
        st.markdown("**Education Level**", unsafe_allow_html=True)
        education_level = st.selectbox("", ["Select education level", "Bachelors", "Masters", "PhD"])

    st.markdown("**Company Size**", unsafe_allow_html=True)
    company_size = st.selectbox("", ["Select company size", "Startup", "Mid-size", "Enterprise"])

    st.markdown("**Skills & Technologies**", unsafe_allow_html=True)
    skills = st.multiselect("", ["JavaScript", "Python", "React", "Node.js", "AWS", "Docker",
                                 "Machine Learning", "SQL", "Data Analysis",
                                 "Project Management", "Leadership", "Sales",
                                 "Design", "Marketing"])

    submitted = st.form_submit_button("Get Salary Prediction")

    if submitted:
        st.success("Prediction logic goes here!")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- FORM PAGE ---------------------- #
if st.session_state.page == 'form':
    st.markdown("<div class='result-card'><h2>💼 Salary Prediction Tool</h2></div>", unsafe_allow_html=True)

    with st.form("salary_form"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Provide Your Details", unsafe_allow_html=True)
        st.markdown("<span style='color:#007fff'><b>🧑‍💻 Job Title</b></span>", unsafe_allow_html=True)
        job_title = st.selectbox("", label_encoders['job_title'].classes_)
        st.markdown("<span style='color:#007fff'><b>⏳ Years of Experience</b></span>", unsafe_allow_html=True)
        years_of_experience = st.number_input("", 0, 50, 2)
        st.markdown("<span style='color:#007fff'><b>📍 Location</b></span>", unsafe_allow_html=True)
        location = st.selectbox("", label_encoders['location'].classes_)
        st.markdown("<span style='color:#007fff'><b>🎓 Education Level</b></span>", unsafe_allow_html=True)
        education_level = st.selectbox("", label_encoders['education_level'].classes_)
        st.markdown("<span style='color:#007fff'><b>🏢 Company Size</b></span>", unsafe_allow_html=True)
        company_size = st.selectbox("", label_encoders['company_size'].classes_)
        st.markdown("<span style='color:#007fff'><b>🛠️ Select Your Skills</b></span>", unsafe_allow_html=True)
        skills_list = st.multiselect("", mlb.classes_)
        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("📊 Predict My Salary")
        if submitted:
            with st.spinner("🔍 Predicting your salary..."):
                salary = predict_salary(job_title, years_of_experience, location, education_level, company_size, skills_list)

            st.toast("✅ Prediction successful!", icon="💰")

            st.session_state.predicted_salary = salary
            st.session_state.user_inputs = {
                'Position': job_title,
                'Experience': years_of_experience,
                'Location': location,
                'Education': education_level,
                'Company Size': company_size,
                'Skills': ", ".join(skills_list) if skills_list else "None"
            }
            go_to_result()

# ---------------------- RESULT PAGE ---------------------- #
elif st.session_state.page == 'result':
    st.button("← Back to Form", on_click=go_back_to_form)

    st.markdown(f"""
        <div class="result-card">
            <h2>💰 Predicted Salary</h2>
            <h1>${st.session_state.predicted_salary:,.0f}</h1>
            <p>Estimated Annual Salary</p>
            <p><b>${st.session_state.predicted_salary * 0.85:,.0f}</b> (Low) — 
               <b>${st.session_state.predicted_salary * 1.15:,.0f}</b> (High)</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'><h4>👤 Your Profile</h4>", unsafe_allow_html=True)
        for key, value in st.session_state.user_inputs.items():
            st.markdown(f"<p style='color:#007fff;'><b>{key}:</b> {value}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='card'>
            <h4>📈 Market Insights</h4>
            <p><b>Industry Avg:</b> $55,200</p>
            <p><b>Top 10% Earners:</b> $87,000</p>
            <p><b>Growth Potential:</b> <span style='color:green; font-weight:bold;'>High</span></p>
            <p><b>Demand Level:</b> <span style='color:blue; font-weight:bold;'>Very High</span></p>
        </div>
        """, unsafe_allow_html=True)

    # -------- Plotly Bar Chart --------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card'><h4>📉 Salary Comparison Chart</h4></div>", unsafe_allow_html=True)

    comparison_chart = go.Figure()

    comparison_chart.add_trace(go.Bar(
        x=['Your Prediction'],
        y=[st.session_state.predicted_salary],
        name='Predicted',
        marker_color='indigo'
    ))
    comparison_chart.add_trace(go.Bar(
        x=['Industry Average'],
        y=[55200],
        name='Industry Avg',
        marker_color='gray'
    ))
    comparison_chart.add_trace(go.Bar(
        x=['Top 10% Earners'],
        y=[87000],
        name='Top 10%',
        marker_color='green'
    ))

    comparison_chart.update_layout(
        barmode='group',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=30, r=30, t=30, b=30),
        yaxis_title="Salary ($)",
        font=dict(size=14)
    )

    st.plotly_chart(comparison_chart, use_container_width=True)
