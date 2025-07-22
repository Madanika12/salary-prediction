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
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.stApp {
    background-color: #f5f7fa;
    color: #007fff; /* Light blue for default text */
}
.stButton>button {
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6em 1.5em;
    font-weight: bold;
    transition: 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(to right, #0072ff, #00c6ff);
    transform: scale(1.05);
}
.card {
    background: white;
    border-radius: 12px;
    padding: 25px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 25px;
    color: #007fff; /* Light blue inside cards */
}
.result-card {
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    padding: 30px;
    border-radius: 14px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 6px 14px rgba(0,0,0,0.15);
    animation: fadeIn 1s ease-in-out;
}
h1, h2, h4 {
    font-family: 'Segoe UI', sans-serif;
    color: #007fff !important;
}
@keyframes fadeIn {
  from {opacity: 0;}
  to {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# ---------------------- FORM PAGE ---------------------- #
if st.session_state.page == 'form':
    st.markdown("<div class='result-card'><h2>💼 Salary Prediction Tool</h2></div>", unsafe_allow_html=True)

    with st.form("salary_form"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Provide Your Details")

        job_title = st.selectbox("🧑‍💻 Job Title", label_encoders['job_title'].classes_)
        years_of_experience = st.number_input("⏳ Years of Experience", 0, 50, 2)
        location = st.selectbox("📍 Location", label_encoders['location'].classes_)
        education_level = st.selectbox("🎓 Education Level", label_encoders['education_level'].classes_)
        company_size = st.selectbox("🏢 Company Size", label_encoders['company_size'].classes_)
        skills_list = st.multiselect("🛠️ Select Your Skills", mlb.classes_)
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
            st.markdown(f"<p><b>{key}:</b> {value}</p>", unsafe_allow_html=True)
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
