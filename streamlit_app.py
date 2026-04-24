import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="AI Job Impact Dashboard", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 20% 10%, #ffeecf 0%, #f8fafc 35%, #e8f0ff 100%);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.header-card {
    background: rgba(255, 255, 255, 0.85);
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}

.header-card h1 {
    margin: 0;
    color: #0f172a;
}

.header-card p {
    margin: 0.4rem 0 0 0;
    color: #334155;
}

section.main [data-testid="stWidgetLabel"],
section[data-testid="stMain"] [data-testid="stWidgetLabel"] {
    opacity: 1 !important;
}

section.main [data-testid="stWidgetLabel"] p,
section.main [data-testid="stWidgetLabel"] label,
section.main [data-testid="stWidgetLabel"] span,
section[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
section[data-testid="stMain"] [data-testid="stWidgetLabel"] label,
section[data-testid="stMain"] [data-testid="stWidgetLabel"] span,
section[data-testid="stMain"] .stNumberInput label,
section[data-testid="stMain"] .stSelectbox label,
section[data-testid="stMain"] .stTextInput label {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    font-weight: 600 !important;
    opacity: 1 !important;
    text-shadow: none !important;
}

section.main [data-testid="stMarkdownContainer"] p,
section.main [data-testid="stMarkdownContainer"] h1,
section.main [data-testid="stMarkdownContainer"] h2,
section.main [data-testid="stMarkdownContainer"] h3 {
    color: #0f172a;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="header-card">
    <h1>AI Job Impact Predictor</h1>
    <p>Interactive frontend for your FastAPI model endpoint.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

api_base_url = st.sidebar.text_input(
    "FastAPI base URL",
    value=os.getenv("API_BASE_URL", "http://localhost:8000"),
    help="Example: http://localhost:8000",
)

st.sidebar.caption("Tip: run FastAPI first, then submit a profile below.")

if st.sidebar.button("Check API health"):
    try:
        health_resp = requests.get(f"{api_base_url.rstrip('/')}/", timeout=10)
        health_resp.raise_for_status()
        st.sidebar.success(f"API healthy: {health_resp.json().get('message', 'OK')}")
    except requests.RequestException as exc:
        st.sidebar.error(f"Cannot reach API: {exc}")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
        industry = st.selectbox(
            "Industry",
            ["IT", "Healthcare", "Finance", "Education", "Manufacturing", "Retail", "Marketing"],
        )
        job_role = st.text_input("Job Role", value="Data Analyst")

    with col2:
        years_experience = st.number_input("Years Experience", min_value=0, max_value=50, value=8, step=1)
        ai_adoption = st.selectbox("AI Adoption Level", ["Low", "Medium", "High"])
        automation_risk = st.selectbox("Automation Risk", ["Low", "Medium", "High"])
        upskilling_required = st.selectbox("Upskilling Required", ["Yes", "No"])
        remote_work = st.selectbox("Remote Work", ["Yes", "No"])

    with col3:
        salary_before_ai = st.number_input("Salary Before AI", min_value=0, value=75000, step=500)
        salary_after_ai = st.number_input("Salary After AI", min_value=0, value=80000, step=500)
        work_hours = st.number_input("Work Hours Per Week", min_value=1, max_value=100, value=40, step=1)
        job_satisfaction = st.number_input("Job Satisfaction (1-10)", min_value=1, max_value=10, value=7, step=1)
        productivity_change = st.number_input("Productivity Change (%)", value=12.5, step=0.1)

    submitted = st.form_submit_button("Predict Job Status")

if submitted:
    payload = {
        "Age": int(age),
        "Gender": gender,
        "Education_Level": education_level,
        "Industry": industry,
        "Job_Role": job_role,
        "Years_Experience": int(years_experience),
        "AI_Adoption_Level": ai_adoption,
        "Automation_Risk": automation_risk,
        "Upskilling_Required": upskilling_required,
        "Salary_Before_AI": int(salary_before_ai),
        "Salary_After_AI": int(salary_after_ai),
        "Work_Hours_Per_Week": int(work_hours),
        "Remote_Work": remote_work,
        "Job_Satisfaction": int(job_satisfaction),
        "Productivity_Change_%": float(productivity_change),
    }

    st.session_state["prediction_result"] = None
    st.session_state["prediction_error"] = None

    try:
        response = requests.post(f"{api_base_url.rstrip('/')}/predict", json=payload, timeout=20)
        if response.status_code != 200:
            st.session_state["prediction_error"] = (
                f"Prediction failed with status {response.status_code}: {response.text}"
            )
        else:
            st.session_state["prediction_result"] = response.json()
    except requests.RequestException as exc:
        st.session_state["prediction_error"] = f"Prediction request failed: {exc}"
    except ValueError as exc:
        st.session_state["prediction_error"] = f"Prediction response could not be parsed as JSON: {exc}"

prediction_error = st.session_state.get("prediction_error")
prediction_result = st.session_state.get("prediction_result")

if prediction_error:
    st.error(prediction_error)

if prediction_result:
    st.success("Prediction completed")

    status_col, factor_col = st.columns([1, 2])
    with status_col:
        st.metric("Predicted Job Status", prediction_result["predicted_status"])
    with factor_col:
        st.info(f"Top Risk Factor: {prediction_result['top_risk_factor']}")

    probs_df = (
        pd.DataFrame(
            [{"Job_Status": label, "Probability": prob} for label, prob in prediction_result["probabilities"].items()]
        )
        .sort_values("Probability", ascending=False)
        .set_index("Job_Status")
    )

    st.subheader("Prediction Probabilities")
    st.bar_chart(probs_df)
    st.dataframe(probs_df, use_container_width=True)
