import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Loan Approval Prediction", layout="wide")

model = joblib.load("xgb_best_model.pkl")

st.title("Loan Approval Prediction Dashboard")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD", "Other"])
    income = st.number_input("Annual Income", 0, 5000000, 50000)
    emp_exp = st.number_input("Employment Years", 0, 40, 2)
    home = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])

with col2:
    loan_amt = st.number_input("Loan Amount", 0, 500000, 10000)
    loan_intent = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "HOME_IMPROVEMENT", "OTHER"])
    loan_int_rate = st.number_input("Loan Interest Rate (%)", 0.0, 30.0, 12.0)
    loan_percent_income = st.number_input("Loan % of Income", 0.0, 1.0, 0.10)
    cred_hist = st.number_input("Credit History Length", 0, 40, 5)
    credit_score = st.number_input("Credit Score", 300, 850, 650)
    prev_default = st.selectbox("Previous Loan Default", ["Yes", "No"])

mapping = {
    "Gender": {"Male": 0, "Female": 1},
    "Education": {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3, "Other": 4},
    "Home": {"RENT": 0, "MORTGAGE": 1, "OWN": 2, "OTHER": 3},
    "Intent": {"EDUCATION": 0, "MEDICAL": 1, "PERSONAL": 2, "VENTURE": 3, "HOME_IMPROVEMENT": 4, "OTHER": 5},
    "Default": {"Yes": 1, "No": 0}
}

input_data = pd.DataFrame({
    "person_age": [age],
    "person_gender": [mapping["Gender"][gender]],
    "person_education": [mapping["Education"][education]],
    "person_income": [income],
    "person_emp_exp": [emp_exp],
    "person_home_ownership": [mapping["Home"][home]],
    "loan_amnt": [loan_amt],
    "loan_intent": [mapping["Intent"][loan_intent]],
    "loan_int_rate": [loan_int_rate],
    "loan_percent_income": [loan_percent_income],
    "cb_person_cred_hist_length": [cred_hist],
    "credit_score": [credit_score],
    "previous_loan_defaults_on_file": [mapping["Default"][prev_default]]
})

st.subheader("Entered Data")
st.dataframe(input_data)

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

status = "Approved" if prediction == 1 else "Rejected"

st.subheader("Loan Decision")
st.markdown(
    f"<h2 style='color:{'green' if status=='Approved' else 'red'}'>{status}</h2>",
    unsafe_allow_html=True
)

st.metric("Approval Probability", f"{probability*100:.2f}%")

risk = 100 - (probability * 100)
st.progress(int(risk))

st.subheader("Risk Analysis")

risks = []

if credit_score < 600:
    risks.append("Low credit score")
if loan_percent_income > 0.30:
    risks.append("Loan amount is too high relative to income")
if prev_default == "Yes":
    risks.append("Previous loan default history")
if emp_exp < 1:
    risks.append("Very low employment experience")
if loan_int_rate > 15:
    risks.append("High interest rate")

if len(risks) == 0:
    st.success("No major risk factors detected")
else:
    for r in risks:
        st.error(r)

st.subheader("Visual Insights")

fig1 = px.bar(
    x=["Income", "Loan Amount"],
    y=[income, loan_amt],
    title="Income vs Loan Amount"
)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(
    x=["Credit Score", "Credit History Length"],
    y=[credit_score, cred_hist],
    markers=True,
    title="Credit Health Indicators"
)
st.plotly_chart(fig2, use_container_width=True)