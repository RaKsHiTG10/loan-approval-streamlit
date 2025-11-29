import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('xgb_best_model.pkl')

st.title("Loan Approval Prediction")

age = st.number_input("Age", min_value=18, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male","Female"])
education = st.selectbox("Education Level", ["High School","Bachelor","Master","PhD","Other"])
income = st.number_input("Income", min_value=0, value=50000)
emp_exp = st.number_input("Employment Years", min_value=0, value=2)
home = st.selectbox("Home Ownership", ["RENT","MORTGAGE","OWN","OTHER"])
loan_amt = st.number_input("Loan Amount", min_value=0, value=10000)
loan_intent = st.selectbox("Loan Purpose", ["EDUCATION","MEDICAL","PERSONAL","VENTURE","HOME_IMPROVEMENT","OTHER"])
loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, value=12.0)
loan_percent_income = st.number_input("Loan % of Income", min_value=0.0, value=0.1)
cred_hist = st.number_input("Credit History Length", min_value=0, value=5)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
prev_default = st.selectbox("Previous Loan Defaults", ["Yes","No"])

user_input = pd.DataFrame({
    'person_age':[age],
    'person_gender':[0 if gender=="Male" else 1],
    'person_education':[0 if education=="High School" else 1 if education=="Bachelor" else 2 if education=="Master" else 3 if education=="PhD" else 4],
    'person_income':[income],
    'person_emp_exp':[emp_exp],
    'person_home_ownership':[0 if home=="RENT" else 1 if home=="MORTGAGE" else 2 if home=="OWN" else 3],
    'loan_amnt':[loan_amt],
    'loan_intent':[0 if loan_intent=="EDUCATION" else 1 if loan_intent=="MEDICAL" else 2 if loan_intent=="PERSONAL" else 3 if loan_intent=="VENTURE" else 4 if loan_intent=="HOME_IMPROVEMENT" else 5],
    'loan_int_rate':[loan_int_rate],
    'loan_percent_income':[loan_percent_income],
    'cb_person_cred_hist_length':[cred_hist],
    'credit_score':[credit_score],
    'previous_loan_defaults_on_file':[0 if prev_default=="No" else 1]
})

prediction = model.predict(user_input)[0]
st.write("Loan Status Prediction:", "Approved" if prediction==1 else "Rejected")
