import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Approval Prediction App", layout="wide")

model = joblib.load("loan_xgb_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.title("Loan Approval Prediction App")

with st.form("loan_form"):
    person_age = st.number_input("Age", min_value=20.0, max_value=39.0, step=1.0)

    gender_map = {"Male": 1.0, "Female": 0.0}
    person_gender = st.selectbox("Gender", options=list(gender_map.keys()))
    
    education_map = {
        "High School": 0.0,
        "Associate": 1.0,
        "Bachelor": 2.0,
        "Master": 3.0,
        "Doctorate": 4.0
    }
    person_education = st.selectbox("Education Level", options=list(education_map.keys()))
    
    home_map = {
        "Rent": 0.0,
        "Mortgage": 1.0,
        "Own": 2.0,
        "Other": 3.0
    }
    person_home_ownership = st.selectbox("Home Ownership", options=list(home_map.keys()))
    
    intent_map = {
        "Debt Consolidation": 0.0,
        "Home Improvements": 1.0,
        "Major Purchase": 2.0,
        "Medical": 3.0,
        "Small Business": 4.0,
        "Education": 5.0
    }
    loan_intent = st.selectbox("Loan Intent", options=list(intent_map.keys()))

    person_income = st.number_input("Annual Income", min_value=8000.0, max_value=168667.125, step=100.0)
    person_emp_exp = st.number_input("Employment Experience (years)", min_value=0.0, max_value=18.5, step=0.5)
    loan_amnt = st.number_input("Loan Amount", min_value=500.0, max_value=23093.125, step=100.0)
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=5.42, max_value=19.59, step=0.01)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=0.37, step=0.01)
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=2.0, max_value=15.5, step=0.5)
    credit_score = st.number_input("Credit Score", min_value=497.5, max_value=773.5, step=1.0)

    previous_map = {"No": 0.0, "Yes": 1.0}
    previous_default = st.selectbox("Previous Loan Defaults", options=list(previous_map.keys()))
    
    submit = st.form_submit_button("Submit")

if submit:
    input_data = {
        "person_age": person_age,
        "person_gender": gender_map[person_gender],
        "person_education": education_map[person_education],
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": home_map[person_home_ownership],
        "loan_amnt": loan_amnt,
        "loan_intent": intent_map[loan_intent],
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_map[previous_default]
    }

    df_input = pd.DataFrame([input_data])

    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    st.subheader("Prediction")
    if prediction == 1:
        st.success(f"Loan Approved (Probability: {prob:.2f})")
    else:
        st.error(f"Loan Not Approved (Probability: {prob:.2f})")

    st.subheader("SHAP Feature Importance")
    try:
        xgb_model = model.named_steps["model"]
        scaler = model.named_steps["scaler"]
        explainer = shap.Explainer(xgb_model)
        shap_values = explainer(scaler.transform(df_input))
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning("SHAP could not be generated.")
        st.text(str(e))

    st.subheader("XGBoost Feature Importance")
    importances = model.named_steps["model"].feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[order])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(df_input.columns[order], rotation=90)
    st.pyplot(fig)