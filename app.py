import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

@st.cache_resource
def load_files():
    model = joblib.load("loan_xgb_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    num_cols = joblib.load("num_cols.pkl")
    cat_cols = joblib.load("cat_cols.pkl")
    return model, label_encoders, num_cols, cat_cols

model, label_encoders, num_cols, cat_cols = load_files()

st.title("Loan Approval Prediction")

st.sidebar.header("Input Features")

def user_input():
    data = {
        "person_age": st.sidebar.number_input("Age", 18, 100, 30),
        "person_gender": st.sidebar.selectbox("Gender (0=Male, 1=Female)", [0, 1]),
        "person_education": st.sidebar.number_input("Education Level (0–5)", 0, 5, 2),
        "person_income": st.sidebar.number_input("Income", 0, 500000, 50000),
        "person_emp_exp": st.sidebar.number_input("Employment Experience", 0, 50, 1),
        "person_home_ownership": st.sidebar.number_input("Home Ownership (0–4)", 0, 4, 1),
        "loan_amnt": st.sidebar.number_input("Loan Amount", 0, 100000, 5000),
        "loan_intent": st.sidebar.number_input("Loan Intent (0–6)", 0, 6, 2),
        "loan_int_rate": st.sidebar.number_input("Interest Rate (%)", 1.0, 40.0, 10.0),
        "loan_percent_income": st.sidebar.number_input("Loan Percent Income", 0.0, 2.0, 0.2),
        "cb_person_cred_hist_length": st.sidebar.number_input("Credit History Length", 0, 50, 3),
        "credit_score": st.sidebar.number_input("Credit Score (300–850)", 300, 850, 600),
        "previous_loan_defaults_on_file": st.sidebar.selectbox("Previous Defaults (0/1)", [0, 1]),
    }
    return pd.DataFrame([data])

df_input = user_input()

st.subheader("Input Data")
st.write(df_input)

if st.button("Submit"):
    df_model = df_input.copy()

    for col in cat_cols:
        df_model[col] = label_encoders[col].transform(df_model[col].astype(str))

    pred = model.predict(df_model)[0]
    prob = model.predict_proba(df_model)[0][1]

    st.subheader("Prediction Result")
    st.write("Approved" if pred == 1 else "Rejected")
    st.write(f"Approval Probability: {prob:.2f}")

    st.subheader("Feature Importance")
    fig_imp, ax = plt.subplots()
    xgb.plot_importance(model, ax=ax)
    st.pyplot(fig_imp)

    st.subheader("SHAP Summary Plot")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_model)

    fig_shap = plt.figure()
    shap.summary_plot(shap_values, df_model, show=False)
    st.pyplot(fig_shap)
