import streamlit as st
import numpy as np
import pandas as pd
import shap
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt


@st.cache_resource
def load_files():
    with open("loan_xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("num_cols.pkl", "rb") as f:
        num_cols = pickle.load(f)
    with open("cat_cols.pkl", "rb") as f:
        cat_cols = pickle.load(f)
    return model, label_encoders, num_cols, cat_cols


model, label_encoders, num_cols, cat_cols = load_files()

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

st.title("Loan Approval Prediction Dashboard")
st.write("Enter applicant and loan details to check approval likelihood.")


st.sidebar.header("User Input Parameters")

def get_user_input():
    person_age = st.sidebar.number_input("Age", 18, 100, 25)
    person_gender = st.sidebar.selectbox("Gender (0=Male, 1=Female)", [0, 1])
    person_education = st.sidebar.number_input("Education Level (0–5)", 0, 5, 3)
    person_income = st.sidebar.number_input("Annual Income", 0, 500000, 50000)
    person_emp_exp = st.sidebar.number_input("Employment Experience (Years)", 0, 50, 1)
    person_home_ownership = st.sidebar.number_input("Home Ownership (0–4)", 0, 4, 3)
    loan_amnt = st.sidebar.number_input("Loan Amount", 0, 100000, 5000)
    loan_intent = st.sidebar.number_input("Loan Intent (0–6)", 0, 6, 2)
    loan_int_rate = st.sidebar.number_input("Interest Rate (%)", 1.0, 40.0, 10.0)
    loan_percent_income = st.sidebar.number_input("Loan Percent Income (0–2)", 0.0, 2.0, 0.2)
    cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length", 0, 50, 3)
    credit_score = st.sidebar.number_input("Credit Score (300–850)", 300, 850, 600)
    previous_loan_defaults_on_file = st.sidebar.selectbox("Previous Defaults (0/1)", [0, 1])

    data = {
        "person_age": person_age,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }

    df = pd.DataFrame([data])
    return df

df_input = get_user_input()

st.subheader("Input Data")
st.write(df_input)

def encode_inputs(df):
    df_copy = df.copy()
    for col in cat_cols:
        if col in df_copy.columns:
            le = label_encoders[col]
            df_copy[col] = le.transform(df_copy[col].astype(str))
    return df_copy

encoded_df = encode_inputs(df_input)

prediction = model.predict(encoded_df)[0]
prob = model.predict_proba(encoded_df)[0][1]

st.subheader("Prediction Result")
st.write(f"**Loan Status Prediction:** {'Approved' if prediction == 1 else 'Rejected'}")
st.write(f"**Approval Probability:** {prob:.2f}")


st.subheader("Feature Importance")

fig, ax = plt.subplots()
xgb.plot_importance(model, ax=ax)
st.pyplot(fig)


st.subheader("SHAP Explainability")

explainer = shap.Explainer(model)
shap_values = explainer(encoded_df)

st.write("### SHAP Summary Plot")
fig_summary = plt.figure()
shap.summary_plot(shap_values.values, encoded_df, show=False)
st.pyplot(fig_summary)

st.write("### SHAP Force Plot (Single Prediction)")
fig_force = shap.plots.force(shap_values[0], matplotlib=True)
st.pyplot(fig_force)
