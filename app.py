import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

@st.cache_resource
def load_files():
    model = joblib.load("loan_xgb_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")   
    num_cols = joblib.load("num_cols.pkl")
    cat_cols = joblib.load("cat_cols.pkl")
    return model, label_encoder, num_cols, cat_cols

model, label_encoder, num_cols, cat_cols = load_files()


st.title("Loan Approval Prediction App")

st.sidebar.header("Enter Applicant Details")



def user_input():

    data = {
        "person_age": st.sidebar.number_input("Age", min_value=18, max_value=96, value=30),

        "person_gender": st.sidebar.selectbox("Gender", options=[0, 1]),

        "person_education": st.sidebar.selectbox("Education Level (0–5)", options=[0, 1, 2, 3, 4, 5]),

        "person_income": st.sidebar.number_input("Annual Income", min_value=0, max_value=500000, value=50000),

        "person_emp_exp": st.sidebar.number_input("Employment Experience (Years)", min_value=0, max_value=50, value=1),

        "person_home_ownership": st.sidebar.selectbox("Home Ownership (0–4)", options=[0, 1, 2, 3, 4]),

        "loan_amnt": st.sidebar.number_input("Loan Amount", min_value=500, max_value=80000, value=5000),

        "loan_intent": st.sidebar.selectbox(
            "Loan Intent (0-5)", options=[0, 1, 2, 3, 4, 5]
        ),

        "loan_int_rate": st.sidebar.number_input(
            "Interest Rate (%)",
            min_value=5.42,
            max_value=19.59,
            value=10.0
        ),

        "loan_percent_income": st.sidebar.number_input(
            "Loan Percent of Income (0-0.37)",
            min_value=0.0,
            max_value=0.37,
            value=0.20,
            step=0.01
        ),

        "cb_person_cred_hist_length": st.sidebar.number_input(
            "Credit History Length (Years)",
            min_value=2.0,
            max_value=15.5,
            value=3.0
        ),

        "credit_score": st.sidebar.number_input(
            "Credit Score",
            min_value=497.5,
            max_value=773.5,
            value=600.0
        ),

        "previous_loan_defaults_on_file": st.sidebar.selectbox("Previous Default", [0, 1]),
    }

    return pd.DataFrame([data])


df_input = user_input()

st.subheader("Entered Details:")
st.write(df_input)


if st.button("Submit"):

    df_model = df_input.copy()

    # encode categorical cols using single label encoder
    for col in cat_cols:
        df_model[col] = df_model[col].astype(str)
        df_model[col] = label_encoder.fit_transform(df_model[col])

    # prediction
    pred = model.predict(df_model)[0]
    prob = model.predict_proba(df_model)[0][1]

    st.subheader("Prediction Result")
    st.write("###  Approved" if pred == 1 else "###  Rejected")
    st.write(f"**Approval Probability:** `{prob:.2f}`")

    # Feature importance
    st.subheader("Feature Importance (XGBoost)")
    fig_imp, ax = plt.subplots()
    xgb.plot_importance(model, ax=ax)
    st.pyplot(fig_imp)

    # SHAP
    st.subheader("SHAP Summary Plot")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_model)

    fig_shap = plt.figure()
    shap.summary_plot(shap_values, df_model, show=False)
    st.pyplot(fig_shap)
