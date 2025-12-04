import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load("loan_xgb_model.pkl")
encoders = joblib.load("label_encoders.pkl")
num_cols = joblib.load("num_cols.pkl")
cat_cols = joblib.load("cat_cols.pkl")

st.set_page_config(page_title="Loan Approval Prediction", layout="wide")

st.title("Loan Approval Prediction App")

with st.form("loan_form"):

    loan_intent = st.selectbox(
        "Loan Intent",
        options=sorted(encoders["loan_intent"].classes_)
    )

    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=25.0, step=0.01)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, step=0.01)

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (Years)", 
        min_value=0.0, max_value=20.0, step=0.1
    )

    credit_score = st.number_input("Credit Score", min_value=300.0, max_value=850.0, step=1.0)

    previous_loan_defaults_on_file = st.selectbox(
        "Previous Loan Defaults",
        ["No", "Yes"]
    )

    submit = st.form_submit_button("Submit")

if submit:

    input_data = {
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }

    df_input = pd.DataFrame([input_data])

    for col in cat_cols:
        df_input[col] = encoders[col].transform(df_input[col])

    prediction = model.predict(df_input)[0]
    prob = model.predict_proba(df_input)[0][1]

    st.subheader("Prediction Result")

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

    st.subheader("Feature Importance (XGBoost)")

    importances = model.named_steps["model"].feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(importances)), importances[sorted_idx])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(df_input.columns[sorted_idx], rotation=90)
    st.pyplot(fig)