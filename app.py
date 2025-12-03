import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

model = joblib.load("loan_xgb_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
num_cols = joblib.load("num_cols.pkl")
cat_cols = joblib.load("cat_cols.pkl")

st.title("Loan Approval Prediction App")

st.subheader("Enter Applicant Details")

input_data = {}

for col in num_cols:
    input_data[col] = st.number_input(col, value=0.0)

for col in cat_cols:
    input_data[col] = st.selectbox(col, options=label_encoder[col].classes_)

df = pd.DataFrame([input_data])

for col in cat_cols:
    df[col] = label_encoder[col].transform(df[col])

if st.button("Submit"):
    prediction = model.predict(df)[0]
    result = "Approved" if prediction == 1 else "Rejected"
    st.subheader(f"Prediction: {result}")

    st.subheader("Feature Importance")
    fig, ax = plt.subplots()
    ax.bar(model.feature_importances_.argsort(), np.sort(model.feature_importances_))
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Importance")
    st.pyplot(fig)

    st.subheader("SHAP Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values, df, show=False)
    st.pyplot(fig2)