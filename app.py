import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Loan Approval Prediction", layout="wide")

# Load model files
model = joblib.load("loan_xgb_model.pkl")
le = joblib.load("label_encoder.pkl")
num_cols = joblib.load("num_cols.pkl")
cat_cols = joblib.load("cat_cols.pkl")

# Load raw dataset (before encoding)
df_raw = pd.read_csv("loan_data.csv")

# Get original raw categories
original_categories = {}
for col in cat_cols:
    original_categories[col] = sorted(df_raw[col].dropna().unique())

# Prepare encoded dataset for visualizations
df = df_raw.copy()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

st.title("Loan Approval Prediction Dashboard")

st.header("Enter Applicant Details")
user_input = {}

with st.form("loan_form"):
    st.subheader("Numeric Features")
    for col in num_cols:
        value = st.number_input(col, value=float(df_raw[col].median()))
        user_input[col] = value

    st.subheader("Categorical Features")
    for col in cat_cols:
        selected = st.selectbox(col, original_categories[col])
        encoded_value = le.transform([selected])[0]
        user_input[col] = encoded_value

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([user_input])

    st.subheader("Entered Data")
    st.dataframe(input_df)

    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.subheader("Loan Decision")
    st.write("Approved" if pred == 1 else "Rejected")

    st.subheader("Approval Probability")
    st.progress(int(prob * 100))
    st.write(f"**{prob*100:.2f}%**")

    st.subheader("Risk Analysis")
    if prob >= 0.75:
        st.write("Low Risk")
    elif prob >= 0.5:
        st.write("Moderate Risk")
    else:
        st.write("High Risk")

    st.subheader("Visual Insights")

    for col in num_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df, x=col, hue='loan_status', bins=20, ax=ax, alpha=0.6)
        ax.axvline(input_df[col].iloc[0], color='red', linestyle='--', linewidth=2)
        ax.set_title(f"{col} vs Loan Status")
        st.pyplot(fig)