import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("loan_xgb_model.pkl")
le = joblib.load("label_encoder.pkl")
num_cols = joblib.load("num_cols.pkl")
cat_cols = joblib.load("cat_cols.pkl")
df = pd.read_csv("loan_data.csv")

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

st.title("Loan Approval Prediction Dashboard")
st.header("Enter Applicant Details")

user_input = {}
with st.form("loan_form"):
    st.subheader("Numeric Features")
    for col in num_cols:
        val = st.number_input(col, value=0)
        user_input[col] = val

    st.subheader("Categorical Features")
    for col in cat_cols:
        options = sorted(df[col].unique())
        val = st.selectbox(col, options)
        user_input[col] = val

    submitted = st.form_submit_button("Submit")

if submitted:
    input_df = pd.DataFrame([user_input])
    for col in cat_cols:
        input_df[col] = le.transform(input_df[col])

    prob = model.predict_proba(input_df)[:,1][0]
    pred = model.predict(input_df)[0]

    st.subheader("Entered Data")
    st.dataframe(input_df)

    st.subheader("Loan Decision")
    if pred == 1:
        st.write("Loan Approved")
    else:
        st.write("Loan Rejected")

    st.subheader("Approval Probability")
    st.progress(int(prob*100))

    st.subheader("Risk Analysis")
    if prob >= 0.75:
        st.write("Low Risk")
    elif prob >= 0.5:
        st.write("Moderate Risk")
    else:
        st.write("High Risk")

    st.subheader("Visual Insights")
    for col in num_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df, x=col, hue='loan_status', bins=20, kde=False, palette='Set2', alpha=0.7)
        plt.axvline(input_df[col].values[0], color='red', linestyle='--')
        st.pyplot(plt)
        plt.clf()