import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

st.set_page_config(page_title="Loan Approval System", layout="wide")

model = joblib.load("loan_xgb_model.pkl")
label_encoders = joblib.load("label_encoder.pkl")
num_cols = joblib.load("num_cols.pkl")
cat_cols = joblib.load("cat_cols.pkl")

st.title("Loan Approval Prediction System")

st.subheader("Enter Applicant Details")
user_input = {}

col1, col2 = st.columns(2)
with col1:
    for col in num_cols:
        user_input[col] = st.number_input(col, value=0.0)

with col2:
    for col in cat_cols:
        options = label_encoders[col].classes_
        selected = st.selectbox(col, options)
        user_input[col] = selected

input_df = pd.DataFrame([user_input])

for col in cat_cols:
    input_df[col] = label_encoders[col].transform(input_df[col])

st.divider()

st.subheader("Prediction Results")

prediction = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]

decision = "Approved" if prediction == 1 else "Rejected"

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Loan Decision", decision)
with c2:
    st.metric("Approval Probability", f"{prob*100:.2f}%")
with c3:
    risk = "Low Risk" if prob > 0.7 else "Medium Risk" if prob > 0.4 else "High Risk"
    st.metric("Risk Analysis", risk)

st.divider()

st.subheader("Visual Insights")

fig, ax = plt.subplots()
shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)
shap.force_plot(explainer.expected_value, shap_values, input_df, matplotlib=True)
st.pyplot(fig)

st.write("")
st.subheader("Feature Importance")
fig2, ax2 = plt.subplots()
importance = model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]
ax2.bar(range(len(importance)), importance[sorted_idx])
ax2.set_xticks(range(len(importance)))
ax2.set_xticklabels(input_df.columns[sorted_idx], rotation=90)
st.pyplot(fig2)

st.write("")
st.subheader("Column vs Target Comparison")

uploaded_df = st.file_uploader("Upload Dataset with Target Column (Optional)", type=["csv"])

if uploaded_df:
    df = pd.read_csv(uploaded_df)
    if "Loan_Status" in df.columns:
        col_to_show = st.selectbox("Select Column", df.columns)
        fig3, ax3 = plt.subplots()
        df.groupby(col_to_show)["Loan_Status"].mean().plot(kind="bar", ax=ax3)
        ax3.set_ylabel("Approval Rate")
        st.pyplot(fig3)
    else:
        st.error("Dataset must contain 'Loan_Status' column.")