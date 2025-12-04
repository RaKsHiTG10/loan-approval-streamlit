# Loan Approval Prediction - Streamlit Application

This project is a machine learning-based web application that predicts loan approval likelihood using a tuned XGBoost model. The Streamlit interface allows users to enter applicant information and instantly receive predictions, approval probability, risk factors, and visual insights.

Live App:  
https://loan-approval-stream.streamlit.app/

GitHub Repository:  
https://github.com/RaKsHiTG10/loan-approval-streamlit

---

## **Features**

- Predicts loan approval or rejection  
- Displays approval probability  
- Provides detailed risk factor analysis  
- Generates visual insights  
- User-friendly Streamlit interface  
- Uses an optimized XGBoost model trained on the loan approval dataset  

---

## **Project Structure**

loan-approval-streamlit/

│── app.py

│── loan_xgb_model.pkl

│── label_encoders.pkl

│── requirements.txt

│── README.md

---

---

## **Machine Learning Details**

### **Models Tested**
- Logistic Regression  
- Random Forest  
- Support Vector Machine  
- XGBoost  

### **Techniques Used**
- Standardization  
- Hyperparameter tuning  
- Evaluation using Accuracy, F1 Score, and AUC  
- Feature importance and SHAP analysis  

*Best performing model: Tuned XGBoost*

---

## **Performance**
- **Accuracy: ~93%**  
- **AUC: ~0.98**

---

## **Run Locally**

git clone https://github.com/RaKsHiTG10/loan-approval-streamlit.git

cd loan-approval-streamlit
pip install -r requirements.txt
streamlit run app.py


---

## **Technologies Used**

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- XGBoost  
- SHAP  
- Matplotlib  
- Streamlit  

---

## **Author**
Rakshit Gupta
