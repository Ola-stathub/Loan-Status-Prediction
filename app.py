import streamlit as st
import pandas as pd
import pickle

with open(r"rf_loan_pred_pipeline", "rb") as model_file:
    pipeline = pickle.load(model_file)

st.title("Loan Approval Prediction")


gender = st.selectbox("Gender", options=["Male", "Female"])
married = st.selectbox("Married", options=["Yes", "No"])
dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"])
education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", options=["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0, value=3000)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0.0, value=100.0)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0.0, value=360.0)
credit_history = st.selectbox("Credit History", options=[1.0, 0.0])
property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])


input_dict = {
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area],
}

input_df = pd.DataFrame(input_dict)


if st.button("Predict Loan Status"):
    prediction = pipeline.predict(input_df)

    
    loan_status_map = {0: "Loan Denied", 1: "Loan Approved"}

    result = loan_status_map.get(prediction[0], prediction[0])

    st.success(f"Prediction: {result}")