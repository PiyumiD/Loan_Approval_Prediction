import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model pipeline
@st.cache_resource
def load_model():
    with open('loan_approval_pipeline.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# App title and description
st.title("Loan Approval Prediction")
st.write("""
This app predicts whether a loan application will be approved or not based on user inputs.
Please fill in the form below and click **Predict**.
""")

# Small text header for "Enter Loan Application Details"
st.markdown("<h3 style='font-size: 18px;'>Enter Loan Application Details</h3>", unsafe_allow_html=True)

# Input form for user data
with st.form("loan_form"):
    # Create two columns for input fields
    col1, col2 = st.columns(2)

    # Input fields for the first column
    with col1:
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0, step=1)
        education = st.selectbox("Education Level", options=["Graduate", "Non-Graduate"], index=0)
        self_employed = st.selectbox("Self Employed", options=["Yes", "No"], index=1)
        income_annum = st.number_input("Annual Income (in ₹)", min_value=0, value=50000, step=1000)
        loan_amount = st.number_input("Loan Amount (in ₹)", min_value=0, value=10000, step=1000)

    # Input fields for the second column
    with col2:
        loan_term = st.number_input("Loan Term (in months)", min_value=1, max_value=360, value=36, step=1)
        cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750, step=1)
        residential_assets_value = st.number_input("Residential Assets Value (in ₹)", min_value=0, value=100000, step=1000)
        commercial_assets_value = st.number_input("Commercial Assets Value (in ₹)", min_value=0, value=50000, step=1000)
        luxury_assets_value = st.number_input("Luxury Assets Value (in ₹)", min_value=0, value=10000, step=1000)
        bank_asset_value = st.number_input("Bank Asset Value (in ₹)", min_value=0, value=20000, step=1000)

    # Submit button
    submitted = st.form_submit_button("Predict")

# Perform prediction
if submitted:
    # Preprocess inputs
    user_data = pd.DataFrame([{
        "no_of_dependents": no_of_dependents,
        "education": 1 if education == "Graduate" else 0,  # Encoding Graduate as 1, Non-Graduate as 0
        "self_employed": 1 if self_employed == "Yes" else 0,  # Encoding Yes as 1, No as 0
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }])

    # Perform prediction
    prediction = model.predict(user_data)
    prediction_proba = model.predict_proba(user_data)[:, 1]  # Probability for "Approved"

    # Threshold for classification
    threshold = 0.5  # Adjust this if needed
    if prediction_proba[0] >= threshold:
        st.error(f"Loan Not Approved ❌")
    else:
        st.success(f"Loan Approved ✅")
