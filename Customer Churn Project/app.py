import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

st.title("Customer Churn Prediction System")

tenure = st.number_input("Tenure Months", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)

if st.button("Predict"):
    # Dummy input (replace with actual features later)
    sample = np.zeros((1, 23))
    sample[0][7] = tenure
    sample[0][20] = monthly_charges

    prediction = model.predict(sample)
    probability = model.predict_proba(sample)

    st.write("Probability of staying:", round(probability[0][0] * 100, 2), "%")
    st.write("Probability of churn:", round(probability[0][1] * 100, 2), "%")

    if prediction[0] == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is likely to stay")