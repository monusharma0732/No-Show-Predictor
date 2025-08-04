# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load saved model and preprocessor
model = joblib.load('rf_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.title("🩺 Patient Appointment No-Show Predictor")

# User Inputs
gender = st.selectbox("Gender", ["F", "M"])
neighbourhood = st.text_input("Neighbourhood", "JARDIM DA PENHA")
scholarship = st.selectbox("Scholarship", [0, 1])
hipertension = st.selectbox("Hipertension", [0, 1])
diabetes = st.selectbox("Diabetes", [0, 1])
alcoholism = st.selectbox("Alcoholism", [0, 1])
sms_received = st.selectbox("SMS Received", [0, 1])
age = st.slider("Age", 0, 115, 30)
handcap = st.selectbox("Handcap", [0, 1])
waiting_days = st.slider("Waiting Days", -7, 150, 10)

# Predict Button
if st.button("Predict No-Show"):
    input_df = pd.DataFrame({
        'Gender': [gender],
        'Neighbourhood': [neighbourhood],
        'Scholarship': [scholarship],
        'Hipertension': [hipertension],
        'Diabetes': [diabetes],
        'Alcoholism': [alcoholism],
        'SMS_received': [sms_received],
        'Age': [age],
        'Handcap': [handcap],
        'WaitingDays': [waiting_days]
    })

    input_encoded = preprocessor.transform(input_df)
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.error(f"❌ The patient is likely to miss the appointment. (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ The patient is likely to show up. (Confidence: {1 - probability:.2f})")
