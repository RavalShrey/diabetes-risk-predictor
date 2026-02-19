import joblib
import numpy as np


model = joblib.load("diabetes_model.pkl")

import streamlit as st
st.title("Diabities Risk Predictor")
st.write("Input patient details below:")

#input_fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

input_data = np.array([[ 
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    pedigree,
    age
]])

st.write("Patient data captured successsfully.")

st.sidebar.header("Clinic Configuration")

threshold = st.sidebar.slider(
    "High Risk Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

if st.button("Predict Risk"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    st.write(f"Diabetes Risk Probability: {probability*100:.2f}%")
    if probability >= threshold:
         st.error("High Risk")
    else:
        st.success("Lower Risk")
    
    st.subheader("Key Contributing Factors")

coefficients = model.coef_[0]
features = [
    "Pregnancies",
    "Glucose",
    "Blood Pressure",
    "Skin Thickness",
    "Insulin",
    "BMI",
    "Diabetes Pedigree Function",
    "Age"
]

feature_importance = list(zip(features, coefficients))

# Sort by absolute importance
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

for feature, coef in feature_importance[:3]:
    st.write(f"{feature} has significant influence on risk assessment.")

 


st.markdown("---")
st.caption(
    "⚠️ Disclaimer: This tool is intended for risk assessment support only. "
    "It does not replace professional medical diagnosis. "
    "Final clinical decisions should be made by qualified healthcare professionals."
)
