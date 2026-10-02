import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_icon("⚙️")
st.title("⚙️ Vehicle Maintenance Analysis Dashboard")

@st.cache_resource
def load_models():
    return joblib.load("final_model.joblib"), joblib.load("scaler.joblib"), joblib.load("feature_names.joblib")

try:
    model, scaler, features = load_models()
    rpm = st.number_input("Engine RPM", 50.0, 3000.0, 700.0)
    lub_p = st.number_input("Lub Oil Pressure (bar)", 0.0, 10.0, 2.49)
    fuel_p = st.number_input("Fuel Pressure (bar)", 0.0, 30.0, 11.79)
    cool_p = st.number_input("Coolant Pressure (bar)", 0.0, 10.0, 3.17)
    lub_t = st.number_input("Lub Oil Temp (°C)", 50.0, 120.0, 84.14)
    cool_t = st.number_input("Coolant Temp (°C)", 50.0, 220.0, 81.63)

    if st.button("PREDICT MAINTENANCE REQUIREMENT"):
        inp = scaler.transform(pd.DataFrame([[rpm, lub_p, fuel_p, cool_p, lub_t, cool_t]], columns=features))
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0]
        if pred == 1:
            st.error(f"⚠️ MAINTENANCE REQUIRED (Confidence: {np.max(prob)*100:.2f}%)")
        else:
            st.success(f"✅ NORMAL CONDITION (Confidence: {np.max(prob)*100:.2f}%)")
except Exception as e:
    st.error(f"Error loading model: {e}")
