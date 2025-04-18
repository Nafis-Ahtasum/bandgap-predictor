import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("xgb_perovskite_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Band Gap Predictor", layout="centered")

st.title("🔮 Band Gap Predictor for Inorganic Double Perovskites")

st.markdown("Enter the chemical and structural properties below to predict the **band gap** (in eV).")

# Define input fields
def user_input_features():
    RA = st.number_input("Ionic Radius of A site (RA)", 0.0, 3.0, 1.5)
    IA = st.number_input("Ionization Energy of A site (IA)", 0.0, 20.0, 10.0)
    
    Rg_B = st.number_input("Ionic Radius of B site", 0.0, 2.0, 1.0)
    VB_B = st.number_input("Valence Electrons of B site", 0, 20, 8)
    EB_B = st.number_input("Formation Enthalpy of B site", -10.0, 10.0, 0.0)

    Rg_Bp = st.number_input("Ionic Radius of B' site", 0.0, 2.0, 1.0)
    VB_Bp = st.number_input("Valence Electrons of B' site", 0, 20, 8)
    PB_Bp = st.number_input("Polarizability of B' site", 0.0, 100.0, 50.0)
    EB_Bp = st.number_input("Formation Enthalpy of B' site", -10.0, 10.0, 0.0)

    Rx = st.number_input("Ionic Radius of X site", 0.0, 2.5, 1.5)
    Ex = st.number_input("Formation Enthalpy of X site", -10.0, 10.0, 0.0)

    BEBX = st.number_input("Bond Energy of B-X", 0.0, 500.0, 200.0)
    BEBpX = st.number_input("Bond Energy of B'-X", 0.0, 500.0, 200.0)

    data = {
        'RA': RA, 'IA': IA, 'Rg_B': Rg_B, 'VB_B': VB_B, 'EB_B': EB_B,
        'Rg_Bp': Rg_Bp, 'VB_Bp': VB_Bp, 'PB_Bp': PB_Bp, 'EB_Bp': EB_Bp,
        'Rx': Rx, 'Ex': Ex, 'BEBX': BEBX, 'BEBpX': BEBpX
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Predict button
if st.button("Predict Band Gap"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"🎯 Predicted Band Gap: {prediction[0]:.4f} eV")
