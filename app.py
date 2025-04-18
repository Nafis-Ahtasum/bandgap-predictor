import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model and scaler
model = joblib.load("xgb_bandgap_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Band Gap Predictor", layout="centered")
st.title("\U0001F52E Band Gap Predictor for Inorganic Double Perovskites")
st.markdown("Enter the chemical and structural properties below to predict the band gap (in eV).")

# Collecting user input
RA = st.number_input("Ionic Radius of A site (RA)", value=1.5)
IA = st.number_input("Ionization Energy of A site (IA)", value=10.0)
Rg_B = st.number_input("Ionic Radius of B site (Rg_B)", value=1.0)
VA_B = st.number_input("Valence Electrons of B site (VA_B)", value=8)
El_B = st.number_input("Formation Enthalpy of B site (Ел_B)", value=0.0)
PB = st.number_input("Polarizability of B site (PB)", value=50.0)
Rg_Bp = st.number_input("Ionic Radius of B' site (Rg_B')", value=1.0)
VA_Bp = st.number_input("Valence Electrons of B' site (VA_B')", value=8)
El_Bp = st.number_input("Formation Enthalpy of B' site (Ел_B')", value=0.0)
Rx = st.number_input("Ionic Radius of X site (Rx)", value=1.5)
Ex = st.number_input("Formation Enthalpy of X site (Ex)", value=0.0)
Zx = st.number_input("Atomic Number of X site (Zx)", value=17)

if st.button("Predict Band Gap"):
    input_data = pd.DataFrame({
        "RA": [RA],
        "IA": [IA],
        "Rg_B": [Rg_B],
        "VA_B": [VA_B],
        "Ел_B": [El_B],
        "PB": [PB],
        "Rg_B'": [Rg_Bp],
        "VA_B'": [VA_Bp],
        "Ел_B'": [El_Bp],
        "Rx": [Rx],
        "Ex": [Ex],
        "Zx": [Zx]
    })

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Band Gap: {prediction:.3f} eV")
