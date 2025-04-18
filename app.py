import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("xgb_bandgap_model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("🔮 Band Gap Predictor for Inorganic Double Perovskites")
st.write("Enter the chemical and structural properties below to predict the band gap (in eV).")

# Input fields (only selected features)
RA = st.number_input("Ionic Radius of A site (RA)", value=1.50)
IA = st.number_input("Ionization Energy of A site (IA)", value=10.00)
Rg_B = st.number_input("Ionic Radius of B site (Rg_B)", value=1.00)
VA_B = st.number_input("Valence Electrons of B site (VA_B)", value=8)
Rg_Bp = st.number_input("Ionic Radius of B' site (Rg_Bp)", value=1.00)
VA_Bp = st.number_input("Valence Electrons of B' site (VA_Bp)", value=8)
P_Bp = st.number_input("Polarizability of B' site (P_Bp)", value=50.00)
Rx = st.number_input("Ionic Radius of X site (Rx)", value=1.50)

# Create input dataframe
input_df = pd.DataFrame([{
    "RA": RA,
    "IA": IA,
    "Rg_B": Rg_B,
    "VA_B": VA_B,
    "Rg_Bp": Rg_Bp,
    "VA_Bp": VA_Bp,
    "P_Bp": P_Bp,
    "Rx": Rx
}])

# Predict
if st.button("Predict Band Gap"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"🎯 Predicted Band Gap: {prediction[0]:.3f} eV")
