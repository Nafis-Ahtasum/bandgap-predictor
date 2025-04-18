import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("xgb_bandgap_model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("🔮 Band Gap Predictor for Inorganic Double Perovskites")
st.write("Enter the chemical and structural properties below to predict the band gap (in eV).")

# Input fields
RA = st.number_input("Ionic Radius of A site (RA)", value=1.50)
IA = st.number_input("Ionization Energy of A site (IA)", value=10.00)
Rg_B = st.number_input("Ionic Radius of B site (Rg_B)", value=1.00)
VA_B = st.number_input("Valence Electrons of B site (VA_B)", value=8)
H_B = st.number_input("Formation Enthalpy of B site (H_B)", value=0.00)
Rg_Bp = st.number_input("Ionic Radius of B' site (Rg_Bp)", value=1.00)
VA_Bp = st.number_input("Valence Electrons of B' site (VA_Bp)", value=8)
P_Bp = st.number_input("Polarizability of B' site (P_Bp)", value=50.00)
H_Bp = st.number_input("Formation Enthalpy of B' site (H_Bp)", value=0.00)
Rx = st.number_input("Ionic Radius of X site (Rx)", value=1.50)
H_X = st.number_input("Formation Enthalpy of X site (H_X)", value=0.00)
EB_B_X = st.number_input("Bond Energy of B-X (EB_B_X)", value=200.00)
EB_Bp_X = st.number_input("Bond Energy of B'-X (EB_Bp_X)", value=200.00)

# Create input dataframe
input_df = pd.DataFrame([{
    "RA": RA,
    "IA": IA,
    "Rg_B": Rg_B,
    "VA_B": VA_B,
    "H_B": H_B,
    "Rg_Bp": Rg_Bp,
    "VA_Bp": VA_Bp,
    "P_Bp": P_Bp,
    "H_Bp": H_Bp,
    "Rx": Rx,
    "H_X": H_X,
    "EB_B_X": EB_B_X,
    "EB_Bp_X": EB_Bp_X
}])

# Predict
if st.button("Predict Band Gap"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Band Gap: {prediction[0]:.3f} eV")

