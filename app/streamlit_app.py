import streamlit as st
import joblib
import pandas as pd

model = joblib.load('../models/best_model.joblib')

st.title("Customer Churn Demo")
st.write("Introduce datos del cliente")

# inputs manuales (ejemplo)
tenure = st.number_input("Tenure (meses)", min_value=0, max_value=200, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
contract = st.selectbox("Contract", ['Month-to-month','One year','Two year'])

input_df = pd.DataFrame({
    'tenure':[tenure],
    'monthlycharges':[monthly_charges],
    'contract':[contract],
    # añade aquí las columnas necesarias y con los mismos nombres que en entrenamiento
})

prob = model.predict_proba(input_df)[:,1][0]
st.metric("Probabilidad de churn", f"{prob:.2%}")