import streamlit as st
import numpy as np
import joblib

# Load files
model = joblib.load("mlp_model.pkl")
le = joblib.load("label_encoder.pkl")
all_symptoms = joblib.load("all_symptoms.pkl")

# UI
st.set_page_config(page_title="Smart Health Assistant", page_icon="🩺")
st.title("🩺 Smart Disease Predictor")
st.markdown("Select symptoms and click **Predict** to see the likely disease.")

# Multi-select
selected = st.multiselect("Select Symptoms", all_symptoms)

# One-hot encode input
input_vector = [1 if symptom in selected else 0 for symptom in all_symptoms]
input_array = np.array([input_vector])

# Predict
if st.button("Predict"):
    pred = model.predict(input_array)[0]
    disease = le.inverse_transform([pred])[0]
    st.success(f"🧬 Predicted Disease: **{disease}**")
