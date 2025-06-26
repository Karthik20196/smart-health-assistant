import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model and label encoder
model = load_model("symptom_disease_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# List of all symptoms (same order used during training)
all_symptoms = [
    "itching", "skin_rash", "nodal_skin_eruptions", "dischromic _patches",
    "continuous_sneezing", "shivering", "chills", "joint_pain",
    "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting",
    # Add the rest of your 100+ symptoms here
]

# App UI
st.set_page_config(page_title="Smart Health Assistant", page_icon="🩺")
st.title("🩺 Smart Disease Predictor")
st.markdown("Select symptoms and click Predict to get the most likely disease.")

# User input
selected = st.multiselect("Select Symptoms", all_symptoms)

# Prepare model input
input_vector = [1 if symptom in selected else 0 for symptom in all_symptoms]
input_array = np.array([input_vector])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_array)
    disease = le.inverse_transform([np.argmax(prediction)])[0]
    st.success(f"🧬 Predicted Disease: **{disease}**")
