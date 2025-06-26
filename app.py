import streamlit as st
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("mlp_model.pkl")
le = joblib.load("label_encoder.pkl")
all_symptoms = joblib.load("all_symptoms.pkl")

# Disease tips dictionary
disease_tips = {
    "Fungal infection": "🧴 Keep the area clean and dry. Use antifungal creams as advised.",
    "Allergy": "🚫 Avoid allergens and take antihistamines. Consult your physician if needed.",
    "GERD": "🍽️ Eat small meals. Avoid spicy food and lying down after eating.",
    "Migraine": "💊 Rest in a dark, quiet room. Stay hydrated. Avoid triggers like stress or caffeine.",
    "Acne": "🧼 Wash your face regularly. Avoid oily products. Seek dermatological advice.",
}
st.set_page_config(page_title="Smart Health Assistant", page_icon="🩺")
st.title("🧠 Smart Health Assistant")
st.markdown("> Enter your symptoms to get predicted diseases along with helpful health tips.")
selected_symptoms = st.multiselect("Select your symptoms:", all_symptoms)

# Predict button
if st.button("Predict"):
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]
    input_array = np.array([input_vector])
    
    # Top 3 predictions
    probs = model.predict_proba(input_array)[0]
    top3 = np.argsort(probs)[::-1][:3]

    st.markdown("### 🔬 Top 3 Predicted Diseases:")
    for i in top3:
        disease = le.inverse_transform([i])[0]
        confidence = probs[i] * 100
        st.write(f"**{disease}** — {confidence:.2f}%")
        
        # Show tips if available
        if disease in disease_tips:
            st.info(f"💡 Tip: {disease_tips[disease]}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by **Kuruva Karthik** · Final Year CSE ·")
