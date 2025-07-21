# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Title
st.title("🌿 Plant Disease Predictor")
st.write("Predict whether a plant has a disease based on environmental conditions.")

# Upload section
uploaded_file = st.file_uploader("Upload your CSV dataset (optional):", type=["csv"])

# Model training (only if no pre-trained model exists)
MODEL_PATH = "random_forest_plant_disease.pkl"

@st.cache_resource
def train_model():
    df = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv("balanced_dataset.csv")
    
    X = df.drop('disease_present', axis=1)
    y = df['disease_present']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42)
    rf.fit(X_scaled, y)
    
    joblib.dump((rf, scaler), MODEL_PATH)
    return rf, scaler

# Load or train the model
if os.path.exists(MODEL_PATH):
    rf_model, scaler = joblib.load(MODEL_PATH)
else:
    rf_model, scaler = train_model()

# User Input Form
st.subheader("🌡️ Enter Environmental Details")
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=10.0)
soil_pH = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)

if st.button("🧪 Predict"):
    input_data = np.array([[temperature, humidity, rainfall, soil_pH]])
    input_scaled = scaler.transform(input_data)
    prediction = rf_model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("🚨 The plant is likely diseased.")
    else:
        st.success("✅ The plant is healthy.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
