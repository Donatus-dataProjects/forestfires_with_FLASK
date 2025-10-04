import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Streamlit app title
st.title("🔥 Fire Prediction App")
st.markdown("Use this app to predict fire risk based on environmental data.")

# Create input fields
Temperature = st.number_input("Temperature", value=0.0)
RH = st.number_input("Relative Humidity (RH)", value=0.0)
Ws = st.number_input("Wind Speed (Ws)", value=0.0)
Rain = st.number_input("Rainfall (Rain)", value=0.0)
FFMC = st.number_input("FFMC", value=0.0)
DMC = st.number_input("DMC", value=0.0)
ISI = st.number_input("ISI", value=0.0)
Classes = st.number_input("Classes", value=0.0)
Region = st.number_input("Region", value=0.0)

# Prediction button
if st.button("Predict Fire Risk"):
    try:
        # Scale input data and predict
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)
        st.success(f"🔥 Predicted Fire Risk: {result[0]:.2f}")
    except Exception as e:
        st.error("⚠️ Please enter valid numeric values for all fields.")

