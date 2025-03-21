import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load models
classification_model = joblib.load('model/final_combined_classification_model.pkl')
regression_model = joblib.load('model/final_combined_regression_model.pkl')

# Load dataset
dataset = pd.read_csv('data/final_combined_soil_dataset.csv')

# Streamlit Interface
st.title('Soil Analysis Prediction')

st.sidebar.header('User Input Parameters')

def user_input_features():
    NIR_Spectroscopy_900nm = st.sidebar.number_input('NIR_Spectroscopy_900nm')
    NIR_Spectroscopy_2500nm = st.sidebar.number_input('NIR_Spectroscopy_2500nm')
    Nutrient_Nitrogen_mg_kg = st.sidebar.number_input('Nutrient_Nitrogen_mg_kg')
    Nutrient_Phosphorus_mg_kg = st.sidebar.number_input('Nutrient_Phosphorus_mg_kg')
    Nutrient_Potassium_mg_kg = st.sidebar.number_input('Nutrient_Potassium_mg_kg')
    pH_Level = st.sidebar.number_input('pH_Level')
    Visible_Light_400nm = st.sidebar.number_input('Visible_Light_400nm')
    Visible_Light_700nm = st.sidebar.number_input('Visible_Light_700nm')
    Temperature_C = st.sidebar.number_input('Temperature_C')
    Moisture_Content_ = st.sidebar.number_input('Moisture_Content_%')
    Electrical_Conductivity_dS_m = st.sidebar.number_input('Electrical_Conductivity_dS_m')
    Organic_Matter_ = st.sidebar.number_input('Organic_Matter_%')
    GPS_Latitude = st.sidebar.number_input('GPS_Latitude')
    GPS_Longitude = st.sidebar.number_input('GPS_Longitude')
    Time_of_Measurement = st.sidebar.number_input('Time_of_Measurement')

    features = {
        'NIR_Spectroscopy_900nm': NIR_Spectroscopy_900nm,
        'NIR_Spectroscopy_2500nm': NIR_Spectroscopy_2500nm,
        'Nutrient_Nitrogen_mg_kg': Nutrient_Nitrogen_mg_kg,
        'Nutrient_Phosphorus_mg_kg': Nutrient_Phosphorus_mg_kg,
        'Nutrient_Potassium_mg_kg': Nutrient_Potassium_mg_kg,
        'pH_Level': pH_Level,
        'Visible_Light_400nm': Visible_Light_400nm,
        'Visible_Light_700nm': Visible_Light_700nm,
        'Temperature_C': Temperature_C,
        'Moisture_Content_%': Moisture_Content_,
        'Electrical_Conductivity_dS_m': Electrical_Conductivity_dS_m,
        'Organic_Matter_%': Organic_Matter_,
        'GPS_Latitude': GPS_Latitude,
        'GPS_Longitude': GPS_Longitude,
        'Time_of_Measurement': Time_of_Measurement
    }
    
    return pd.DataFrame(features, index=[0])

# User input features
input_data = user_input_features()

# Preprocess input data (standard scaling or any preprocessing as needed)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(input_data)

# Model predictions
classification_pred = classification_model.predict(scaled_data)
regression_pred = regression_model.predict(scaled_data)

# Display predictions
st.subheader('Classification Prediction (e.g., Soil Fertility Level)')
st.write(classification_pred)

st.subheader('Regression Predictions (e.g., Nutrient Levels, Organic Matter, etc.)')
st.write(f'Nitrogen: {regression_pred[0][0]} mg/kg')
st.write(f'Phosphorus: {regression_pred[0][1]} mg/kg')
st.write(f'Potassium: {regression_pred[0][2]} mg/kg')
st.write(f'Organic Matter: {regression_pred[0][3]}%')
st.write(f'Water Retention Capacity: {regression_pred[0][4]}')
st.write(f'Lime Requirement: {regression_pred[0][5]}')
st.write(f'Soil Erosion Risk: {regression_pred[0][6]}')
