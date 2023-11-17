import streamlit as st
import joblib
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from create_mlp_model import create_mlp_model
from joblib import load
from tensorflow import keras
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

model_path = 'new_leslie_modell.plk'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

scaler = joblib.load( 'scaler (1).joblib')

label_encoder = joblib.load('label_encoder.joblib')


# Streamlit app title
st.title('Customer Churn Prediction')

# Input fields
tenure = st.number_input('Tenure', min_value=0)
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, format='%f')
total_charges = st.number_input('Total Charges', min_value=0.0, format='%f')
gender = st.selectbox('Gender', ['Female', 'Male'])
senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
phone_service = st.selectbox('PhoneService', ['Yes', 'No'])
multiple_lines = st.selectbox('MultipleLines', ['Yes', 'No', 'No phone service'])
internet_service = st.selectbox('InternetService', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('OnlineSecurity', ['Yes', 'No', 'No internet service'])
online_backup = st.selectbox('OnlineBackup', ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox('DeviceProtection', ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox('TechSupport', ['Yes', 'No', 'No internet service'])
streaming_tv = st.selectbox('StreamingTV', ['Yes', 'No', 'No internet service']) # Added based on your dataset
streaming_movies = st.selectbox('StreamingMovies', ['Yes', 'No', 'No internet service'])
paperless_billing = st.selectbox('PaperlessBilling', ['Yes', 'No'])
payment_method = st.selectbox('PaymentMethod', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])

# Button to make prediction
if st.button('Predict Churn'):
    # Create a DataFrame from the inputs
    input_df = pd.DataFrame([[tenure, monthly_charges, total_charges, gender, senior_citizen, 
                              partner, dependents, phone_service, multiple_lines, internet_service, 
                              online_security, online_backup, device_protection, 
                              tech_support, streaming_tv, streaming_movies, paperless_billing, 
                              payment_method, contract]],
                            columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'SeniorCitizen',
                                     'Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService',
                                     'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                     'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod', 'Contract'])

    # Convert 'TotalCharges' to numeric
    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')

    # Map 'SeniorCitizen' from 'Yes/No' to 1/0
    input_df['SeniorCitizen'] = input_df['SeniorCitizen'].map({'Yes': 1, 'No': 0})

    # Apply label encoding to categorical variables
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']

    categorical_features_encoded = label_encoder.fit_transform(categorical_cols) 

    # One-hot encode 'PaymentMethod' and 'Contract'
    input_df = pd.get_dummies(input_df, columns=['PaymentMethod', 'Contract'])

    # Apply scaling
    input_data_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]

    # Display the prediction
    st.write(f'Churn Probability: {churn_probability}')
