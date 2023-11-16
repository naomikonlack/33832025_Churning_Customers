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

model_path = 'new_leslie_model.plk'
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
gender = st.selectbox('Gender', ['Male', 'Female'])
senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
phone_service = st.selectbox('PhoneService', ['Yes', 'No'])
multiple_lines = st.selectbox('MultipleLines', ['Yes', 'No', 'No phone service'])
internet_service = st.selectbox('InternetService', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('OnlineSecurity', ['Yes', 'No', 'No internet service'])
online_backup = st.selectbox('OnlineBackup', ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox('DeviceProtection', ['Yes', 'No', 'No internet service'])
TechSupport = st.selectbox('TechSupport', ['Yes', 'No', 'No internet service'])
StreamingMovies = st.selectbox('StreamingMovies', ['Yes', 'No', 'No internet service'])
PaperlessBilling = st.selectbox('PaperlessBilling', ['Yes', 'No'])
payment_method_2 = st.selectbox('PaymentMethod', ['Electronic check'])
contract_0 = st.selectbox('Contract', ['Month-to-month'])
contract_2 = st.selectbox('Contract', [ 'Two year'])
# Button to make prediction
if st.button('Predict Churn'):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame([[tenure, monthly_charges, total_charges, gender, senior_citizen, 
                                partner, dependents, phone_service, multiple_lines, internet_service, 
                                online_security, online_backup, device_protection, 
                                TechSupport, StreamingMovies, PaperlessBilling, 
                                payment_method_2, contract_0,contract_2]],
                              columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'SeniorCitizen',
                                       'Partner', 'Dependents', 'MultipleLines', 'InternetService',
                                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod_2', 'Contract_0',
       'Contract_2'])
    # Process the inputs

    
    # Convert categorical variables using label encoding
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling','PaymentMethod_2', 'Contract_0',
       'Contract_2']
    for col in categorical_cols:
        input_data[col] = label_encoder.transform(input_data[col])

    # One-hot encode 'PaymentMethod' and 'Contract'
    input_data = pd.get_dummies(input_data, columns=['PaymentMethod', 'Contract'])

    # Scaling
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]

    # Display the prediction
    st.write(f'Churn Probability: {churn_probability}')
