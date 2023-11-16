import streamlit as st
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from create_mlp_model import create_mlp_model
from joblib import load
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Load the saved components
modelpath='model.joblib'
# Load the Keras model from a pickled file

model = joblib.load(modelpath)

from joblib import load
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
phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])

# Button to make prediction
if st.button('Predict Churn'):
    # Create a DataFrame from the inputs
    input_data = pd.DataFrame([[tenure, monthly_charges, total_charges, gender, senior_citizen, 
                                partner, dependents, phone_service, multiple_lines, internet_service, 
                                online_security, online_backup, device_protection, 
                                tech_support, streaming_tv, streaming_movies, paperless_billing, 
                                payment_method, contract]],
                              columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'SeniorCitizen',
                                       'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod', 'Contract'])

    # Process the inputs
    # Convert categorical variables using label encoding
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']

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
