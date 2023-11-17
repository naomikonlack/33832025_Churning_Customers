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
def main():
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
    
        # Apply label encoding to categorical variables
        categorical_cols = ['gender', 'Partner', 'Dependents', 'SeniorCitizen','PhoneService', 'MultipleLines',
                            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
    
        categorical_features_encoded = label_encoder.fit_transform(categorical_cols) 
        numerical_columns= {
                'PaymentMethod': [payment_method],
                'Contract': [contract]
            }
        df = pd.DataFrame(numerical_columns)
        # One-hot encode 'PaymentMethod' and 'Contract'
        input_df = pd.get_dummies(df, columns=['PaymentMethod', 'Contract'])
    
        flattened_encoded_df =input_df.values.flatten()
        input_features = np.concatenate([
                np.array([ tenure, monthly_charges, total_charges, *categorical_features_encoded]),
                flattened_encoded_df
            ]).reshape(1, -1)
    
            # Scale the input features
        input_features_scaled = scaler.transform(input_features)
    
            # Make predictions
        prediction = model.predict(input_features_scaled)
    
        label_mapping = {1: 'Yes', 0: 'No'}
    
        predicted_churn_label =int(prediction[0])
            # Map the predicted label using the dictionary
        predicted_churn = label_mapping[predicted_churn_label]
            # Display the prediction
        st.write(f"Predicted Churn: {predicted_churn}") 
       
        
if __name__ == "__main__":
    main()
