# 33832025_Churning_Customers
Telecom Customer Churn Prediction Model by Leslie Konlack

Overview

In this project, my focus was on the critical issue of customer churn in the telecom industry.
I aimed to develop a deep learning-based predictive model to help telecom operators identify customers at high risk of churn. 
The model is designed to enable proactive strategies for improving customer retention.

Approach and Methodology

-Exploratory Data Analysis (EDA)

I conducted an in-depth EDA to uncover pivotal factors influencing customer churn. 
This involved analyzing various customer profiles and service usage patterns to pinpoint characteristics correlated with a higher likelihood of churn.

-Feature Engineering

Leveraging insights from EDA, I extracted essential features from the dataset that directly impact churn.
My approach was to ensure a robust feature set that encapsulates the multifaceted nature of customer interactions and behaviors.

-Deep Learning Model

Utilizing TensorFlow's Functional API, I designed a Multi-Layer Perceptron (MLP) model. 
This API provided the flexibility to experiment with complex network architectures, allowing me to fine-tune the model for optimal churn prediction.
I meticulously tuned the model parameters to balance accuracy and computational efficiency, ensuring a reliable and scalable solution.

-Model Evaluation

I adopted a thorough approach to evaluate the model's performance. 
This included not just assessing the accuracy but also calculating the AUC score to validate the model's proficiency in distinguishing potential churners.

-Application Development with Streamlit

To host the model, I developed an interactive web application using Streamlit. 
My focus was on creating an intuitive user interface that simplifies the process of inputting customer data and receiving predictions.
Streamlit's ability to seamlessly integrate with TensorFlow models made it an ideal choice for this application, ensuring smooth functionality and real-time interaction.

-Technologies and Tools

Python: The backbone for all scripting, data processing, and model development.
TensorFlow & Keras: Used for constructing, training, and optimizing the MLP model.
Streamlit: Chosen for its efficacy in deploying interactive data applications.

Running the Application

-Local Setup

Ensure all dependencies (Python, TensorFlow, Streamlit, etc.) are installed.
Download the project repository and navigate to the app's directory.
Execute streamlit run app.py in the terminal to start the application locally.

-Web Access

Access the application directly through Your Streamlit Web App Link, hosted online for ease of use. 
Streamlit Web App Link: https://33832025churningcustomers-d9sufr9fuxfpxsq5i9lkms.streamlit.app

How to Use the Application

Upon launching the application, you're greeted with a clean, straightforward interface.
Input fields are provided for each relevant customer feature. Fill in these fields based on the customer data you wish to analyze.
After inputting the data, click the 'Predict Churn' button.
The model processes the input and displays a churn prediction along with the confidence level of the prediction.
The interface is designed for clarity and ease of use, ensuring that users can effortlessly navigate and understand the results.

Dataset

The dataset used for this project provides a comprehensive view of customer behavior in the telecom industry. 
Dataset Link: https://drive.google.com/file/d/1deqC-VzcKNvTIrGcXO3nGEfJ5_Gzyl0S/view



