# app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_pipeline = joblib.load('heart_disease_model.pkl')

# Function to display prediction result with styled output
def display_prediction_result(prediction):
    if prediction[0] == 1:
        st.error("The model predicts that the patient is likely to have heart disease.")
    else:
        st.success("The model predicts that the patient is unlikely to have heart disease.")

# Streamlit app
st.title("Heart Disease Prediction")

# Input fields
age = st.slider("Age", min_value=20, max_value=90, value=50)
sex = st.selectbox("Sex", ['Male', 'Female'])
cp = st.selectbox("Chest Pain Type", ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
chol = st.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['Yes', 'No'])
restecg = st.selectbox("Resting ECG Results", ['normal', 'ST-T abnormality', 'LV hypertrophy'])
thalch = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", ['Yes', 'No'])
oldpeak = st.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", ['upsloping', 'flat', 'downsloping'])
ca = st.select_slider("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
thal = st.selectbox("Thal", ['normal', 'fixed defect', 'reversible defect'])

# Make prediction when button is clicked
if st.button("Predict"):
    input_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalch': thalch,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    # Convert the input dictionary to a dataframe
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model_pipeline.predict(input_df)

    # Display prediction result
    display_prediction_result(prediction)