import streamlit as st
import numpy as np
import joblib

# Load the trained model and the scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Create the input form
st.title('Stroke Prediction')
st.write('Fill in the details below to predict the likelihood of stroke.')

age = st.number_input('Age', min_value=0, max_value=100, value=50)
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, step=0.1, value=100.0)
bmi = st.number_input('BMI', min_value=0.0, step=0.1, value=25.0)
gender = st.selectbox('Gender', options=['Female', 'Male'])
ever_married = st.selectbox('Ever Married', options=['No', 'Yes'])
Residence_type = st.selectbox('Residence Type', options=['Rural', 'Urban'])
hypertension = st.selectbox('Hypertension', options=['No', 'Yes'])
heart_disease = st.selectbox('Heart Disease', options=['No', 'Yes'])
work_type = st.selectbox('Work Type', options=['Govt_job', 'Private', 'Self-employed', 'children'])
smoking_status = st.selectbox('Smoking Status', options=['Unknown', 'formerly smoked', 'never smoked', 'smokes'])

# Convert categorical inputs to numerical values
gender = 1 if gender == 'Male' else 0
ever_married = 1 if ever_married == 'Yes' else 0
Residence_type = 1 if Residence_type == 'Urban' else 0
hypertension = 1 if hypertension == 'Yes' else 0
heart_disease = 1 if heart_disease == 'Yes' else 0

work_type_mapping = {
    'Govt_job': [1, 0, 0, 0],
    'Private': [0, 1, 0, 0],
    'Self-employed': [0, 0, 1, 0],
    'children': [0, 0, 0, 1]
}

smoking_status_mapping = {
    'Unknown': [1, 0, 0, 0],
    'formerly smoked': [0, 1, 0, 0],
    'never smoked': [0, 0, 1, 0],
    'smokes': [0, 0, 0, 1]
}

work_type_values = work_type_mapping[work_type]
smoking_status_values = smoking_status_mapping[smoking_status]

# Apply MinMax scaling to the input features
input_data = np.array([[age, avg_glucose_level, bmi]])
scaled_data = scaler.transform(input_data)

# Combine scaled data with other features
input_features = np.array([[
    gender,
    scaled_data[0][0],  # Scaled age
    hypertension,
    heart_disease,
    ever_married,
    Residence_type,
    scaled_data[0][1],  # Scaled avg_glucose_level
    scaled_data[0][2],  # Scaled bmi
    *work_type_values,
    *smoking_status_values
]])

# Predict stroke
prediction = model.predict(input_features)

# Display prediction
if st.button('Predict'):
    result = 'Stroke' if prediction[0] == 1 else 'No Stroke'
    st.write(f'Prediction: {result}')
