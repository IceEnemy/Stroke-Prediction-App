import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Function to remove outliers using IQR
def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[mask]

# Load data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df = df.drop(columns=['id'])
df = df[df['gender'] != 'Other']
df = df[df['work_type'] != 'Never_worked']
df_numeric = df[['age','avg_glucose_level','bmi','stroke']]

# Separate DataFrames for each 'stroke' group
df_stroke_0 = df_numeric[df_numeric['stroke'] == 0]
df_stroke_1 = df_numeric[df_numeric['stroke'] == 1]

# Remove outliers within each group
df_stroke_0_no_outliers = remove_outliers_iqr(df_stroke_0)
df_stroke_1_no_outliers = remove_outliers_iqr(df_stroke_1)

# Combine the DataFrames back together
df_no_outliers = pd.concat([df_stroke_0_no_outliers, df_stroke_1_no_outliers])  
df_combined = df.merge(df_no_outliers, on=['age', 'avg_glucose_level', 'bmi', 'stroke'], how='inner')
df = df_combined

# Label encoding
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['ever_married'] = le.fit_transform(df['ever_married'])
df['Residence_type'] = le.fit_transform(df['Residence_type'])
df = pd.get_dummies(df, columns=['work_type','smoking_status'], dtype=int)

# Scaling
scaler = MinMaxScaler()
df[['age','avg_glucose_level','bmi']] = scaler.fit_transform(df[['age','avg_glucose_level','bmi']])

# Impute missing BMI using linear regression
train_data = df[df['bmi'].notnull()]  
test_data = df[df['bmi'].isnull()]
X_train_lr = train_data.drop(columns=['bmi','stroke'])
y_train_lr = train_data['bmi']
X_test_lr = test_data.drop(columns=['bmi','stroke'])
model_lr = LinearRegression()
model_lr.fit(X_train_lr, y_train_lr)
y_pred_lr = model_lr.predict(X_test_lr)
df.loc[df['bmi'].isnull(), 'bmi'] = y_pred_lr

# Prepare data for logistic regression
X = df.drop(columns=['stroke'])
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train logistic regression model
classifier = LogisticRegression()
classifier.fit(X_train_res, y_train_res)

# Define the input form
def user_input_features():
    gender = st.selectbox('Gender', ('Male', 'Female'))
    age = st.slider('Age', 0, 100, 50)
    hypertension = st.selectbox('Hypertension', (0, 1))
    heart_disease = st.selectbox('Heart Disease', (0, 1))
    ever_married = st.selectbox('Ever Married', ('Yes', 'No'))
    work_type = st.selectbox('Work Type', ('Private', 'Self-employed', 'Govt_job', 'children'))
    residence_type = st.selectbox('Residence Type', ('Urban', 'Rural'))
    avg_glucose_level = st.slider('Average Glucose Level', 0.0, 300.0, 100.0)
    smoking_status = st.selectbox('Smoking Status', ('formerly smoked', 'never smoked', 'smokes'))
    bmi = st.slider('BMI', 0.0, 60.0, 25.0)

    data = {'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'smoking_status': smoking_status,
            'bmi': bmi}
    features = pd.DataFrame(data, index=[0])
    return features

# User input
input_df = user_input_features()

# Encode and scale input data
input_df['gender'] = le.transform(input_df['gender'])
input_df['ever_married'] = le.transform(input_df['ever_married'])
input_df['Residence_type'] = le.transform(input_df['Residence_type'])
input_df = pd.get_dummies(input_df, columns=['work_type','smoking_status'], dtype=int)

# Align the input_df to have the same columns as the training data
required_columns = X_train.columns  # Align columns
input_df = input_df.reindex(columns=required_columns, fill_value=0)

# Scale the input data
input_df[['age', 'avg_glucose_level', 'bmi']] = scaler.transform(input_df[['age', 'avg_glucose_level', 'bmi']])

# Prediction
prediction = classifier.predict(input_df)
prediction_proba = classifier.predict_proba(input_df)

st.subheader('Prediction')
stroke = np.array(['No Stroke', 'Stroke'])
st.write(stroke[prediction][0])

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Main function to run the Streamlit app
if __name__ == '__main__':
    st.title('Stroke Prediction App')
    st.write('This app predicts the likelihood of a stroke based on user input.')
