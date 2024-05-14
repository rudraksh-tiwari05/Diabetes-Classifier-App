import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Separate features and target variable
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the features
scaler = StandardScaler()
standardized_data = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(standardized_data, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Streamlit UI
st.title('Diabetes Classifier')

# Input form for user to enter new data
st.header('Enter New Data')
pregnancies = st.number_input('Pregnancies')
glucose = st.number_input('Glucose')
blood_pressure = st.number_input('Blood Pressure')
skin_thickness = st.number_input('Skin Thickness')
insulin = st.number_input('Insulin')
bmi = st.number_input('BMI')
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function')
age = st.number_input('Age')

# Predict on new data
input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
std_data = scaler.transform(input_data)
prediction = classifier.predict(std_data)

# Display prediction
st.subheader('Prediction')
if prediction[0] == 0:
    st.write('The person is not diabetic.')
else:
    st.write('The person is diabetic.')
