import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model and label encoders
model = joblib.load("/Users/abhishekkanade/Documents/notebook/My_Project/AICTE_IBM_Internshipe(Edunet)/salary_prediction_model.pkl")
encoders = joblib.load("/Users/abhishekkanade/Documents/notebook/My_Project/AICTE_IBM_Internshipe(Edunet)/label_encoders.pkl")  # Dict of LabelEncoders per column

# Page config
st.set_page_config(page_title="Employee Income Prediction", layout="centered")

# App title and intro
st.title("💼 Employee Income Prediction")
st.markdown("Predict whether a person's income exceeds $50K/year based on demographic information.")

# Sidebar Inputs
st.sidebar.header("🔧 Input Features")
gender = st.sidebar.selectbox("Gender", encoders['gender'].classes_)
age = st.sidebar.slider("Age", 17, 75, 30)
fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=1000, max_value=1000000, value=100000, step=1000)
workclass = st.selectbox("Workclass", encoders['workclass'].classes_)
education_num = st.sidebar.slider("Education Level (Numerical)", 1, 16, 10)
marital_status = st.selectbox("Marital Status", encoders['marital-status'].classes_)
occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
relationship = st.selectbox("Relationship", encoders['relationship'].classes_)
race = st.selectbox("Race", encoders['race'].classes_)
hours_per_week = st.sidebar.slider("Hours Worked per Week", 1, 80, 40)
native_country = st.selectbox("Native Country", encoders['native-country'].classes_)

# Construct input DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [education_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# Encode categorical columns
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship',
                    'race', 'gender', 'native-country']

for col in categorical_cols:
    le = encoders[col]
    input_data[col] = le.transform(input_data[col])

# Prediction section
st.subheader("📊 Prediction")
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "Income >50K" if prediction[0] == 1 else "Income <=50K"
    st.success(f"Prediction: **{result}**")

