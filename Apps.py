from pycaret.classification import *
import streamlit as st
import pandas as pd
import numpy as np

#load pretrained model for classification
model = load_model('model')

# Gender
gender_options = ['Female', 'Male']
gender_index = 1  # default to Male
gender = st.selectbox('Gender', gender_options, index=gender_index)

# Age
age = st.number_input('Age', min_value=0, max_value=200, value=27)

# Occupation
occupation_options = ['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher', 'Nurse']
occupation_index = 0  # default to Software Engineer
occupation = st.selectbox('Occupation', occupation_options, index=occupation_index)

# Sleep Duration
sleep_duration = st.number_input('Sleep Duration (hours)', min_value=0.0, max_value=24.0, value=8.0,  step=0.1)

# Quality of Sleep
quality_of_sleep = st.number_input('Quality of Sleep (0-10)', min_value=0, max_value=10, value=6)

# Physical Activity Level
physical_activity_level = st.number_input('Physical Activity Level (0-100)', min_value=0, max_value=100, value=50)

# BMI Category
bmi_category_options = ['Normal Weight', 'Normal', 'Overweight', 'Obese']
bmi_category_index = 1  # default to Normal
bmi_category = st.selectbox('BMI Category', bmi_category_options, index=bmi_category_index)

# Stress Level
stress_level = st.number_input('Stress Level (0-10)', min_value=0, max_value=10, value=6)

# Heart Rate
heart_rate = st.number_input('Heart Rate (bpm)', min_value=0, max_value=100, value=77)

# Daily Steps
daily_steps = st.number_input('Daily Steps', min_value=0, max_value=20000, value=4200)

# Mapping the input values to the model's expected format
occupation_mapping = {
    'Software Engineer': 0.0,
    'Doctor': 0.0,
    'Sales Representative': 0.0,
    'Teacher': 0.0,
    'Nurse': 0.0
}
occupation_mapping[occupation] = 1.0  # set the selected occupation to 1.0

bmi_category_mapping = {
    'Normal Weight': 0.0,
    'Normal': 0.0,
    'Overweight': 0.0,
    'Obese': 0.0
}
bmi_category_mapping[bmi_category] = 1.0  # set the selected BMI category to 1.0

# Prepare the input data
data = {
    'Gender': [1.0 if gender == 'Male' else 0.0],
    'Age': [age],
    'Occupation_Engineer': [occupation_mapping['Software Engineer']],
    'Occupation_Nurse': [occupation_mapping['Nurse']],
    'Occupation_Lawyer': [occupation_mapping['Doctor']],  # Assuming Lawyer is the same as Doctor
    'Occupation_Salesperson': [occupation_mapping['Sales Representative']],
    'Occupation_Teacher': [occupation_mapping['Teacher']],
    'Occupation_Manager': [occupation_mapping['Doctor']],  # Assuming Manager is the same as Doctor
    'Occupation_Software Engineer': [occupation_mapping['Software Engineer']],
    'Occupation_Doctor': [occupation_mapping['Doctor']],
    'Occupation_Accountant': [occupation_mapping['Doctor']],  # Assuming Accountant is the same as Doctor
    'Occupation_Scientist': [occupation_mapping['Doctor']],  # Assuming Scientist is the same as Doctor
    'Occupation_Sales Representative': [occupation_mapping['Sales Representative']],
    'Sleep Duration': [sleep_duration],
    'Quality of Sleep': [quality_of_sleep],
    'Physical Activity Level': [physical_activity_level],
    'Stress Level': [stress_level],
    'BMI Category_Normal': [bmi_category_mapping['Normal']],
    'BMI Category_Overweight': [bmi_category_mapping['Overweight']],
    'BMI Category_Normal Weight': [bmi_category_mapping['Normal Weight']],
    'BMI Category_Obese': [bmi_category_mapping['Obese']],
    'Heart Rate': [heart_rate],
    'Daily Steps': [daily_steps]
}

# Predict button
if st.button('Predict'):
    # Perform prediction using the model
    prediction = model.predict(pd.DataFrame(data))
    st.write('Prediction:', prediction)
