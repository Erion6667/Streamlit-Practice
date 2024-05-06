from pycaret.classification import *
import streamlit as st
import pandas as pd
import numpy as np

#load pretrained model for classification
model = load_model('model')

#define classification function to call
def predict(model, input_df):
    predictions_df = predict_model(model, data = input_df)
    st.write(predictions_df)
    predictions= predictions_df['prediction_label'][0]
    return predictions

def run():
    #mengambil gambar yang ada difolder
    from PIL import Image
    image = Image.open('images/logo_sdt.png')
    image_diabetes = Image.open('images/OIP.jpeg')

    #add sidebar to the app
    st.sidebar.title('Praktikum Streamlit')
    st.sidebar.markdown("Aplikasi klasifikasi dari Sleep Health and Lifestyle Dataset bertujuan untuk memprediksi apakah seseorang memiliki gangguan tidur berdasarkan berbagai faktor terkait tidur dan kebiasaan harian mereka. Dengan menggunakan teknik klasifikasi, model akan mempelajari pola dari data yang mencakup variabel seperti jenis kelamin, usia, pekerjaan, durasi tidur, kualitas tidur, tingkat aktivitas fisik, tingkat stres, kategori BMI, tekanan darah, denyut jantung, jumlah langkah harian, dan keberadaan gangguan tidur.")
    st.sidebar.info("Aplikasi ini contoh praktikum streamlit pada mata kuliah MLOps")
    st.sidebar.success("By: Bayu Kurniawan / 3322600019")
    st.sidebar.image(image)
    
    #add title and subtitle to the main interface of the app
    st.image(image_diabetes)
    st.title("Klasifikasi Sleep Health and Lifestyle Circle")
    st.markdown("Dengan model klasifikasi ini, kita dapat mengidentifikasi pola-pola yang berkaitan dengan gangguan tidur dan menggunakan informasi tersebut untuk memberikan rekomendasi atau intervensi yang sesuai kepada individu yang berisiko mengalami gangguan tidur. Hal ini dapat membantu dalam pemantauan kesehatan tidur dan pencegahan gangguan tidur lebih lanjut.")
    
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
        if prediction[0] == 0:
            st.error('Terkena Gangguan Tidur : Insomnia')
        elif prediction[0] == 2:
            st.error('Terkena Gangguan Tidur : Sleep Apnea')
        else:
            st.success('Tidak Terkena Gangguan Tidur')  

    page_element="""
    <style>
    [data-testid="stAppViewContainer"]{
      background-image: url("https://p4.wallpaperbetter.com/wallpaper/385/220/840/space-clouds-aurorae-dark-wallpaper-preview.jpg");
      background-size: cover;
    }
    
    [data-testid="stSidebar"]> div:first-child{
        background-image: url("https://p4.wallpaperbetter.com/wallpaper/209/95/584/anime-sky-landscape-clouds-portrait-display-hd-wallpaper-preview.jpg");
        background-size: cover;
    }
    
    </style>
    """

    st.markdown(page_element, unsafe_allow_html=True)

   
if __name__ == '__main__':
    run()