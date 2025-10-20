# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open('saved_models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('saved_models/heart_disease_model.sav', 'rb'))

breast_cancer_model = pickle.load(open('breast_cancer_model.sav', 'rb'))

scaler = pickle.load(open('saved_models/breast_cancer_scaler.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Breast Cancer Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0, step=1)
    with col2:
        glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=120, step=1)
    with col3:
        blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=200, value=70, step=1)
    
    with col1:
        skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20, step=1)
    with col2:
        insulin = st.number_input('Insulin Level (µU/mL)', min_value=0, max_value=1000, value=80, step=1)
    with col3:
        bmi = st.number_input('BMI (kg/m²)', min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    
    with col1:
        diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    with col2:
        age = st.number_input('Age of the Person', min_value=1, max_value=120, value=30, step=1)



    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                      bmi, diabetes_pedigree_function, age
                      ]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=25)
    
    with col2:
        sex = st.selectbox('Sex', options=[0,1], format_func=lambda x: 'Female' if x==0 else 'Male')
    
    with col3:
        cp = st.selectbox('Chest Pain types', options=[0,1,2,3], 
                          format_func=lambda x: {0:'Typical Angina', 1:'Atypical', 2:'Non-anginal', 3:'Asymptomatic'}[x])
    
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=250, value=120)
    
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=100, max_value=600, value=200)
    
    with col3:
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0,1])
    
    with col1:
        restecg = st.selectbox('Resting Electrocardiographic results', options=[0,1,2])
    
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', min_value=60, max_value=220, value=150)
    
    with col3:
        exang = st.selectbox('Exercise Induced Angina', options=[0,1])
    
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    with col2:
        slope = st.selectbox('Slope of the peak exercise ST segment', options=[0,1,2])
    
    with col3:
        ca = st.selectbox('Major vessels colored by flourosopy', options=[0,1,2,3])
    
    with col1:
        thal = st.selectbox('thal: 1 = normal; 2 = fixed defect; 3 = reversable defect', options=[1,2,3])

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)
# ==============================
# 🧬 Breast Cancer Prediction
# ==============================
if selected == "Breast Cancer Prediction":
    st.title("🩺 Breast Cancer Prediction using Machine Learning")

    # --- Row 1 ---
    col1, col2, col3, col4, col5 = st.columns(5, gap="medium")

    # --- Row 1 ---
    with col1:
        radius_mean = st.number_input('Mean Radius', min_value=0.0, value=20.0, format="%.3f")
    with col2:
        texture_mean = st.number_input('Mean Texture', min_value=0.0, value=25.0, format="%.3f")
    with col3:
        perimeter_mean = st.number_input('Mean Perimeter', min_value=0.0, value=130.0, format="%.3f")
    with col4:
        area_mean = st.number_input('Mean Area', min_value=0.0, value=1000.0, format="%.3f")
    with col5:
        smoothness_mean = st.number_input('Mean Smoothness', min_value=0.0, value=0.1, format="%.3f")
    
    # --- Row 2 ---
    with col1:
        compactness_mean = st.number_input('Mean Compactness', min_value=0.0, value=0.2, format="%.3f")
    with col2:
        concavity_mean = st.number_input('Mean Concavity', min_value=0.0, value=0.3, format="%.3f")
    with col3:
        concave_points_mean = st.number_input('Mean Concave Points', min_value=0.0, value=0.15, format="%.3f")
    with col4:
        symmetry_mean = st.number_input('Mean Symmetry', min_value=0.0, value=0.2, format="%.3f")
    with col5:
        fractal_dimension_mean = st.number_input('Mean Fractal Dimension', min_value=0.0, value=0.06, format="%.3f")
    
    # --- Row 3 ---
    with col1:
        radius_se = st.number_input('Radius SE', min_value=0.0, value=1.0, format="%.3f")
    with col2:
        texture_se = st.number_input('Texture SE', min_value=0.0, value=1.5, format="%.3f")
    with col3:
        perimeter_se = st.number_input('Perimeter SE', min_value=0.0, value=2.0, format="%.3f")
    with col4:
        area_se = st.number_input('Area SE', min_value=0.0, value=20.0, format="%.3f")
    with col5:
        smoothness_se = st.number_input('Smoothness SE', min_value=0.0, value=0.02, format="%.3f")
    
    # --- Row 4 ---
    with col1:
        compactness_se = st.number_input('Compactness SE', min_value=0.0, value=0.03, format="%.3f")
    with col2:
        concavity_se = st.number_input('Concavity SE', min_value=0.0, value=0.04, format="%.3f")
    with col3:
        concave_points_se = st.number_input('Concave Points SE', min_value=0.0, value=0.02, format="%.3f")
    with col4:
        symmetry_se = st.number_input('Symmetry SE', min_value=0.0, value=0.02, format="%.3f")
    with col5:
        fractal_dimension_se = st.number_input('Fractal Dimension SE', min_value=0.0, value=0.002, format="%.3f")
    
    # --- Row 5 ---
    with col1:
        radius_worst = st.number_input('Worst Radius', min_value=0.0, value=25.0, format="%.3f")
    with col2:
        texture_worst = st.number_input('Worst Texture', min_value=0.0, value=30.0, format="%.3f")
    with col3:
        perimeter_worst = st.number_input('Worst Perimeter', min_value=0.0, value=170.0, format="%.3f")
    with col4:
        area_worst = st.number_input('Worst Area', min_value=0.0, value=1500.0, format="%.3f")
    with col5:
        smoothness_worst = st.number_input('Worst Smoothness', min_value=0.0, value=0.15, format="%.3f")
    
    # --- Row 6 ---
    with col1:
        compactness_worst = st.number_input('Worst Compactness', min_value=0.0, value=0.3, format="%.3f")
    with col2:
        concavity_worst = st.number_input('Worst Concavity', min_value=0.0, value=0.4, format="%.3f")
    with col3:
        concave_points_worst = st.number_input('Worst Concave Points', min_value=0.0, value=0.2, format="%.3f")
    with col4:
        symmetry_worst = st.number_input('Worst Symmetry', min_value=0.0, value=0.3, format="%.3f")
    with col5:
        fractal_dimension_worst = st.number_input('Worst Fractal Dimension', min_value=0.0, value=0.08, format="%.3f")

    # Collect all features in correct order
    input_data = [
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
        compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
        radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
        compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]

    # Scale and predict
    breast_input_scaled = scaler.transform([input_data])
    breast_prediction = breast_cancer_model.predict(breast_input_scaled)

    if st.button("🔍 Breast Cancer Test Result"):
        if breast_prediction[0] == 1:
            st.error("⚠️ The model predicts: The person **has Breast Cancer (Malignant)**.")
        else:
            st.success("✅ The model predicts: The person **does not have Breast Cancer (Benign)**.")
