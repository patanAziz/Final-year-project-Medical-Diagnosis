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

parkinsons_model = pickle.load(open('saved_models/parkinsons_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
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

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        fo = st.number_input('Average Vocal Frequency (MDVP:Fo Hz)', value=150.0, step=0.1)
    with col2:
        fhi = st.number_input('Maximum Vocal Frequency (MDVP:Fhi Hz)', value=200.0, step=0.1)
    with col3:
        flo = st.number_input('Minimum Vocal Frequency (MDVP:Flo Hz)', value=100.0, step=0.1)
    with col4:
        jitter_percent = st.number_input('Jitter Percentage (MDVP:Jitter%)', value=0.003, step=0.0001)
    with col5:
        jitter_abs = st.number_input('Absolute Jitter (MDVP:Jitter(Abs))', value=0.0002, step=0.00001)
    
    with col1:
        rap = st.number_input('Relative Average Perturbation (RAP)', value=0.003, step=0.00001)
    with col2:
        ppq = st.number_input('5-Point Period Perturbation Quotient (PPQ)', value=0.003, step=0.00001)
    with col3:
        ddp = st.number_input('Delta-Delta Perturbation (Jitter:DDP)', value=0.003, step=0.00001)
    with col4:
        shimmer = st.number_input('Shimmer (MDVP:Shimmer)', value=0.03, step=0.0001)
    with col5:
        shimmer_db = st.number_input('Shimmer in dB (MDVP:Shimmer(dB))', value=0.2, step=0.01)
    
    with col1:
        apq3 = st.number_input('3-Point Amplitude Perturbation (Shimmer:APQ3)', value=0.02, step=0.0001)
    with col2:
        apq5 = st.number_input('5-Point Amplitude Perturbation (Shimmer:APQ5)', value=0.02, step=0.0001)
    with col3:
        apq = st.number_input('Amplitude Perturbation Quotient (MDVP:APQ)', value=0.02, step=0.0001)
    with col4:
        dda = st.number_input('Delta Delta Amplitude (Shimmer:DDA)', value=0.03, step=0.0001)
    with col5:
        nhr = st.number_input('Noise-to-Harmonics Ratio (NHR)', value=0.03, step=0.0001)
    
    with col1:
        hnr = st.number_input('Harmonics-to-Noise Ratio (HNR)', value=20.0, step=0.1)
    with col2:
        rpde = st.number_input('Recurrence Period Density Entropy (RPDE)', value=0.5, step=0.01)
    with col3:
        dfa = st.number_input('Detrended Fluctuation Analysis (DFA)', value=0.5, step=0.01)
    with col4:
        spread1 = st.number_input('Nonlinear Measure Spread1', value=-5.0, step=0.01)
    with col5:
        spread2 = st.number_input('Nonlinear Measure Spread2', value=2.0, step=0.01)
    
    with col1:
        d2 = st.number_input('Correlation Dimension (D2)', value=2.0, step=0.01)
    with col2:
        ppe = st.number_input('Pitch Period Entropy (PPE)', value=0.5, step=0.01)

    
    # Prediction
    parkinsons_diagnosis = ''
    
    if st.button("Parkinson's Test Result"):
        # Collect all inputs into a list
        user_input = [
            fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
            shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
            rpde, dfa, spread1, spread2, d2, ppe
        ]
    
        # Convert inputs to float (already numeric, so optional)
        user_input = [float(x) for x in user_input]
    
        # Make prediction
        parkinsons_prediction = parkinsons_model.predict([user_input])
    
        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
    
    st.success(parkinsons_diagnosis)
