import streamlit as st
import pickle
import numpy as np


model = pickle.load(open('trained_pipeline-0.1.2.pkl','rb'))

def predict_heartdisease(ST_Slope_Flat, ST_Slope_Up, ST_Slope_Down, 
                         ChestPainType_ASY, ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA, 
                         FastingBS):

    input = np.array([[ST_Slope_Flat, ST_Slope_Up, ST_Slope_Down, 
                         ChestPainType_ASY, ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA, 
                         FastingBS]]).astype(np.bool_)
    
    prediction = model.predict(input)
    
    return bool(prediction)

def main():
    st.title("Heart Failure Classifier")
    """__Authors__: M. Matthes, Z. Daniali""" 
    """[contact](mmatthes@greenbootcamps.com) | [corresponding gitHub repo](https://github.com/marie-matthes/Heart-Disease-Classifier) \
    """
    """ __Goal__ :  As a health insurance company, you want to find out whether testing a patient for heart disease is necessary or not."""
    """ __Dataset__ : The Logistic Regression Model was trained on the [Heart Failure Prediction Dataset from Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)"""
    """* * *"""

    ST_Slope_option = st.selectbox(
        "Select slope of the peak exercise ST segment of the patient",
        ("Up", "Flat", "Down"))

    # st.write("You selected:", ST_Slope_option)

    if ST_Slope_option == "Up":
        ST_Slope_Up = 1
        ST_Slope_Flat = 0
        ST_Slope_Down = 0
    elif ST_Slope_option == "Flat":
        ST_Slope_Flat = 1
        ST_Slope_Up = 0
        ST_Slope_Down = 0
    else:
        ST_Slope_Down = 1
        ST_Slope_Up = 0
        ST_Slope_Flat = 0

    ChestPainType_option = st.selectbox(
        "Select Chest Pain Type of the patient",
        ("ASY", "ATA", "NAP", "TA"))

    # st.write("You selected:", ChestPainType_option)

    if ChestPainType_option == "ASY":
        ChestPainType_ASY = 1
        ChestPainType_ATA = 0
        ChestPainType_NAP = 0
        ChestPainType_TA = 0
    elif ChestPainType_option == "ATA":
        ChestPainType_ATA = 1
        ChestPainType_ASY = 0
        ChestPainType_NAP = 0
        ChestPainType_TA = 0
    elif ChestPainType_option == "NAP":
        ChestPainType_NAP = 1
        ChestPainType_ATA = 0
        ChestPainType_ASY = 0
        ChestPainType_TA = 0
    else:
        ChestPainType_TA = 1
        ChestPainType_ATA = 0
        ChestPainType_NAP = 0
        ChestPainType_ASY = 0

    FastingBS_option = st.selectbox(
        "Does the patient have a high Fasting Blood Sugar?",
        ("Yes", "No"))

    # st.write("You selected:", FastingBS_option)

    if FastingBS_option == "Yes":
        FastingBS = 1
    else:
        FastingBS = 0

    if st.button("Get Prediction"):
        output = predict_heartdisease(ST_Slope_Flat, ST_Slope_Up, ST_Slope_Down, 
                         ChestPainType_ASY, ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA, 
                         FastingBS)
        
        if output == True:
            st.error('Patient will likely develop Heart Disease (89% Accuracy)')
        else:
            st.success("Patient won't likely develop Heart Disease (80% Accuracy)")
        
        st.write("Overall Model Accuracy: 85%")

if __name__=='__main__':
    main()