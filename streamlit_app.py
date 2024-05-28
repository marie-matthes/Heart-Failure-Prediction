import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px



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
    """[contact](mmatthes@greenbootcamps.com) | [corresponding GitHub repo](https://github.com/marie-matthes/Heart-Failure-Classifier) \
    """
    """ __Goal__ :  As a health insurance company, you want to find out whether testing a patient for heart disease is necessary or not."""
    """ __Dataset__ : The Logistic Regression Model was trained on the [Heart Failure Prediction Dataset from Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)"""
    """* * *"""

    # Sidebar with input field descriptions
    st.sidebar.header("Description of required input fields")
    st.sidebar.markdown("**ST Slope**: Pattern of ST-segment changes during peak exercise compared to baseline: Up, Flat or Down")
    st.sidebar.markdown("**Chest Pain Type**: TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic")
    st.sidebar.markdown("**Fasting Blood Sugar**: 1: if FastingBS > 120 mg/dl, 0: otherwise")

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
    
    st.write("")
    row_space1, row_1, row_space2, row_2, row_space3 = st.columns(
        (0.1, 1, 0.1, 1, 0.1)
    )
    
    df = pd.read_csv('./data/918_heart_failure_dataset.csv')

    with row_1:
        st.subheader("ST Slope")
        fig = px.histogram(
            df,
            x='ST_Slope',
            color='HeartDisease',
            color_discrete_sequence=['#9EE6CF', '#4D7267'],
            barmode='group'
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        st.markdown(
            f"The plot above shows the distribution of ST Slope in the dataset that was used for training the model"
        )


    with row_2:
        st.subheader("Chest Pain Type")
        fig = px.histogram(
            df,
            x='ChestPainType',
            color='HeartDisease',
            color_discrete_sequence=['#9EE6CF', '#4D7267'],
            barmode='group'
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        st.markdown(
            f"The plot above shows the distribution of Chestpain Type in the dataset that was used for training the model"
        )

if __name__=='__main__':
    main()