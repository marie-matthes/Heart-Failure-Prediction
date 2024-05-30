import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from streamlit_extras.add_vertical_space import add_vertical_space
import io

st.set_page_config(page_title="Heart Disease Classifier", page_icon="❤️", layout="wide")

model = pickle.load(open('trained_pipeline-0.1.2.pkl','rb'))
# df = pd.read_csv('./data/918_heart_failure_dataset.csv')

def predict_heartdisease(ST_Slope_Flat, ST_Slope_Up, ST_Slope_Down, 
                         ChestPainType_ASY, ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA, 
                         FastingBS):

    input = np.array([[ST_Slope_Flat, ST_Slope_Up, ST_Slope_Down, 
                         ChestPainType_ASY, ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA, 
                         FastingBS]]).astype(np.bool_)
    
    prediction = model.predict(input)
    
    return bool(prediction)

def main():
    st.sidebar.title("Heart Failure Classifier")
    st.sidebar.write("__Authors__: M. Matthes, Z. Daniali  \n[contact](mmatthes@greenbootcamps.com) | [corresponding GitHub repo](https://github.com/marie-matthes/Heart-Failure-Prediction)")
    st.sidebar.write("__Is your patient likely to develop a Heart Disease?__ As a health insurance company, you want to find out whether testing a patient for heart disease is necessary or not.")
    st.sidebar.write("__Dataset__ : The Logistic Regression Model was trained on the [Heart Failure Prediction Dataset from Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)")

    # Sidebar for prediction
    
    ST_Slope_option = st.sidebar.selectbox("**ST Slope**  \nSelect slope of the peak exercise ST segment of the patient", 
                                           ("Up", "Flat", "Down"))
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
    
    ChestPainType_option = st.sidebar.selectbox(
        "**Chest Pain Type**  \nSelect Chest Pain Type of the patient",
        ("ASY", "ATA", "NAP", "TA"))

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

    FastingBS_option = st.sidebar.selectbox(
        "**Fasting Blood Sugar**  \nDoes the patient have a high Fasting Blood Sugar?",
        ("Yes", "No"))

    if FastingBS_option == "Yes":
        FastingBS = 1
    else:
        FastingBS = 0
    
    # Get prediction
    if st.sidebar.button("Get Prediction"):
        output = predict_heartdisease(ST_Slope_Flat, ST_Slope_Up, ST_Slope_Down, 
                         ChestPainType_ASY, ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA, 
                         FastingBS)
        
        if output == True:
            st.sidebar.error('Patient will likely develop Heart Disease (89% Accuracy)')
        else:
            st.sidebar.success("Patient won't likely develop Heart Disease (80% Accuracy)")
        
        st.sidebar.write("Overall Model Accuracy: 85%")

    # statistics
    st.title("Heart Disease Statistics")
    st.image('./images/header.jpg')
    st.write("Please upload your current healthcare data to receive an overview of the distribution of the metrics utilized in the predictive analysis.")
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns(
            (0.1, 1, 0.1, 1, 0.1)
        )

        with row1_1:
            heart_disease_counts = df['HeartDisease'].value_counts().reset_index()
            heart_disease_counts.columns = ['Heart Disease', 'Count'] 

            st.subheader("Distribution Heart Disease")
            fig = px.pie(
                heart_disease_counts, 
                names='Heart Disease',
                color_discrete_sequence=['#9EE6CF', '#4D7267'], 
                values='Count'
            )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            st.markdown(
                f"The plot above shows the distribution of Heart Disease observations in your data set."
            )
            st.markdown("0 indicates that a patient did not develop a Heart Disease, 1 indicates the person did develop a Heart Disease.") 
            
        with row1_2:
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
                f"The plot above shows the distribution of ST Slope in your data set."
            )
            st.markdown("Pattern of ST-segment changes during peak exercise compared to baseline: Up, Flat or Down.")

        add_vertical_space()
        row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns(
            (0.1, 1, 0.1, 1, 0.1)
        )

        with row2_1:
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
                f"The plot above shows the distribution of Chestpain Type in the dataset."
            )
            st.markdown("The four categories are: TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic.")

        with row2_2:
            st.subheader("Fasting Blood Sugar")
            fig = px.histogram(
                df,
                x='FastingBS',
                color='HeartDisease',
                color_discrete_sequence=['#9EE6CF', '#4D7267'],
                barmode='group'
            )
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
            st.markdown(
                f"The plot above shows the distribution of Fasting Blood Sugar."
            )
            st.markdown("The two categories are: 1: if FastingBS > 120 mg/dl, 0: otherwise.")
            

if __name__=='__main__':
    main()
