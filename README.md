# Heart Failure Analysis & Prediction
Greenbootcamps Final Project

## About the Data Set

In today's fast-paced healthcare environment, health insurance companies need powerful tools to assess risk, make informed decisions, and engage with their customers. Our machine learning project provides an innovative solution that compares the performance of leading classification algorithms, including Logistic Regression, Support Vector Machines (SVM), Random Forest, and XGBoost, to predict whether a patient is likely to develop heart disease.

Join us as we leverage the power of machine learning to transform health insurance. Our project is not just about technology—it's about creating a healthier future for everyone.

We are working with the [Heart Failure Prediction Data Set from Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data)

| #  	| Name           	| Description                                	| Data Type                                                                                                                                                                                            	|
|----	|----------------	|--------------------------------------------	|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| 0  	| Age            	| age of the patient                         	| years (int)                                                                                                                                                                                                      	|
| 1  	| Sex            	| sex of the patient                         	| M: Male  <br>F: Female                                                                                                                                                                                       	|
| 2  	| ChestPainType  	| chest pain type                            	| TA: Typical Angina <br>ATA: Atypical Angina <br>NAP: Non-Anginal Pain <br>ASY: Asymptomatic                                                                                                                  	|
| 3  	| RestingBP      	| resting blood pressure                     	| mm Hg (int)                                                                                                                                                                                                      	|
| 4  	| Cholesterol    	| serum cholesterol                          	| mm/dl (int)                                                                                                                                                                                                      	|
| 5  	| FastingBS      	| fasting blood sugar                        	| 1: if FastingBS > 120 mg/dl <br>0: otherwise                                                                                                                                                                         	|
| 6  	| RestingECG     	| resting electrocardiogram results          	| Normal: Normal <br>ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) <br>LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria 	|
| 7  	| MaxHR          	| maximum heart rate achieved                	| Numeric value between 60 and 202 (int)                                                                                                                                                                           	|
| 8  	| ExerciseAngina 	| exercise-induced angina                    	| Y: Yes <br>N: No                                                                                                                                                                                             	|
| 9 	| Oldpeak        	| oldpeak = ST                               	| Numeric value measured in depression (float)                                                                                                                                                                     	|
| 10 	| ST_Slope       	| the slope of the peak exercise ST segment  	| Up: upsloping <br>Flat: flat Down: downsloping                                                                                                                                                           	|
| 11 	| HeartDisease   	| output class                               	| 1: heart disease <br>0: Normal                                                                                                                                                                                       	|

## Settings
````
pyenv local 3.12.2

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt 
````

## Further Resources:
* [Slides: Final Presenatation (PDF)](https://drive.google.com/file/d/1qQD9iv8uvoN-lx0Tpivh1KybbYwhJJgT/view?usp=sharing)
* [Streamlit App: Heart Failure Classifier](https://please-dont-go-breaking-my-heart.streamlit.app/)
* [Tableau Dashboard: Heart Failure Dataset](https://public.tableau.com/app/profile/zahra.daniali/viz/HeartAttackDashboard_17163037134360/HeartFailureDashboard?publish=yes)
