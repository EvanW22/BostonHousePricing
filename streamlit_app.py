import pickle
import numpy as np
import streamlit as st

#load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

st.title("Boston House Price Prediction (1970s)")

#create input fields using Streamlit widgets
CRIM = st.number_input("Per Capita Crime Rate by Town (CRIM)", min_value=0.0, step=0.1)
ZN = st.number_input("Proportion of Residential Land Zoned for Lots Over 25,000 sq.ft. (ZN)", min_value=0.0, step=1.0)
INDUS = st.number_input("Proportion of Non-Retail Business Acres per Town (INDUS)", min_value=0.0, step=1.0)
CHAS = st.selectbox("Charles River Dummy Variable (=1 if tract bounds river; 0 otherwise) (CHAS)", options=[0, 1])
NOX = st.number_input("Nitric Oxides Concentration (NOX) in parts per 10 million", min_value=0.0, step=0.01)
RM = st.number_input("Average Number of Rooms per Dwelling (RM)", min_value=0.0, step=0.1)
Age = st.number_input("Proportion of Owner-Occupied Units Built Prior to 1940 (AGE)", min_value=0.0, step=0.1)
DIS = st.number_input("Weighted Distances to Five Boston Employment Centres (DIS)", min_value=0.0, step=0.1)
RAD = st.number_input("Index of Accessibility to Radial Highways (RAD)", min_value=0, step=1)
TAX = st.number_input("Full-Value Property-Tax Rate per $10,000 (TAX)", min_value=0, step=1)
PTRATIO = st.number_input("Pupil-Teacher Ratio by Town (PTRATIO)", min_value=0.0, step=0.1)
B = st.number_input("1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town (B)", min_value=0.0, step=1.0)
LSTAT = st.number_input("Percentage of Lower Status of the Population (LSTAT)", min_value=0.0, step=0.1)

input_data = [
    CRIM, ZN, INDUS, CHAS, NOX, RM, Age, DIS, RAD, TAX, PTRATIO, B, LSTAT
]

if st.button('Predict'):
    #scale the inputs using the loaded scalar
    scaled_data = scalar.transform(np.array(input_data).reshape(1, -1))
    
    #make the prediction using the trained model
    prediction = regmodel.predict(scaled_data)[0]
    
    #display the predicted house price
    st.write(f"The predicted house price is: {prediction:,.2f} (in $1000's)")