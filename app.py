import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.title('House Price Prediction')

# load model
model = pickle.load(open('Linear_model.pkl','rb'))

# input features
Square_Footage = st.number_input('Square Footage', min_value=503.0, max_value=4999.0, value=503.0)
Num_Bedrooms = st.number_input('Num Bedrooms', min_value=1, max_value=5, value=1)
Year_Built = st.number_input('Year Built', min_value=1950, max_value=2022, value=2010)
Lot_Size = st.number_input('Lot Size', min_value=0.506058, max_value=4.98, value=1.0)
Garage_Size = st.number_input('Garage Size', min_value=0, max_value=2, value=0)


# create dataframe (MATCH TRAINING COLUMN NAMES EXACTLY)
input_data = pd.DataFrame({
    'Square_Footage': [Square_Footage],
    'Num_Bedrooms': [Num_Bedrooms],
    'Year_Built': [Year_Built],
    'Lot_Size': [Lot_Size],
    'Garage_Size': [Garage_Size],
})
scaler=StandardScaler()
input_data[['Square_Footage','Num_Bedrooms','Lot_Size','Garage_Size']]=scaler.fit_transform(input_data[['Square_Footage','Num_Bedrooms','Lot_Size','Garage_Size']])

# prediction
if st.button('Predict'):
    predictions = model.predict(input_data)
    output = round(predictions[0], 2)
    st.success(f'Predicted House Price: {output}')

