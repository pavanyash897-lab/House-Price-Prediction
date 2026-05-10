import streamlit as st
import pandas as pd
import joblib

# Load Saved Model
model = joblib.load("gradient_boosting_pipeline.pkl")

# Page Configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title("🏠 House Price Prediction")

st.write(
    "Enter house details below to predict the estimated price."
)

# Sidebar
st.sidebar.header("About Project")

st.sidebar.info(
    """
    Machine Learning based House Price Prediction System.

    Final Model:
    XGBoost Regressor
    """
)

# User Inputs
square_footage = st.number_input(
    "Square Footage",
    min_value=500,
    max_value=5000,
    value=2800
)

num_bedrooms = st.slider(
    "Number of Bedrooms",
    min_value=1,
    max_value=5,
    value=3
)

year_built = st.number_input(
    "Year Built",
    min_value=1950,
    max_value=2022,
    value=2000
)

lot_size = st.slider(
    "Lot Size",
    min_value=0.5,
    max_value=5.0,
    value=2.5
)

garage_size = st.slider(
    "Garage Size",
    min_value=0,
    max_value=2,
    value=1
)

# Prediction
if st.button("Predict House Price"):

    # Create Input DataFrame
    input_data = pd.DataFrame([{
        'Square_Footage': square_footage,
        'Num_Bedrooms': num_bedrooms,
        'Year_Built': year_built,
        'Lot_Size': lot_size,
        'Garage_Size': garage_size
    }])

    # Predict
    prediction = model.predict(input_data)

    # Show Result
    st.success(
        f"🏡 Estimated House Price: ₹ {prediction[0]:,.2f}"
    )


# Footer
st.markdown("---")

st.caption(
    "Developed using Streamlit, Scikit-Learn, and XGBoost"
)
