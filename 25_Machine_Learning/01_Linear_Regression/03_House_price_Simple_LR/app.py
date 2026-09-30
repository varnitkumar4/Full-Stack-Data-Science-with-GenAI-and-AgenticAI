import streamlit as st
import joblib

# Load trained model
model = joblib.load("house_price_model.pkl")

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title("🏠 House Price Predictor")
st.write("Predict house price based on the area in square feet.")

st.divider()

# User input
area = st.number_input(
    "Enter House Area (sqft)",
    min_value=100,
    max_value=10000,
    value=2500,
    step=50
)

# Prediction button
if st.button("Predict Price", type="primary"):

    prediction = model.predict([[area]])[0]

    st.success(
        f"Estimated House Price: ₹{prediction:.2f} Lakh"
    )

    st.info(
        f"For a house of {area:,.0f} sqft, "
        f"the predicted price is ₹{prediction:.2f} Lakh."
    )