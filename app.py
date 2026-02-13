import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Page configuration
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

# Load model
model = joblib.load("model.pkl")

# Custom CSS Styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #1f77b4;
    }
    .footer {
        position: fixed;
        bottom: 10px;
        width: 100%;
        text-align: center;
        font-size: 14px;
        color: grey;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Session state to switch page
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------- HOME PAGE ----------
if st.session_state.page == "home":
    
    st.markdown("<div class='main-title'>🏠 House Price Predictor</div>", unsafe_allow_html=True)
    st.write("")

    image = Image.open("house.jpg")
    st.image(image, use_container_width=True)


    if st.button("Start Prediction"):
        st.session_state.page = "predict"

# ---------- PREDICTION PAGE ----------
elif st.session_state.page == "predict":

    st.title("Enter House Details")

    MedInc = st.number_input("Median Income")
    HouseAge = st.number_input("House Age")
    AveRooms = st.number_input("Average Rooms")
    AveBedrms = st.number_input("Average Bedrooms")
    Population = st.number_input("Population")
    AveOccup = st.number_input("Average Occupancy")
    Latitude = st.number_input("Latitude")
    Longitude = st.number_input("Longitude")

    if st.button("Predict Price"):
        new_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                              Population, AveOccup, Latitude, Longitude]])

        prediction = model.predict(new_data)

        st.success(f"Predicted House Price: {prediction[0]}")

# ---------- FOOTER ----------
st.markdown("""
    <div class="footer">
        Made by Arun Tech Solutions
    </div>
""", unsafe_allow_html=True)
