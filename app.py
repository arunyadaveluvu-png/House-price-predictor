import streamlit as st
import numpy as np
import joblib
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Arun AI Hub",
    page_icon="🤖",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
house_model = joblib.load("model.pkl")
sales_model = joblib.load("sales_predictor.pkl")
crop_model = joblib.load("crop_model.pkl")

# ---------------- INTRO ANIMATION ----------------
if "intro" not in st.session_state:
    st.session_state.intro = True

if st.session_state.intro:

    st.markdown(
        """
        <div style="text-align:center; padding-top:150px;">
            <h1 style="font-size:60px;">🤖 Arun AI Lab</h1>
            <h3>Machine Learning Applications</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.02)
        progress.progress(i+1)

    st.session_state.intro = False
    st.rerun()

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ======================================================
# ---------------- HOME PAGE ----------------
# ======================================================
if st.session_state.page == "home":

    st.markdown(
        """
        <h1 style='text-align:center;'>🚀 AI Model Dashboard</h1>
        <p style='text-align:center;'>Explore Machine Learning Predictors</p>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # House Model
    with col1:
        st.image("house.jpg", use_container_width=True)
        if st.button("🏠 House Price Predictor"):
            st.session_state.page = "house"
            st.rerun()

    # Sales Model
    with col2:
        st.image("sales.jpg", use_container_width=True)
        if st.button("📊 Sales Predictor"):
            st.session_state.page = "sales"
            st.rerun()

    col3, col4 = st.columns(2)

    # Crop Model
    with col3:
        st.image("crop.jpg", use_container_width=True)
        if st.button("🌾 Crop Predictor"):
            st.session_state.page = "crop"
            st.rerun()

# ======================================================
# ---------------- HOUSE PAGE ----------------
# ======================================================
elif st.session_state.page == "house":

    st.markdown(
        """
        <style>
        body {
        background-image: url("https://images.unsplash.com/photo-1560184897-67f4a3f9a7fa");
        background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🏠 House Price Predictor")

    MedInc = st.number_input("Median Income")
    HouseAge = st.number_input("House Age")
    AveRooms = st.number_input("Average Rooms")
    AveBedrms = st.number_input("Average Bedrooms")
    Population = st.number_input("Population")
    AveOccup = st.number_input("Average Occupancy")
    Latitude = st.number_input("Latitude")
    Longitude = st.number_input("Longitude")

    if st.button("Predict House Price"):

        data = np.array([[MedInc,HouseAge,AveRooms,AveBedrms,
                          Population,AveOccup,Latitude,Longitude]])

        prediction = house_model.predict(data)

        st.success(f"Predicted Price: {prediction[0]:.2f}")

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

# ======================================================
# ---------------- SALES PAGE ----------------
# ======================================================
elif st.session_state.page == "sales":

    st.markdown(
        """
        <style>
        body {
        background-image: url("https://images.unsplash.com/photo-1551288049-bebda4e38f71");
        background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("📊 Sales Predictor")

    TV = st.number_input("TV Budget")
    Radio = st.number_input("Radio Budget")
    Newspaper = st.number_input("Newspaper Budget")

    if st.button("Predict Sales"):

        data = np.array([[TV,Radio,Newspaper]])

        prediction = sales_model.predict(data)

        st.success(f"Predicted Sales: {prediction[0]:.2f}")

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

# ======================================================
# ---------------- CROP PAGE ----------------
# ======================================================
elif st.session_state.page == "crop":

    st.markdown(
        """
        <style>
        body {
        background-image: url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");
        background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🌾 Smart Crop Recommendation")

    N = st.number_input("Nitrogen")
    P = st.number_input("Phosphorus")
    K = st.number_input("Potassium")
    temperature = st.number_input("Temperature")
    humidity = st.number_input("Humidity")
    ph = st.number_input("pH")
    rainfall = st.number_input("Rainfall")

    if st.button("Predict Crop"):

        data = np.array([[N,P,K,temperature,humidity,ph,rainfall]])

        prediction = crop_model.predict(data)

        st.success(f"Recommended Crop: {prediction[0]}")

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

# ---------------- FOOTER ----------------
st.markdown(
"""
<div style='position:fixed; bottom:0; width:100%; text-align:center; background:black; color:white; padding:10px;'>
© Arun AI Lab | Made with ❤️ by Arun
</div>
""",
unsafe_allow_html=True
)