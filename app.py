import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Predictor",
    page_icon="🤖",
    layout="centered"
)

# ---------------- LOAD MODELS ----------------
house_model = joblib.load("model.pkl")
sales_model = joblib.load("sales_predictor.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 20px;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #111;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 15px;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    img {
        height: 250px !important;
        object-fit: cover;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# =========================================================
# ---------------- HOME PAGE ----------------
# =========================================================
if st.session_state.page == "home":

    st.markdown("<div class='main-title'>🤖 AI Predictors</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.image("house.jpg", use_container_width=True)
        if st.button("🏠 House Price Prediction"):
            st.session_state.page = "house"
            st.rerun()

    with col2:
        st.image("sales.jpg", use_container_width=True)
        if st.button("📊 Sales Prediction"):
            st.session_state.page = "sales"
            st.rerun()


# =========================================================
# ---------------- HOUSE PREDICTOR PAGE ----------------
# =========================================================
elif st.session_state.page == "house":

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
        new_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                              Population, AveOccup, Latitude, Longitude]])
        prediction = house_model.predict(new_data)
        st.success(f"Predicted House Price: {prediction[0]:.2f}")

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
        st.rerun()


# =========================================================
# ---------------- SALES PREDICTOR PAGE ----------------
# =========================================================
elif st.session_state.page == "sales":

    st.title("📊 Sales Predictor")

    TV = st.number_input("TV Ad Budget ($)")
    Radio = st.number_input("Radio Ad Budget ($)")
    Newspaper = st.number_input("Newspaper Ad Budget ($)")

    if st.button("Predict Sales"):
        new_data = np.array([[TV, Radio, Newspaper]])
        prediction = sales_model.predict(new_data)
        st.success(f"Predicted Sales ($): {prediction[0]:.2f}")

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
        st.rerun()


# ---------------- FOOTER ----------------
st.markdown("""
    <div class="footer">
        Copy-right © Arun. Made with ❤️ by Arun Software Solutions.
    </div>
""", unsafe_allow_html=True)
