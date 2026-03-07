import streamlit as st
import numpy as np
import joblib
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Arun AI Lab", layout="wide")

# ---------------- LOAD MODELS ----------------
house_model = joblib.load("model.pkl")
sales_model = joblib.load("sales_predictor.pkl")
crop_model = joblib.load("crop_predictor.pkl")

# ---------------- NEURAL NETWORK BACKGROUND ----------------
st.markdown("""
<style>

#particles-js{
position:fixed;
width:100%;
height:100%;
top:0;
left:0;
z-index:-1;
background:#0f2027;
}

.title{
text-align:center;
font-size:50px;
color:white;
font-weight:bold;
}

.subtitle{
text-align:center;
color:#ccc;
margin-bottom:40px;
}

.card{
background:#111;
border-radius:15px;
padding:15px;
box-shadow:0 0 20px rgba(0,0,0,0.5);
transition:0.3s;
}

.card:hover{
transform:scale(1.05);
}

.card-img{
width:100%;
height:230px;
border-radius:10px;
object-fit:cover;
}

.stButton>button{
width:100%;
height:45px;
border-radius:10px;
background:#0d6efd;
color:white;
font-size:16px;
}

.footer{
position:fixed;
bottom:0;
left:0;
width:100%;
text-align:center;
background:#111;
color:white;
padding:10px;
}

</style>

<div id="particles-js"></div>

<script src="https://cdn.jsdelivr.net/npm/particles.js"></script>

<script>
particlesJS("particles-js",{
"particles":{
"number":{"value":80},
"size":{"value":3},
"move":{"speed":2},
"line_linked":{"enable":true}
}
});
</script>

""", unsafe_allow_html=True)

# ---------------- INTRO ----------------
if "intro" not in st.session_state:
    st.session_state.intro = True

if st.session_state.intro:

    st.markdown("""
    <div style="text-align:center;padding-top:200px;">
    <h1 style="font-size:70px;color:white;">MODEL HUB</h1>
    <h3 style="color:white;">Machine Learning Applications</h3>
    </div>
    """, unsafe_allow_html=True)

    bar = st.progress(0)

    for i in range(100):
        time.sleep(0.02)
        bar.progress(i+1)

    st.session_state.intro=False
    st.rerun()

# ---------------- PAGE STATE ----------------
if "page" not in st.session_state:
    st.session_state.page="home"

# ====================================================
# HOME PAGE
# ====================================================
if st.session_state.page=="home":

    st.markdown('<div class="title">WELCOME BACK</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Explore Machine Learning Predictors</div>', unsafe_allow_html=True)

    col1,col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image("house.jpg", use_container_width=True)
        if st.button("🏠 House Price Predictor"):
            st.session_state.page="house"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image("sales.jpg", use_container_width=True)
        if st.button("📊 Sales Predictor"):
            st.session_state.page="sales"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    col3,col4 = st.columns(2)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image("https://images.unsplash.com/photo-1500382017468-9049fed747ef", use_container_width=True)
        if st.button("🌾 Crop Predictor"):
            st.session_state.page="crop"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ====================================================
# HOUSE PAGE
# ====================================================
elif st.session_state.page=="house":

    st.title("🏠 House Price Predictor")

    MedInc=st.number_input("Median Income")
    HouseAge=st.number_input("House Age")
    AveRooms=st.number_input("Average Rooms")
    AveBedrms=st.number_input("Average Bedrooms")
    Population=st.number_input("Population")
    AveOccup=st.number_input("Average Occupancy")
    Latitude=st.number_input("Latitude")
    Longitude=st.number_input("Longitude")

    if st.button("Predict Price"):

        data=np.array([[MedInc,HouseAge,AveRooms,AveBedrms,
                        Population,AveOccup,Latitude,Longitude]])

        prediction=house_model.predict(data)

        st.success(f"Predicted Price: {prediction[0]:.2f}")

    if st.button("⬅ Back"):
        st.session_state.page="home"
        st.rerun()

# ====================================================
# SALES PAGE
# ====================================================
elif st.session_state.page=="sales":

    st.title("📊 Sales Predictor")

    TV=st.number_input("TV Budget")
    Radio=st.number_input("Radio Budget")
    Newspaper=st.number_input("Newspaper Budget")

    if st.button("Predict Sales"):

        data=np.array([[TV,Radio,Newspaper]])

        prediction=sales_model.predict(data)

        st.success(f"Predicted Sales: {prediction[0]:.2f}")

    if st.button("⬅ Back"):
        st.session_state.page="home"
        st.rerun()

# ====================================================
# CROP PAGE
# ====================================================
elif st.session_state.page=="crop":

    st.title("🌾 Crop Recommendation")

    N=st.number_input("Nitrogen")
    P=st.number_input("Phosphorus")
    K=st.number_input("Potassium")
    temperature=st.number_input("Temperature")
    humidity=st.number_input("Humidity")
    ph=st.number_input("pH")
    rainfall=st.number_input("Rainfall")

    if st.button("Predict Crop"):

        data=np.array([[N,P,K,temperature,humidity,ph,rainfall]])

        prediction=crop_model.predict(data)

        st.success(f"Recommended Crop: {prediction[0]}")

    if st.button("⬅ Back"):
        st.session_state.page="home"
        st.rerun()

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
© Arun Software Solutions | Made with ❤️ by Arun
</div>
""", unsafe_allow_html=True)