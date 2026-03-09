import streamlit as st
import pandas as pd
import joblib
import os
import requests
import plotly.express as px

# SSL errors ko ignore karne ke liye settings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="Walmart Sales Predictor", layout="wide")

def load_model():
    # Direct Google Drive link logic
    file_id = '1OmWDx2Vju3fq0RBwhZEZcA1zFZlHAWDX'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    output = 'walmart_model.pkl'
    
    # Agar purani file choti hai (<1MB), toh wo kharab hai, use delete karo
    if os.path.exists(output) and os.path.getsize(output) < 1000000:
        os.remove(output)

    if not os.path.exists(output):
        with st.spinner('Downloading Model (Approx 50MB)... Please wait 1-2 minutes.'):
            try:
                # verify=False is the solution for Error 60
                response = requests.get(url, verify=False, stream=True)
                with open(output, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()
    
    return joblib.load(output)

st.title("📊 Walmart Sales Forecasting Dashboard")

try:
    model = load_model()
    
    st.sidebar.header("Input Features")
    store = st.sidebar.number_input("Store Number", 1, 45, 1)
    holiday = st.sidebar.selectbox("Holiday?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    temp = st.sidebar.slider("Temperature", -10.0, 100.0, 60.0)
    fuel = st.sidebar.slider("Fuel Price", 2.0, 5.0, 3.5)
    cpi = st.sidebar.slider("CPI", 120.0, 230.0, 170.0)
    unemployment = st.sidebar.slider("Unemployment", 3.0, 15.0, 7.5)

    if st.button("Predict Weekly Sales"):
        input_df = pd.DataFrame([[store, holiday, temp, fuel, cpi, unemployment]], 
                               columns=['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'])
        prediction = model.predict(input_df)
        st.success(f"### Predicted Sales: ${prediction[0]:,.2f}")
        st.plotly_chart(px.bar(x=['Sales'], y=[prediction[0]]))

except Exception as e:
    st.warning("Model loading in progress or setup incomplete.")
    if st.button("Force Reset & Redownload"):
        if os.path.exists('walmart_model.pkl'):
            os.remove('walmart_model.pkl')
        st.rerun()
