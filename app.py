import streamlit as st
import pandas as pd
import joblib
import os
import requests
import plotly.express as px

st.set_page_config(page_title="Walmart Sales Predictor", layout="wide")

def load_model():
    # Your verified Google Drive File ID
    file_id = '1OmWDx2Vju3fq0RBwhZEZcA1zFZlHAWDX'
    output = 'walmart_model.pkl'
    
    # If the file exists but is too small, it's corrupt, so we delete it
    if os.path.exists(output) and os.path.getsize(output) < 100000:
        os.remove(output)

    if not os.path.exists(output):
        with st.spinner('Downloading ML Model (Large File)... Please wait.'):
            # Special logic for Google Drive Large Files
            session = requests.Session()
            download_url = "https://docs.google.com/uc?export=download"
            response = session.get(download_url, params={'id': file_id}, stream=True)
            
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            
            if token:
                response = session.get(download_url, params={'id': file_id, 'confirm': token}, stream=True)
            
            with open(output, "wb") as f:
                for chunk in response.iter_content(32768):
                    if chunk: f.write(chunk)
    
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
    st.error(f"Error: {e}")
    # This button helps you clear the bad file if it fails
    if st.button("Clear Cache & Retry"):
        if os.path.exists('walmart_model.pkl'):
            os.remove('walmart_model.pkl')
        st.rerun()
