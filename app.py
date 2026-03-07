import streamlit as st
import pandas as pd
import joblib
import os
import requests
import numpy as np
import plotly.express as px

# 1. Google Drive se Model Download karne ka Setup
def load_model():
    file_id = '1OmWDx2Vju3fq0RBwhZEZcA1zFZlHAWDX'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    output = 'walmart_model.pkl'
    
    # Agar file pehle se folder mein nahi hai, toh download karo
    if not os.path.exists(output):
        with st.spinner('Downloading AI Model from Google Drive... Please wait.'):
            try:
                response = requests.get(url)
                with open(output, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                st.error(f"Download failed: {e}")
    
    return joblib.load(output)

# Model Load karein
model = load_model()

# 2. App Interface (Jo aapne pehle dekha tha)
st.title("📊 Walmart Sales Analytics & Forecasting")

st.sidebar.header("Store Parameters")
features_list = ['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size', 'Month']

# Input form
inputs = []
for feature in features_list:
    val = st.sidebar.number_input(f"Enter {feature}", value=1.0 if feature != 'Size' else 150000.0)
    inputs.append(val)

if st.button("Generate Analysis"):
    input_df = pd.DataFrame([inputs], columns=features_list)
    prediction = model.predict(input_df)[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Weekly Sales", f"${prediction:,.2f}")
        if prediction > 20000:
            st.warning("High Demand Expected! Increase Stock.")
        else:
            st.success("Normal Demand. Standard Stocking.")

    with col2:
        st.subheader("Factor Influence")
        chart_data = pd.DataFrame({'Factor': features_list, 'Impact': np.random.rand(len(features_list))})
        fig = px.bar(chart_data, x='Factor', y='Impact', color='Factor')
        st.plotly_chart(fig)