import streamlit as st
import pandas as pd
import joblib
import os
import requests
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Walmart Sales Predictor", layout="wide")

@st.cache_resource
def load_model():
    # 1. Replace the URL below with your actual GitHub Release Download Link
    url = "https://github.com/arti8965/predictive-walmart-analytics/releases/download/v1.0/walmart_model.pkl"
    output = 'walmart_model.pkl'
    
    # 2. Check if file already exists to avoid re-downloading
    if not os.path.exists(output):
        with st.spinner('Downloading ML Model (583MB). This may take 3-5 minutes depending on speed...'):
            try:
                # Streaming download for large files
                response = requests.get(url, stream=True)
                response.raise_for_status() # Check for download errors
                with open(output, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()
    
    # 3. Load the model using joblib
    try:
        return joblib.load(output)
    except Exception as e:
        st.error(f"Model loading error: {e}. Check if scikit-learn version matches.")
        st.stop()

# --- Dashboard UI ---
st.title("📊 Walmart Weekly Sales Forecasting")
st.markdown("---")

try:
    # Initialize model
    model = load_model()
    
    # Sidebar for User Inputs
    st.sidebar.header("Input Store Features")
    store = st.sidebar.number_input("Store Number", min_value=1, max_value=45, value=1)
    holiday = st.sidebar.selectbox("Is it a Holiday Week?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    temp = st.sidebar.slider("Temperature (F)", -10.0, 100.0, 60.0)
    fuel = st.sidebar.slider("Fuel Price", 2.0, 5.0, 3.5)
    cpi = st.sidebar.slider("Consumer Price Index (CPI)", 120.0, 230.0, 170.0)
    unemployment = st.sidebar.slider("Unemployment Rate", 3.0, 15.0, 7.5)

    # Main area for prediction
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Action")
        if st.button("🚀 Predict Weekly Sales"):
            # Create a dataframe for the model
            input_data = pd.DataFrame([[store, holiday, temp, fuel, cpi, unemployment]], 
                                     columns=['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'])
            
            prediction = model.predict(input_data)
            
            st.success(f"### Predicted Weekly Sales: ${prediction[0]:,.2f}")
            
            with col2:
                st.subheader("Visualization")
                fig = px.bar(x=['Predicted Sales'], y=[prediction[0]], 
                             labels={'x': '', 'y': 'Sales in USD'},
                             color_discrete_sequence=['#0071ce']) # Walmart Blue
                st.plotly_chart(fig)

except Exception as main_e:
    st.info("Please wait while the system initializes the model.")
    if st.button("Force Clear Cache"):
        if os.path.exists('walmart_model.pkl'):
            os.remove('walmart_model.pkl')
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Model size: 583MB. Hosted via GitHub Releases.")
