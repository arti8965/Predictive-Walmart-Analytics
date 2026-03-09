import streamlit as st
import pandas as pd
import joblib
import os
import requests
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Walmart Sales Predictor", layout="wide")

# Function to download and load the model
def load_model():
    # Your Google Drive File ID
    file_id = '1OmWDx2Vju3fq0RBwhZEZcA1zFZlHAWDX'
    
    # Direct link that bypasses Google's large file warning
    url = f'https://drive.google.com/uc?export=download&id={file_id}&confirm=t'
    output = 'walmart_model.pkl'
    
    # If file doesn't exist or is too small (meaning it's a corrupt HTML file), download it
    if not os.path.exists(output) or os.path.getsize(output) < 5000:
        with st.spinner('Downloading ML Model from Google Drive... This may take a moment.'):
            try:
                # Using a session to handle potential cookies/redirects
                session = requests.Session()
                response = session.get(url, stream=True)
                with open(output, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()
    
    # Try to load the model
    try:
        model = joblib.load(output)
        return model
    except Exception as e:
        # If loading still fails, delete the file and show a manual fix
        if os.path.exists(output):
            os.remove(output)
        st.error("Error: The downloaded model file is incompatible. Please Reboot the app from the Manage App menu.")
        st.stop()

# Initialize App
st.title("📊 Walmart Sales Forecasting Dashboard")
st.markdown("Predicting weekly sales based on store data and economic indicators.")

# Load the model
model = load_model()

if model:
    st.sidebar.header("Input Features")
    
    # User Input Fields
    store = st.sidebar.number_input("Store Number", min_value=1, max_value=45, value=1)
    holiday = st.sidebar.selectbox("Is it a Holiday Week?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    temp = st.sidebar.slider("Temperature", -10.0, 100.0, 60.0)
    fuel_price = st.sidebar.slider("Fuel Price", 2.0, 5.0, 3.5)
    cpi = st.sidebar.slider("CPI (Consumer Price Index)", 120.0, 230.0, 170.0)
    unemployment = st.sidebar.slider("Unemployment Rate", 3.0, 15.0, 7.5)

    # Prepare data for prediction
    input_data = pd.DataFrame([[store, holiday, temp, fuel_price, cpi, unemployment]], 
                              columns=['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'])

    # Prediction Button
    if st.button("Predict Weekly Sales"):
        prediction = model.predict(input_data)
        st.success(f"### Predicted Weekly Sales: ${prediction[0]:,.2f}")
        
        # Simple Visualization
        fig = px.bar(x=['Predicted Sales'], y=[prediction[0]], labels={'x': '', 'y': 'Sales in USD'})
        st.plotly_chart(fig)
