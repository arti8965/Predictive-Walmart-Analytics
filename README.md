# 📊 Walmart Sales Analytics & Forecasting

An interactive Machine Learning dashboard designed to predict weekly sales for Walmart stores. This application provides real-time insights based on various economic and environmental factors like CPI, Fuel Price, and Unemployment.

## 🚀 Live Project Link
[👉 Click here to view the Live App](https://smart-store-analytics-pro.streamlit.app) 
*(Note: Replace this with your actual URL once deployed)*

## 📝 Project Overview
The goal of this project is to provide a user-friendly interface for a Machine Learning model that forecasts retail sales. It allows stakeholders to input specific store parameters and instantly see the predicted revenue, helping in inventory management and resource allocation.

## 🛠️ Tech Stack
* **Programming Language:** Python 3.10+
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-learn, Joblib
* **Data Visualization:** Plotly Express, Pandas, Numpy
* **Hosting:** Streamlit Cloud
* **External Storage:** Google Drive API (used to fetch the 25MB+ pre-trained model)

## 🌟 Key Features
* **Automated Model Retrieval:** The app identifies if the ML model is missing and automatically downloads it from a secure Google Drive link.
* **Interactive Sidebar:** Users can adjust 9 different parameters (Store ID, Dept, Temperature, etc.) to see how they affect sales.
* **Predictive Metrics:** Displays the exact dollar amount predicted for weekly sales.
* **Smart Alerts:** Automatically flags "High Demand" scenarios if predicted sales cross a specific threshold.
* **Dynamic Visualizations:** An interactive bar chart showing the impact of various factors on the prediction.

## 📂 Repository Structure
* `app.py`: The core Python application script containing the logic and UI.
* `requirements.txt`: Configuration file listing all required Python libraries for the environment.
* `README.md`: Project documentation and presentation.

## ⚙️ Installation & Local Setup
To run this project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/arti8965/Predictive-Walmart-Analytics.git](https://github.com/arti8965/Predictive-Walmart-Analytics.git)
  
  # Navigate to the project directory:
Bash
cd Predictive-Walmart-Analytics
 #Install dependencies:
Bash
pip install -r requirements.txt
 #Run the application:
Bash
streamlit run app.py
   
