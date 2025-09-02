# 🛠️ Predictive Maintenance Web App

A machine learning web application that predicts equipment failure using sensor data. Built with **XGBoost** for high-accuracy predictions and **Streamlit** for an interactive user interface.

## 🚀 Features

- Predict equipment failure in real-time
- User-friendly interface built with Streamlit
- Visualize predictions and input data
- Upload custom data for batch predictions
- XGBoost model with 98.6% accuracy


## 🧠 Model

- **Algorithm**: XGBoost Classifier
- **Accuracy**: 98.6%
- **Input Features**: Sensor data (e.g., temperature, vibration, pressure)
- **Output**: Binary classification (Failure / No Failure)

## 📂 Dataset  
The dataset includes sensor readings and machine status logs.  

- **Source:** Publicly available industrial maintenance datasets  
- **Features:** Temperature, vibration, pressure, load, etc.  
- **Target:** Machine failure status (Fail / No Fail)  

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shrivastava03/predictive-maintenance-app.git
   cd predictive-maintenance-app

2. Install Dependencies
   pip install -r requirements.txt
   
3. Run the Streamlit App
   streamlit run app.py
   
## Future Improvements

  Integration with real-time IoT sensor data

  More advanced ML models (Deep Learning, RNNs)

  Automated retraining with live data

Cloud deployment (AWS/GCP/Azure)
