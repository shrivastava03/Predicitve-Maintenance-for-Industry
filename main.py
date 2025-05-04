import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Configure the page
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# Load trained ML model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Display logo and app title
st.image("assets/logo.png", width=120)
st.title("🛠 Predictive Maintenance for Industrial Equipment")

# Sidebar navigation
st.sidebar.title("🔧 Navigation")
section = st.sidebar.radio("Go to:", ["🏠 Home", "🔍 Single Prediction", "📂 Batch Prediction", "📊 Visual Insights", "ℹ️ About"])

# CSS animation for smooth entry
st.markdown("""
    <style>
    .fade-in { animation: fadeIn 1.5s ease-in; }
    @keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
    </style>
""", unsafe_allow_html=True)

# --- Home Section ---
if section == "🏠 Home":
    st.title("🔧 Predictive Maintenance Demo")
    st.markdown("Predict failures before they happen. Let the machines talk. ⚙️💥")

# --- Single Prediction Section ---
elif section == "🔍 Single Prediction":
    st.header("🔍 Predict a Single Machine Failure")

    # Input fields for machine parameters
    machine_type = st.selectbox("Machine Type", ['L', 'M', 'H'])
    air_temp = st.number_input("Air Temperature [K]")
    process_temp = st.number_input("Process Temperature [K]")
    speed = st.number_input("Rotational Speed [rpm]")
    torque = st.number_input("Torque [Nm]")
    wear = st.number_input("Tool Wear [min]")

    if st.button("Predict"):
        type_encoded = {'L': 0, 'M': 1, 'H': 2}[machine_type]
        features = np.array([[type_encoded, air_temp, process_temp, speed, torque, wear]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)

        if prediction[0] == 1:
            st.error("⚠️ Machine is likely to FAIL!")
        else:
            st.success("✅ Machine is operating normally.")

# --- Batch Prediction Section ---
elif section == "📂 Batch Prediction":
    st.header("📂 Batch Prediction from CSV")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview", df.head())

        try:
            df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})
            features = df[['Type', 'Air temperature [K]', 'Process temperature [K]',
                           'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
            features_scaled = scaler.transform(features)
            predictions = model.predict(features_scaled)

            df['Failure_Prediction'] = predictions
            st.write("✅ Predictions", df.head())

            csv_download = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions", csv_download, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            with open("assets/sample_input.csv", "rb") as f:
                st.download_button("📄 Download Sample Template", f, "sample_input.csv", "text/csv")

# --- Visual Insights Section ---
elif section == "📊 Visual Insights":
    st.header("📊 Explore Data Visualizations")

    demo_file = st.file_uploader("Upload a dataset to explore", type=["csv"], key="vis")

    if demo_file:
        data = pd.read_csv(demo_file)
        st.write("📄 Data Preview", data.head())

        if data['Type'].dtype == object:
            data['Type'] = data['Type'].map({'L': 0, 'M': 1, 'H': 2})

        numeric_cols = data.select_dtypes(include='number').columns

        # Correlation heatmap
        st.markdown("### 🔥 Correlation Heatmap")
        plt.figure(figsize=(10, 5))
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt.gcf())
        plt.clf()

        # Failure distribution pie chart
        if 'Machine failure' in data.columns:
            st.markdown("### ⚠️ Failure Distribution")
            failure_counts = data['Machine failure'].value_counts().rename({0: 'No Failure', 1: 'Failure'})
            fig = px.pie(names=failure_counts.index, values=failure_counts.values, title="Failure vs No Failure")
            st.plotly_chart(fig)

        # Distribution charts
        st.markdown("### 📈 Feature Histograms")
        for col in ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']:
            fig = px.histogram(data, x=col, title=f"{col}")
            st.plotly_chart(fig)

        # Box plots
        st.markdown("### 📦 Box Plots for Outlier Detection")
        for col in ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']:
            fig = px.box(data, y=col, title=f"{col} Box Plot")
            st.plotly_chart(fig)

# --- About Section ---
elif section == "ℹ️ About":
    st.header("ℹ️ About this App")
    st.markdown("""
    This application predicts potential machine failures using a trained machine learning model.  
    Built using **Streamlit**, **Scikit-Learn**, **Plotly**, and **Seaborn**.  
    **Author**: [Ishan Shrivastava](https://github.com/yourgithub)  
    📧 ishanshrivastava03@hotmail.com  
    """)

# --- Model Performance Metrics ---
st.markdown("## 📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", "96.3%", "+0.5%")
col2.metric("Precision", "95.1%", "+1.0%")
col3.metric("Recall", "97.8%", "-0.2%")

# --- Footer ---
st.markdown("""---  
💡 Developed by Ishan Shrivastava | 📧 ishanshrivastava03@hotmail.com  
""")
