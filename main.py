import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import time

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Predictive Maintenance",
    layout="wide",
    page_icon="⚙️",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed
)

# Initialize session state for sidebar toggle
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'collapsed'

# -------------------------------
# CUSTOM DARK THEME + UI STYLING
# -------------------------------
st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364) !important;
            color: #f2f2f2 !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #161b22, #0d1117) !important;
            color: #c9d1d9 !important;
            box-shadow: 2px 0 8px rgba(255, 255, 255, 0.05);
        }
        [data-testid="stHeader"] {
            background: #0d1117 !important;
            border-bottom: 1px solid #30363d !important;
        }
        [data-testid="stToolbar"],
        button[kind="header"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# Additional global styling
st.markdown("""
<style>
.stApp {
    background: transparent !important;
    color: #f2f2f2;
    font-family: 'Inter', sans-serif;
}

:root {
    --primary-color: #238636;
    --background-color: #0d1117;
    --background-color-secondary: #161b22;
    --text-color: #c9d1d9;
    --shadow-color: rgba(255, 255, 255, 0.05);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22, #0d1117);
    color: var(--text-color);
    box-shadow: 2px 0 8px var(--shadow-color);
}

.sidebar-title {
    font-size: 1.4em;
    font-weight: 600;
    color: var(--primary-color);
    text-align: center;
    margin-bottom: 12px;
}

div[role="radiogroup"] > label {
    background-color: var(--background-color-secondary);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 10px;
    padding: 8px 12px;
    margin: 5px 0;
    cursor: pointer;
    transition: all 0.2s ease;
    color: var(--text-color);
    box-shadow: 0 2px 4px var(--shadow-color);
}
div[role="radiogroup"] > label:hover {
    background-color: rgba(100, 149, 237, 0.12);
    transform: translateX(3px);
}
div[role="radiogroup"] > label[data-checked="true"] {
    background-color: var(--primary-color);
    color: white !important;
    font-weight: 600;
    border: none;
    box-shadow: 0 0 6px rgba(0, 123, 255, 0.4);
}

button[kind="primary"] {
    background-color: #00adb5 !important;
    color: white !important;
    border-radius: 10px !important;
    transition: 0.3s;
}
button[kind="primary"]:hover {
    background-color: #05c3de !important;
    transform: scale(1.02);
}

div[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: 700;
    color: #00ffcc;
}
div[data-testid="stMetricLabel"] {
    font-size: 1.1rem;
    font-weight: 600;
}

.fade-in { animation: fadeIn 1.2s ease-in-out; }
@keyframes fadeIn {
    0% {opacity: 0; transform: translateY(15px);}
    100% {opacity: 1; transform: translateY(0);}
}

.footer {
    text-align: center;
    color: #aaa;
    font-size: 0.9rem;
    margin-top: 2em;
}

/* Menu Toggle Button Styling */
.menu-toggle {
    position: fixed;
    top: 20px;
    left: 20px;
    z-index: 999;
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 24px;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(35, 134, 54, 0.4);
    transition: all 0.3s ease;
}

.menu-toggle:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950);
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(35, 134, 54, 0.6);
}

.menu-toggle:active {
    transform: translateY(0);
}
</style>
""", unsafe_allow_html=True)

# --- Cache busting (forces CSS reload on redeploy)
st.markdown(f"<div id='version' style='display:none;'>{time.time()}</div>", unsafe_allow_html=True)

# -------------------------------
# MENU TOGGLE BUTTON (Top of page)
# -------------------------------
menu_col1, menu_col2 = st.columns([1, 20])
with menu_col1:
    if st.button("☰", key="menu_toggle", help="Toggle Navigation Menu"):
        # Toggle sidebar state
        if st.session_state.sidebar_state == 'collapsed':
            st.session_state.sidebar_state = 'expanded'
        else:
            st.session_state.sidebar_state = 'collapsed'
        st.rerun()

# Apply sidebar state
if st.session_state.sidebar_state == 'expanded':
    st.markdown("""
        <style>
            [data-testid="stSidebar"][aria-expanded="true"] {
                display: block !important;
            }
            [data-testid="stSidebar"] {
                display: block !important;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: none !important;
            }
        </style>
    """, unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL & SCALER
# -------------------------------
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# -------------------------------
# HEADER SECTION
# -------------------------------
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("logo.png", width=120)

with col_title:
    st.markdown("<h1 class='fade-in'>🛠 Predictive Maintenance for Industrial Equipment</h1>", unsafe_allow_html=True)

st.markdown("<p class='fade-in'>Predict failures before they happen. Let the machines talk. ⚙️💥</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
st.sidebar.markdown('<div class="sidebar-title">🔧 Navigation</div>', unsafe_allow_html=True)

# Add close button in sidebar
if st.sidebar.button("✕ Close Menu", key="close_sidebar"):
    st.session_state.sidebar_state = 'collapsed'
    st.rerun()

section = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "🔍 Single Prediction", "📂 Batch Prediction", "📊 Visual Insights", "ℹ️ About"],
    label_visibility="collapsed"
)

# -------------------------------
# HOME SECTION
# -------------------------------
if section == "🏠 Home":
    st.subheader("🔧 Predictive Maintenance Demo")
    st.markdown("""
    Welcome to the **Predictive Maintenance System**.  
    This tool helps you anticipate machine failures before they occur — minimizing downtime and keeping your operations efficient.  
    Upload data, visualize trends, and predict potential failures in seconds.
    """)

# -------------------------------
# SINGLE PREDICTION
# -------------------------------
elif section == "🔍 Single Prediction":
    st.header("🔍 Predict a Single Machine Failure")

    col1, col2, col3 = st.columns(3)
    with col1:
        machine_type = st.selectbox("Machine Type", ['L', 'M', 'H'])
        air_temp = st.number_input("Air Temperature [K]")
    with col2:
        process_temp = st.number_input("Process Temperature [K]")
        speed = st.number_input("Rotational Speed [rpm]")
    with col3:
        torque = st.number_input("Torque [Nm]")
        wear = st.number_input("Tool Wear [min]")

    if st.button("🚀 Predict"):
        try:
            type_encoded = {'L': 0, 'M': 1, 'H': 2}[machine_type]
            features = np.array([[type_encoded, air_temp, process_temp, speed, torque, wear]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)

            if prediction[0] == 1:
                st.error("⚠️ Machine is likely to FAIL!")
            else:
                st.success("✅ Machine is operating normally.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------------------
# BATCH PREDICTION
# -------------------------------
elif section == "📂 Batch Prediction":
    st.header("📂 Batch Prediction from CSV")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Data Preview", df.head())

        try:
            df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})
            features = df[['Type', 'Air temperature [K]', 'Process temperature [K]',
                           'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
            scaled = scaler.transform(features)
            df['Failure_Prediction'] = model.predict(scaled)

            st.success("✅ Predictions generated successfully!")
            st.write(df.head())

            csv_download = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions", csv_download, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            with open("assets/sample_input.csv", "rb") as f:
                st.download_button("📄 Download Sample Template", f, "sample_input.csv", "text/csv")

# -------------------------------
# VISUAL INSIGHTS
# -------------------------------
elif section == "📊 Visual Insights":
    st.header("📊 Explore Data Visualizations")

    file = st.file_uploader("Upload a dataset to explore", type=["csv"], key="vis")

    if file:
        data = pd.read_csv(file)
        st.write("📄 Data Preview", data.head())

        if data['Type'].dtype == object:
            data['Type'] = data['Type'].map({'L': 0, 'M': 1, 'H': 2})

        numeric_cols = data.select_dtypes(include='number').columns

        st.markdown("### 🔥 Correlation Heatmap")
        plt.figure(figsize=(10, 5))
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt.gcf())
        plt.clf()

        if 'Machine failure' in data.columns:
            st.markdown("### ⚠️ Failure Distribution")
            failure_counts = data['Machine failure'].value_counts().rename({0: 'No Failure', 1: 'Failure'})
            fig = px.pie(names=failure_counts.index, values=failure_counts.values, title="Failure vs No Failure")
            fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font_color='#f0f6fc')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📈 Feature Histograms")
        for col in ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']:
            fig = px.histogram(data, x=col, title=f"{col}")
            fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font_color='#f0f6fc')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📦 Box Plots for Outlier Detection")
        for col in ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']:
            fig = px.box(data, y=col, title=f"{col} Box Plot")
            fig.update_layout(paper_bgcolor='#0d1117', plot_bgcolor='#0d1117', font_color='#f0f6fc')
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# ABOUT SECTION
# -------------------------------
elif section == "ℹ️ About":
    st.header("ℹ️ About this App")
    st.markdown("""
    This application predicts potential machine failures using a trained machine learning model.  
    Built with **Streamlit**, **Scikit-Learn**, **Plotly**, and **Seaborn**.

    **Author:** [Ishan Shrivastava](https://github.com/shrivastava03)  
    📧 **Email:** ishanshrivastava03@hotmail.com  
    """)

# -------------------------------
# MODEL PERFORMANCE (Home + About)
# -------------------------------
if section in ["🏠 Home", "ℹ️ About"]:
    st.markdown("---")
    st.markdown("## 📊 Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "96.3%", "+0.5%")
    c2.metric("Precision", "95.1%", "+1.0%")
    c3.metric("Recall", "97.8%", "-0.2%")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<div class='footer'>
💡 Developed by <b>Ishan Shrivastava</b> | 📧 ishanshrivastava03@hotmail.com
</div>
""", unsafe_allow_html=True)
