import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

# ------------------------
# Dummy Model Preparation
# ------------------------
data = {
    'age': np.random.randint(20, 70, 100),
    'sleep_hours': np.random.uniform(4.0, 12.0, 100).round(1),
    'work_pressure': np.random.randint(1, 6, 100),
    'social_score': np.random.randint(1, 11, 100),
    'is_depressed': np.random.randint(0, 2, 100)
}
dummy_df = pd.DataFrame(data)

X_dummy = dummy_df[['age', 'sleep_hours', 'work_pressure', 'social_score']]
y_dummy = dummy_df['is_depressed']

dummy_model = RandomForestClassifier(random_state=42)
dummy_model.fit(X_dummy, y_dummy)

model_path = "dummy_depression_model.pkl"
if not os.path.exists(model_path):
    with open(model_path, "wb") as f:
        pickle.dump(dummy_model, f)

# ------------------------
# Streamlit Config
# ------------------------
st.set_page_config(
    page_title="Depression Predictor",
    layout="wide",
    page_icon="🧠"
)

# ------------------------
# Theme Toggle (Switch)
# ------------------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

if dark_mode:
    st.markdown(
        """
        <style>
        .stApp, .st-emotion-cache-1r6slb0, .st-emotion-cache-6qob1r { 
            background-color: #121212 !important; 
            color: #ffffff !important; 
        }
        .sidebar .st-emotion-cache-6qob1r, .css-1d391kg {
            background-color: #1e1e1e !important;
        }
        .stButton>button {
            background-color: #1565c0;
            color: white;
            border-radius: 10px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp, .st-emotion-cache-1r6slb0, .st-emotion-cache-6qob1r {
            background-color: #f9f9f9 !important;
            color: #000000 !important;
        }
        .sidebar .st-emotion-cache-6qob1r, .css-1d391kg {
            background-color: #ffffff !important;
        }
        .stButton>button {
            background-color: #1e88e5;
            color: white;
            border-radius: 10px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True
    )

# ------------------------
# Sidebar Inputs
# ------------------------
with st.sidebar.expander("⚙️ Input Parameters", expanded=True):
    age = st.slider("Age", 18, 80, 35)
    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)
    work_pressure = st.slider("Work Pressure (1=Low, 5=High)", 1, 5, 3)
    social_score = st.slider("Social Score (1=Low, 10=High)", 1, 10, 5)

user_data = pd.DataFrame({
    "age": [age],
    "sleep_hours": [sleep_hours],
    "work_pressure": [work_pressure],
    "social_score": [social_score]
})

# ------------------------
# Main UI
# ------------------------
st.title("🧠 Depression Risk Predictor")
st.write("This tool predicts the risk of depression based on lifestyle and social factors.")

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Your Data")
    st.table(user_data)

with col2:
    st.subheader("Prediction Result")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if st.button("Get Prediction"):
        proba = model.predict_proba(user_data)[0][1]
        st.metric("Depression Risk Score", f"{proba:.2f}")

        if proba >= 0.7:
            st.error("❗ High Risk - Please consult a professional.")
        elif proba >= 0.4:
            st.warning("⚠️ Moderate Risk - Consider talking to someone.")
        else:
            st.success("✅ Low Risk - Keep focusing on your well-being.")

st.info("Disclaimer: This demo is not medical advice. Always consult professionals for health concerns.")
