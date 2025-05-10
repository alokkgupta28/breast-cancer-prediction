import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data

# Train model
@st.cache_resource
def train_model(df):
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    return model, scaler, acc

# Streamlit UI
st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.title("🔬 Breast Cancer Prediction App")
st.markdown("Using Random Forest on Breast Cancer Wisconsin Diagnostic Dataset")

# Load data and model
df, raw_data = load_data()
model, scaler, accuracy = train_model(df)

# Sidebar - feature input
st.sidebar.header("Input Features")
user_input = {}

for feature in raw_data.feature_names:
    val = st.sidebar.slider(
        label=feature,
        min_value=float(df[feature].min()),
        max_value=float(df[feature].max()),
        value=float(df[feature].mean())
    )
    user_input[feature] = val

# Prediction
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)[0]

st.subheader("📈 Model Accuracy")
st.write(f"Accuracy on test data: **{accuracy * 100:.2f}%**")

st.subheader("🧪 Prediction Result")
st.write(f"**Prediction:** {'Malignant' if prediction == 0 else 'Benign'}")
st.write(f"**Confidence:** {prediction_proba[prediction] * 100:.2f}%")

st.subheader("🔍 Feature Values You Provided")
st.write(input_df)
