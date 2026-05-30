
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os

st.set_page_config(
    page_title="Electricity Forecasting",
    page_icon="⚡",
    layout="wide"
)

st.title(
    "⚡ Electricity Consumption Forecasting"
)

# ------------------------------------
# CHECK FILES
# ------------------------------------

required_files = [
    "energy_rnn_model.h5",
    "scaler.pkl"
]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"Missing file: {file}")
        st.stop()

# ------------------------------------
# LOAD MODEL
# ------------------------------------

model = tf.keras.models.load_model(
    "energy_rnn_model.h5",
    compile=False
)

scaler = joblib.load(
    "scaler.pkl"
)

st.subheader(
    "Enter Previous 24 Hours Consumption"
)

values = []

for i in range(24):
    value = st.number_input(
        f"Hour {i+1}",
        value=100.0
    )

    values.append(value)

if st.button(
    "Predict Next Hour Consumption"
):

    arr = np.array(
        values
    ).reshape(-1,1)

    arr = scaler.transform(
        arr
    )

    arr = arr.reshape(
        1,
        24,
        1
    )

    prediction = model.predict(
        arr,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        prediction
    )

    st.success(
        f"Predicted Consumption: {prediction[0][0]:.2f}"
    )
