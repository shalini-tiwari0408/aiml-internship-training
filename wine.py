import streamlit as st
st.title("Wine Quality Prediction")
st.write("this is a app designed to predict wine quality.")
fixed_acidity = st.number_input("Enter Fixed Acidity", min_value=0.0, max_value=20.0, step=0.1)
volatile_acidity = st.number_input("Enter Volatile Acidity", min_value=0.0, max_value=5.0, step=0.1)
citric_acid = st.number_input("Enter Citric Acid", min_value=0.0, max_value=5.0, step=0.1)
residual_sugar = st.number_input("Enter Residual Sugar", min_value=0.0, max_value=20.0, step=0.1)
chlorides = st.number_input("Enter Chlorides", min_value=0.0, max_value=1.0, step=0.01)
free_sulfur_dioxide = st.number_input("Enter Free Sulfur Dioxide", min_value=0.0, max_value=100.0, step=1.0)
total_sulfur_dioxide = st.number_input("Enter Total Sulfur Dioxide", min_value=0.0, max_value=300.0, step=1.0)
density = st.number_input("Enter Density", min_value=0.0, max_value=2.0, step=0.0001)
pH = st.number_input("Enter pH", min_value=0.0, max_value=14.0, step=0.01)
sulphates = st.number_input("Enter Sulphates", min_value=0.0, max_value=5.0, step=0.01)
alcohol = st.number_input("Enter Alcohol", min_value=0.0, max_value=20.0, step=0.1)

import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model("wine_quality_model.h5")
prediction = model.predict(np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]))
if st.button("Predict Quality"):
    pred = np.argmax(prediction, axis=1)[0] + 3  # Adjusting back to original labels
    st.write(f"Predicted Wine Quality: {pred}")