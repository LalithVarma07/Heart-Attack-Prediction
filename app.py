import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.title("💓 Heart Attack Prediction App")

st.write("Enter patient details below:")

# Example inputs (replace with your real features)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")


if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    
    prediction = model.predict(input_data)
    output = round(prediction[0], 2)

    st.success(f"Result of Heart Attack Prediction: {output}")