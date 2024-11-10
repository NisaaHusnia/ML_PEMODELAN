import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

with open("fruit_model.pkl", "rb") as file:
    model = pickle.load(file)


# Streamlit UI
st.title("Fruit Classification KNN")
fitur1 = st.number_input("Diameter")
fitur2 = st.number_input("Weight")
fitur3 = st.number_input("Red")
fitur4 = st.number_input("Green")
fitur5 = st.number_input("Blue")

# Predict button
if st.button("Predict"):
    prediction = model.predict([[fitur1, fitur2,fitur3,fitur4,fitur5]])
    st.write(f"Predicted Class: {prediction[0]}")