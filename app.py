import pandas as pd
import streamlit as st
import joblib

# load saved model, scaler
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('robust_scaler.pkl')

# define predict function
def model_process(weight, passengers, length, rooms, model, scaler):
    df = pd.DataFrame([[weight, passengers, length, rooms]],
                      columns=['トン数', '乗客数', '長さ', '船室'])
    X_scaled = scaler.transform(df)
    # X_scaled = pd.DataFrame(scaler.transform(df),
    #                               columns=scaler.get_feature_names_out())
    prediction = model.predict(X_scaled)
    return prediction

# Streamlit app code

st.title("Cruiseship Crew Predictor")
st.image("https://cruisepassenger.com.au/wp-content/uploads/2022/03/WOTS-1.png")
st.header('Enter the cruiseship details:')
st.text('Default values are for Titanic')

# Input features
weight = st.number_input('Weight of Cruiseship in Tons:', 46428)
passengers = st.number_input('Number of passengers:', 2435)
length = st.number_input('Length of cruiseship in Meters:', 269)
rooms = st.number_input('Number of rooms:', 840)

if st.button('Predict Crew'):
    crew = round(model_process(weight, passengers, length, rooms, model, scaler), 0)
    st.success(f'The predicted number of crew needed is {crew[0]}')
