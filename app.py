import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare data
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Streamlit app
st.title("🏠 California House Price Predictor")
st.write("Adjust the sliders to predict the price of a house.")

# Create sliders for inputs
medinc = st.slider("Median Income", 0.5, 15.0, 5.0)
house_age = st.slider("House Age", 1, 52, 25)
avg_rooms = st.slider("Average Rooms", 1.0, 15.0, 5.0)
avg_bedrooms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
population = st.slider("Population", 100, 5000, 1000)
avg_occup = st.slider("Average Occupancy", 1.0, 10.0, 3.0)
latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
longitude = st.slider("Longitude", -125.0, -114.0, -118.0)

# Make prediction
input_features = np.array([
    medinc, house_age, avg_rooms, avg_bedrooms,
    population, avg_occup, latitude, longitude
]).reshape(1, -1)

input_scaled = scaler.transform(input_features)
predicted_price = model.predict(input_scaled)[0] * 100000

st.subheader(f"💰 Predicted House Price: ${predicted_price:,.2f}")
