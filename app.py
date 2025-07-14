# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Style
plt.style.use('fivethirtyeight')

# Title
st.title("🪙 Bitcoin Price Predictor")
st.write("Using LSTM Model to Predict Future Bitcoin Prices")

# Load model
model = load_model("model.keras")

# Load data
end = datetime.now()
start = datetime(end.year - 15, end.month, end.day)
stock = 'BTC-USD'
stock_data = yf.download(stock, start=start, end=end)
closing_price = stock_data[['Close']]

# Display data
st.subheader("📈 Recent Closing Prices")
st.line_chart(closing_price['Close'].tail(100))

# Show Moving Averages
closing_price['MA_365'] = closing_price['Close'].rolling(window=365).mean()
closing_price['MA_100'] = closing_price['Close'].rolling(window=100).mean()

st.subheader("📉 Closing Price with Moving Averages")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(closing_price['Close'], label='Close Price', color='blue')
ax.plot(closing_price['MA_365'], label='365 Days MA', color='red', linestyle='--')
ax.plot(closing_price['MA_100'], label='100 Days MA', color='green', linestyle='--')
ax.set_title("Close Price with Moving Averages")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_price[['Close']].dropna())

# Prepare prediction
base_days = 100
last_100 = scaled_data[-base_days:].reshape(1, -1, 1)
future_predictions = []

for _ in range(10):
    next_day = model.predict(last_100)
    future_predictions.append(scaler.inverse_transform(next_day)[0][0])
    last_100 = np.append(last_100[:, 1:, :], next_day.reshape(1, -1, 1), axis=1)

# Plot prediction
st.subheader("🔮 Future Bitcoin Price (Next 10 Days)")
fig2, ax2 = plt.subplots(figsize=(12, 4))
days = list(range(1, 11))
ax2.plot(days, future_predictions, marker="o", label='Predicted Price', color='purple')
for i, val in enumerate(future_predictions):
    ax2.text(i + 1, val, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
ax2.set_title("Future Predictions for 10 Days")
ax2.set_xlabel("Days Ahead")
ax2.set_ylabel("Price (USD)")
ax2.grid(alpha=0.3)
ax2.legend()
st.pyplot(fig2)
