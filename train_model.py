# train_model.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib

# Fetch data
end = datetime.now()
start = datetime(end.year - 15, end.month, end.day)
data = yf.download('BTC-USD', start=start, end=end)[['Close']]
data.dropna(inplace=True)

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare sequences
x, y = [], []
for i in range(100, len(scaled_data)):
    x.append(scaled_data[i - 100:i])
    y.append(scaled_data[i])
x, y = np.array(x), np.array(y)

# Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(x.shape[1], 1)),
    LSTM(64),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, batch_size=10, epochs=5)

# Save model and scaler
model.save("model.keras")
joblib.dump(scaler, "scaler.save")
