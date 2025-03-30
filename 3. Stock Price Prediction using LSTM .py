""" 
📈 Stock Price Prediction using LSTM (Python + TensorFlow + Keras)

👉 यह प्रोजेक्ट LSTM (Long Short-Term Memory) Neural Network का उपयोग करके Stock Price Prediction करेगा।
👉 हम Yahoo Finance API से Real-time Stock Data लेंगे और LSTM Model से Future Prediction करेंगे।
👉 आप इस प्रोजेक्ट को GitHub पर Upload करके Resume में डाल सकते हैं! 🚀
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error

# Load Stock Data (Example: Apple Stock - AAPL)
stock_symbol = "AAPL"
start_date = "2015-01-01"
end_date = "2024-03-30"

df = yf.download(stock_symbol, start=start_date, end=end_date)

# Display Data
print(df.head())

# Plot Stock Price
plt.figure(figsize=(12, 6))
plt.plot(df["Close"], label="Stock Price (Close)")
plt.title(f"{stock_symbol} Stock Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

# Select Closing Price Column
data = df[["Close"]].values  # Convert to numpy array

# Normalize Data (0-1 Scale)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Create Sequences (Last 60 Days -> Next Day Price)
X, y = [], []
sequence_length = 60

for i in range(sequence_length, len(data_scaled)):
    X.append(data_scaled[i-sequence_length:i, 0])
    y.append(data_scaled[i, 0])

X, y = np.array(X), np.array(y)

# Split Data into Training & Testing (80-20 Split)
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Reshape Data for LSTM (Batch Size, Time Steps, Features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")


# Build LSTM Model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

# Compile Model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# Predict on Test Data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))  # Convert to original scale

# Convert y_test back to original scale
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate MAE (Mean Absolute Error)
mae = mean_absolute_error(actual_prices, predictions)
print(f"Model MAE: {mae:.2f}")


# Plot Predictions vs Actual Prices
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, color="blue", label="Actual Stock Price")
plt.plot(predictions, color="red", linestyle="dashed", label="Predicted Stock Price")
plt.title(f"{stock_symbol} Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()


# Use Last 60 Days Data for Prediction
last_60_days = data_scaled[-sequence_length:]
X_future = np.array([last_60_days])
X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))

# Predict Next Day Price
predicted_price = model.predict(X_future)
predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))

print(f"📈 Predicted Next Day Price: ${predicted_price[0][0]:.2f}")

