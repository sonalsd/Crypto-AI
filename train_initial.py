import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os

def build_training_pipeline():
    print("--- Phase 1: Data Engineering ---")
    # Fetching 60 days of hourly BTC data for a robust training set
    data = yf.download("BTC-USD", period="60d", interval="1h")
    
    # Selecting the 'Close' price for the univariate time-series
    df = data[['Close']].fillna(method='ffill')
    
    # Scaling data: A core AISE requirement for neural network stability
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Creating sequences (sliding window of 10 hours to predict the 11th)
    X, y = [], []
    for i in range(10, len(scaled_data)):
        X.append(scaled_data[i-10:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    print("--- Phase 2: Model Engineering ---")
    # Defining an LSTM Architecture
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(units=50),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Training the model... please wait.")
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)

    print("--- Phase 3: Model Persistence ---")
    # Saving the model and the scaler for the 'Operation' phase
    model.save('crypto_model.h5')
    joblib.dump(scaler, 'scaler.gz')
    print("Success! 'crypto_model.h5' and 'scaler.gz' have been created.")

if __name__ == "__main__":
    build_training_pipeline()
