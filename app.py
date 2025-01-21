import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import datetime

# Define the file path for the model
model_path = 'H:\Data_science_srk\Projects\Resume Project\Stock-Price-Predictions-main\Stock_price_prediction.keras'

# Function to fetch the latest data
def fetch_data(stock, start_date, end_date):
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(stock, start=start_date, end=end_date)
    df = pd.DataFrame(data)
    df.reset_index(inplace=True)
    return df

# Function to preprocess the data
def preprocess_data(df):
    """Scale the 'Close' prices and prepare the data for training or prediction."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Function to create training data
def create_training_data(scaled_data):
    """Prepare the training data using the past 100 days as input."""
    x_train, y_train = [], []
    for i in range(100, len(scaled_data)):
        x_train.append(scaled_data[i - 100:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train

# Function to create and retrain the model
def retrain_model(x_train, y_train, model_path):
    """Retrain the LSTM model on new data."""
    print(f"Retraining model at {datetime.datetime.now()}")  # Log message
    # Define the LSTM model structure
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=5, batch_size=32)

    # Save the updated model
    model.save(model_path)
    print(f"Model retrained and saved at {datetime.datetime.now()}")  # Confirmation
    return model

# Function to check if the model needs retraining
def is_model_recent(model_path, days=7):
    if os.path.exists(model_path):
        last_modified_time = os.path.getmtime(model_path)
        last_modified_date = datetime.datetime.fromtimestamp(last_modified_time)
        if (datetime.datetime.now() - last_modified_date).days <= days:
            return True
        else:
            return False
    else:
        return False

# Streamlit app setup
st.header('Stock Market Price Predictor')
st.subheader('Predicts the future stock prices')

# Input for stock symbol
stock = st.text_input('Enter the Stock Symbol (e.g., GOOG)', 'GOOG')

# Date range selection for historical data
st.sidebar.header('Select Date Range')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2005-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('2024-11-01'))

# Ensure start_date is before end_date
if start_date > end_date:
    st.sidebar.error('Start date must be before end date.')

# Input for number of days to predict
days_to_predict = st.number_input('Enter the number of future days to predict:', min_value=1, max_value=365, value=30)

# Fetch historical data
df = fetch_data(stock, start_date, end_date)

# Display recent data for context
st.write('Historical Stock Data (last 5 rows):')
st.dataframe(df.tail(5))

# Preprocess the data
scaled_data, scaler = preprocess_data(df)

# Prepare training data
x_train, y_train = create_training_data(scaled_data)

# Check if the model needs retraining
if datetime.datetime.now().weekday() == 5:  # 5 represents Saturday
    if not is_model_recent(model_path):
        print("Retraining model on Saturday")  # Confirm retraining
        loaded_model = retrain_model(x_train, y_train, model_path)
    else:
        print("Model has been updated recently. No retraining needed.")
else:
    try:
        loaded_model = load_model(model_path)
        print("Model loaded successfully.")
    except:
        print("No existing model found. Training a new model.")
        loaded_model = retrain_model(x_train, y_train, model_path)

# Prepare input for predictions using the last 100 days
last_100_days = scaled_data[-100:].reshape(1, 100, 1)

# Predict the specified number of future days
future_predictions = []
for _ in range(days_to_predict):
    prediction = loaded_model.predict(last_100_days, verbose=0)
    future_predictions.append(prediction[0, 0])
    last_100_days = np.append(last_100_days[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

# Reverse scaling to get actual stock prices
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Prepare the dates for future predictions
future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=days_to_predict, freq='D')

# Create a DataFrame for the predictions
predicted_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Close Price': future_predictions.flatten()
})

# Display predicted stock prices
st.write(f'Predicted Stock Prices for the Next {days_to_predict} Days:')
st.dataframe(predicted_df)

# Plotting the results
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Close'], label='Historical Stock Price')
plt.plot(predicted_df['Date'], predicted_df['Predicted Close Price'], label='Predicted Stock Price', linestyle='--')
plt.title(f'{stock} Future Stock Price Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
st.pyplot(plt)
