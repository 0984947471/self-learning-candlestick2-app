import streamlit as st
import pandas as pd
import numpy as np
from alpha_vantage.forex import Forex
import talib as ta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Title of the app
st.title("Advanced Forex Price Movement Prediction")

# Sidebar for user input
st.sidebar.header("Input Parameters")
forex_pair = st.sidebar.text_input("Enter Forex Pair (e.g., EURUSD):", "EURUSD")

# Fetch live forex data
@st.cache_data
def fetch_forex_data(forex_pair, interval='1min', output_size='compact'):
    api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'  # Replace with your Alpha Vantage API key
    forex = Forex(key=api_key)
    data, _ = forex.get_currency_exchange_intraday(forex_pair, interval=interval, outputsize=output_size)
    data = data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close'
    })
    return data

data = fetch_forex_data(forex_pair)
data.index = pd.to_datetime(data.index)

# Add technical indicators
data['SMA'] = ta.SMA(data['Close'], timeperiod=14)
data['RSI'] = ta.RSI(data['Close'], timeperiod=14)
data['MACD'], data['MACD_signal'], data['MACD_hist'] = ta.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
data['Upper_Band'], data['Middle_Band'], data['Lower_Band'] = ta.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
data['ATR'] = ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)

# Add lag features
for lag in range(1, 4):
    data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)

# Add rolling statistics
data['Rolling_Mean'] = data['Close'].rolling(window=14).mean()
data['Rolling_Std'] = data['Close'].rolling(window=14).std()

# Add time-based features
data['Hour'] = data.index.hour
data['Day_of_Week'] = data.index.dayofweek

# Drop rows with NaN values (created by indicators and lag features)
data.dropna(inplace=True)

# Display raw data
if st.checkbox("Show Raw Data"):
    st.write(data)

# Plot candlestick chart
st.header("Candlestick Chart")
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(len(data)):
    open_price, high, low, close = data['Open'][i], data['High'][i], data['Low'][i], data['Close'][i]
    color = 'green' if close > open_price else 'red'
    ax.plot([i, i], [low, high], color=color, linewidth=1)
    ax.plot([i, i], [open_price, close], color=color, linewidth=4)
ax.set_xticks(range(0, len(data), 10))  # Show fewer x-axis labels
ax.set_xticklabels(data.index[::10].strftime('%Y-%m-%d %H:%M'), rotation=45)
st.pyplot(fig)

# Prepare data for machine learning
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)  # 1 if price goes up, 0 otherwise
data.dropna(inplace=True)

# Features and target
features = ['Open', 'High', 'Low', 'Close', 'SMA', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'Upper_Band', 'Middle_Band', 'Lower_Band', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Rolling_Mean', 'Rolling_Std', 'ATR', 'Hour', 'Day_of_Week']
X = data[features]
y = data['Target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build an LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Reshape data for LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train LSTM model
lstm_model = build_lstm_model((X_train_lstm.shape[1], 1))
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test))

# Train XGBoost model
xgb_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Ensemble model (XGBoost + LSTM)
ensemble_model = VotingClassifier(estimators=[
    ('xgb', xgb_model),
    ('lstm', lstm_model)
], voting='soft')

ensemble_model.fit(X_train, y_train)

# Predictions
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display model accuracy
st.header("Price Prediction Model")
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict next movement
last_row = data.iloc[-1][features].values.reshape(1, -1)
last_row_scaled = scaler.transform(last_row)
last_row_lstm = last_row_scaled.reshape((last_row_scaled.shape[0], last_row_scaled.shape[1], 1))
prediction = ensemble_model.predict(last_row_scaled)[0]
st.write(f"Prediction for Next Movement: {'Up' if prediction == 1 else 'Down'}")

# Feedback system
st.header("Feedback System")
feedback = st.radio("Was the prediction accurate?", ("Yes", "No"))
if feedback == "Yes":
    st.write("Thank you for your feedback! The model will use this to improve.")
elif feedback == "No":
    st.write("Thank you for your feedback! The model will retrain with new data.")

# Retrain model with feedback
if feedback == "No":
    data = fetch_forex_data(forex_pair)  # Fetch latest data
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    data.dropna(inplace=True)
    X = data[features]
    y = data['Target']
    ensemble_model.fit(X, y)  # Retrain the model
    st.write("Model retrained with latest data.")