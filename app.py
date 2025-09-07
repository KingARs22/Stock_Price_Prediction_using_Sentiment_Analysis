import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import os
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
API_KEY = os.getenv("NEWSAPI_KEY")
if not API_KEY:
    raise ValueError("Please set NEWSAPI_KEY environment variable")

analyzer = SentimentIntensityAnalyzer()
SEQ_LENGTH = 60

# --- Helper functions ---
def fetch_news(query, start_date, end_date):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 100,
        "apiKey": API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data.get("articles", [])

def compute_sentiment(articles):
    if not articles:
        return 0
    scores = [analyzer.polarity_scores(a["title"])["compound"] for a in articles if a.get("title")]
    return np.mean(scores) if scores else 0

def get_latest_sentiment(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    articles = fetch_news(f"{ticker} stock", start_date, end_date)
    sentiment_score = compute_sentiment(articles)
    return sentiment_score

def adjust_with_sentiment(pred_price, sentiment_score, alpha):
    return pred_price * (1 + alpha * sentiment_score)

def create_sequences(data, seq_length=SEQ_LENGTH):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def rescale_prediction(pred_scaled, scaler):
    return scaler.inverse_transform(np.array([[pred_scaled, 0]]))[0][0]

# --- Streamlit UI ---
st.title("Stock Price Prediction with Sentiment Adjustment")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
alpha = st.slider("Sentiment Sensitivity (α)", min_value=0.0, max_value=0.5, value=0.1, step=0.01)

if ticker:
    model_filename = f"{ticker}_lstm.h5"
    
    # Fetch historical stock data
    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)
    stock_data = yf.download(ticker, start=start, end=end)
    stock_data.reset_index(inplace=True)
    
    st.write(f"Showing last 5 rows for {ticker}")
    st.dataframe(stock_data.tail())
    
    # Prepare features for LSTM
    features_for_training = stock_data[['Close', 'Volume']].values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_for_training)
    X, y = create_sequences(scaled_features)
    
    # Train or load model
    if os.path.exists(model_filename):
        lstm_model = tf.keras.models.load_model(model_filename)
        st.write(f"Loaded existing model for {ticker}")
    else:
        st.write(f"Training new model for {ticker}...")
        lstm_model = build_lstm((X.shape[1], X.shape[2]))
        lstm_model.fit(X, y, epochs=5, batch_size=32, verbose=1)
        lstm_model.save(model_filename)
        st.write("Model trained and saved.")
    
    # Predict all historical sequences
    pred_scaled_all = lstm_model.predict(X)
    pred_prices_all = [rescale_prediction(ps[0], scaler) for ps in pred_scaled_all]
    
    stock_data_seq = stock_data[SEQ_LENGTH:].copy()
    stock_data_seq['Predicted_Close'] = pred_prices_all
    
    # Predict next day
    X_latest = X[-1].reshape(1, X.shape[1], X.shape[2])
    pred_scaled_next = lstm_model.predict(X_latest)[0][0]
    pred_price_next = rescale_prediction(pred_scaled_next, scaler)
    
    latest_sentiment = get_latest_sentiment(ticker)
    adjusted_price_next = adjust_with_sentiment(pred_price_next, latest_sentiment, alpha)
    
    # Display predictions
    st.subheader(f"Predicted Next Close Price for {ticker}")
    st.write(f"Raw LSTM Prediction: {pred_price_next:.2f}")
    st.write(f"Sentiment-Adjusted Prediction: {adjusted_price_next:.2f}")
    st.write(f"Latest Sentiment Score (last 30 days): {latest_sentiment:.3f}")
    
    # Plot historical vs predicted + next-day prediction
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(stock_data['Date'], stock_data['Close'], label='Historical Close')
    ax.plot(stock_data_seq['Date'], stock_data_seq['Predicted_Close'], label='Predicted Close', color='orange')
    ax.scatter(stock_data['Date'].iloc[-1] + pd.Timedelta(days=1), adjusted_price_next,
               color='red', label='Next Predicted Close', s=100)
    ax.set_title(f"{ticker} Closing Prices vs LSTM Predictions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    st.pyplot(fig)
