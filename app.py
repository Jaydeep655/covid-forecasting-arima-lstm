"""
COVID-19 Case Forecasting — ARIMA vs LSTM
------------------------------------------
Pulls (or loads) a country's daily COVID-19 case time series, fits both an
ARIMA model and an LSTM, and lets the user compare forecasts interactively.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Data source (auto-downloaded): Our World in Data COVID-19 dataset
https://covid.ourworldindata.org/data/owid-covid-data.csv
(If offline / rate-limited, drop a local copy as `owid-covid-data.csv` in this folder.)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="COVID-19 Forecaster", layout="wide")

DATA_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
LOCAL_CACHE = "owid-covid-data.csv"


@st.cache_data
def load_data():
    if os.path.exists(LOCAL_CACHE):
        return pd.read_csv(LOCAL_CACHE, parse_dates=["date"])
    df = pd.read_csv(DATA_URL, parse_dates=["date"])
    df.to_csv(LOCAL_CACHE, index=False)
    return df


def fit_arima(series, order=(5, 1, 0), horizon=14):
    model = ARIMA(series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=horizon)
    return forecast


def make_sequences(data, window=14):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)


def fit_lstm(series_scaled, window=14, horizon=14, epochs=15):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    X, y = make_sequences(series_scaled, window)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation="relu", input_shape=(window, 1)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, verbose=0)

    last_window = series_scaled[-window:].reshape((1, window, 1))
    preds = []
    for _ in range(horizon):
        next_val = model.predict(last_window, verbose=0)[0, 0]
        preds.append(next_val)
        last_window = np.append(last_window[:, 1:, :], [[[next_val]]], axis=1)
    return np.array(preds)


st.title("🦠 COVID-19 Case Forecasting: ARIMA vs LSTM")
st.caption("Compare a classical statistical model against a deep-learning sequence model on real case data.")

df = load_data()
countries = sorted(df["location"].dropna().unique())
default_idx = countries.index("India") if "India" in countries else 0
country = st.selectbox("Country", countries, index=default_idx)
horizon = st.slider("Forecast horizon (days)", 7, 30, 14)

country_df = df[df["location"] == country][["date", "new_cases"]].dropna()
country_df = country_df.set_index("date").asfreq("D").fillna(method="ffill")
series = country_df["new_cases"].clip(lower=0)

st.line_chart(series.tail(180), height=250)

if st.button("Run forecasts", type="primary"):
    with st.spinner("Fitting ARIMA..."):
        arima_forecast = fit_arima(series, horizon=horizon)

    with st.spinner("Training LSTM (this simulates the notebook pipeline, ~10-20s)..."):
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        lstm_forecast_scaled = fit_lstm(series_scaled, horizon=horizon)
        lstm_forecast = scaler.inverse_transform(lstm_forecast_scaled.reshape(-1, 1)).flatten()

    future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)
    result_df = pd.DataFrame({
        "ARIMA": arima_forecast.values,
        "LSTM": lstm_forecast,
    }, index=future_dates)

    st.subheader(f"{horizon}-day forecast comparison")
    st.line_chart(result_df)
    st.dataframe(result_df.style.format("{:.0f}"))

    st.info(
        "Note: forecast accuracy on raw new-case counts is noisy due to reporting lags; "
        "in the write-up, report MAE/RMSE against a held-out tail window for credibility, "
        "not just the forecast plot."
    )
