import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


DATA_PATH = Path("data/Crude_Oil_Data.csv")
TIME_STEP = 60
EPOCHS = 50
BATCH_SIZE = 32


def create_dataset(data: np.ndarray, time_step: int = 60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place Crude_Oil_Data.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    print("Missing values:")
    print(df.isnull().sum())

    # EDA plots
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Close"], label="Actual Oil Prices")
    plt.title("Oil Prices Over Time (2000-2024)")
    plt.xlabel("Date")
    plt.ylabel("Price in USD")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    df["Month"] = df["Date"].dt.month
    monthly_avg_prices = df.groupby("Month")["Close"].mean()

    plt.figure(figsize=(12, 6))
    monthly_avg_prices.plot(kind="bar")
    plt.title("Average Oil Prices Per Month (2000-2024)")
    plt.xlabel("Month")
    plt.ylabel("Average Price in USD")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    df["Year"] = df["Date"].dt.year
    annual_avg_prices = df.groupby("Year")["Close"].mean()

    plt.figure(figsize=(12, 6))
    annual_avg_prices.plot(kind="line", marker="o")
    plt.title("Average Oil Prices Per Year (2000-2024)")
    plt.xlabel("Year")
    plt.ylabel("Average Price in USD")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    df["MA30"] = df["Close"].rolling(window=30).mean()
    df["MA100"] = df["Close"].rolling(window=100).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Close"], label="Actual Prices")
    plt.plot(df["Date"], df["MA30"], label="30-Day Moving Average")
    plt.plot(df["Date"], df["MA100"], label="100-Day Moving Average")
    plt.title("Oil Prices with Moving Averages (2000-2024)")
    plt.xlabel("Date")
    plt.ylabel("Price in USD")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    df["High-Low Difference"] = df["High"] - df["Low"]

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["High-Low Difference"], label="High-Low Difference")
    plt.title("High-Low Difference of Oil Prices Over Time (2000-2024)")
    plt.xlabel("Date")
    plt.ylabel("Difference in Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Modeling
    model_df = df[["Close"]].copy()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(model_df)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    X_train, y_train = create_dataset(train_data, TIME_STEP)
    X_test, y_test = create_dataset(test_data, TIME_STEP)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1),
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    y_pred = model.predict(X_test)

    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    plt.figure(figsize=(14, 7))
    plt.plot(y_test_rescaled, label="Actual Price")
    plt.plot(y_pred_rescaled, label="Predicted Price")
    plt.title("Oil Price Prediction using LSTM")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
