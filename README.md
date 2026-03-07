# Oil Price Prediction with LSTM

A deep learning project for forecasting crude oil closing prices using an LSTM (Long Short-Term Memory) network on historical data from 2000 to 2024.

## Example Prediction Result

![Oil Price Prediction](results/prediction_plot.png)


## Project Overview

This project explores how recurrent neural networks can be applied to financial time-series forecasting. It starts with exploratory data analysis (EDA) to understand oil price behavior over time, then builds an LSTM model to predict future closing prices based on previous observations.

## Problem Statement

Forecasting oil prices is challenging because prices are affected by market volatility, geopolitical events, supply-demand shifts, and macroeconomic conditions. This project focuses on a simplified univariate forecasting setup where only the historical `Close` price is used to predict the next value.

## Dataset

The project expects a CSV file named `Crude_Oil_Data.csv` inside the `data/` folder.

Required columns:
- `Date`
- `Close`
- `High`
- `Low`

Example path:
```bash
data/Crude_Oil_Data.csv
```

## Project Structure

```text
oil-price-lstm-github-project/
├── data/
│   └── README.md
├── notebooks/
│   └── oil_price_prediction_lstm.ipynb
├── results/
│   └── README.md
├── src/
│   └── train_lstm.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Workflow

1. Load and inspect the dataset
2. Convert `Date` to datetime format
3. Perform EDA:
   - Price trend over time
   - Monthly average prices
   - Yearly average prices
   - Moving averages
   - High-Low price spread
4. Scale the `Close` price using MinMaxScaler
5. Create rolling sequences for LSTM input
6. Train an LSTM model
7. Evaluate with:
   - MSE
   - RMSE
   - R²
8. Plot actual vs predicted prices

## Model

The forecasting model uses:
- LSTM layer (50 units, return sequences = True)
- Dropout (0.2)
- LSTM layer (50 units)
- Dropout (0.2)
- Dense output layer (1 unit)

Loss function:
- Mean Squared Error

Optimizer:
- Adam

## How to Run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Add the dataset

Place your dataset in:

```bash
data/Crude_Oil_Data.csv
```

### 3) Run the training script

```bash
python src/train_lstm.py
```

Or open the notebook:

```bash
notebooks/oil_price_prediction_lstm.ipynb
```

## Notes

- This is a **univariate** forecasting approach using only the closing price.
- Results can be improved by adding external features such as:
  - news sentiment
  - macroeconomic indicators
  - political events
  - supply/demand variables

## Future Improvements

- Add multivariate features
- Perform hyperparameter tuning
- Use walk-forward validation
- Compare LSTM with GRU, XGBoost, and Prophet
- Save trained models and prediction plots automatically

## Why this project is useful for GitHub

This repository demonstrates:
- time-series preprocessing
- exploratory data analysis
- deep learning with LSTM
- regression evaluation
- project organization and reproducibility

It is a strong portfolio project for data science, machine learning, and AI roles.
