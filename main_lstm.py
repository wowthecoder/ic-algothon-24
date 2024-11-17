import pandas as pd
import numpy as np
import cryptpandas as crp
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, TimeDistributed
from scipy.optimize import minimize

# Constants
DATA_FOLDER = "./data_releases"
TEAM_NAME = "limoji"
PASSCODE = "014ls434>"
SUBMISSION_FILE = "submissions1.txt"
LATEST_RELEASE = "6427"
PASSWORD = "knPxrnUGohNFbMW0"
TIME_STEPS = 64

def process_data(file_path, password):
    try:
        decrypted_data = crp.read_encrypted(file_path, password=password)
        print(f"[DEBUG] Successfully decrypted data from {file_path}.")
        return decrypted_data
    except Exception as e:
        print(f"[ERROR] Failed to process data from {file_path}: {e}")
        raise

def clean_data(data):
    print("[DEBUG] Cleaning data...")
    data.replace([np.inf, -np.inf], 0, inplace=True)
    data.fillna(0, inplace=True)
    print(f"[DEBUG] Data after cleaning:\n{data}")
    return data

def scale_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def build_lstm_model(input_shape):
    """
    Build an LSTM model for time-series forecasting.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100, return_sequences=True))
    model.add(TimeDistributed(Dense(input_shape[1])))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_lstm_input(data, time_steps):
    """
    Prepare data for LSTM input by creating sequences of `time_steps`.
    """
    X, y = [], []
    for i in range(time_steps, len(data)-time_steps):
        X.append(data[i-time_steps:i])
        y.append(data[i:i+time_steps])
    return np.array(X), np.array(y)

def forecast_returns(data, model, scaler, time_steps=TIME_STEPS):
    scaled_data, _ = scale_data(data)
    X = []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
    X = np.array(X)
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return pd.DataFrame(predictions, columns=data.columns)

def mean_reversion_strategy(data):
    mean_returns = data.mean()
    strategy_weights = -mean_returns / mean_returns.abs().sum()  # Invert mean for reversion
    return normalize_weights(strategy_weights)

def momentum_strategy(data):
    mean_returns = data.mean()
    strategy_weights = mean_returns / mean_returns.abs().sum()  # Favors momentum
    return normalize_weights(strategy_weights)

def lstm_strategy(data, lstm_model, scaler, time_steps):
    forecasted_returns = forecast_returns(data, lstm_model, scaler, time_steps)
    print("[DEBUG]Forecasted return is:\n", forecasted_returns)
    return normalize_weights(forecasted_returns)

def equal_weight_strategy(data):
    num_assets = data.shape[1]
    strategy_weights = np.ones(num_assets) / num_assets
    return normalize_weights(strategy_weights)

def normalize_weights(weights):
    """
    Normalize weights to ensure abs(sum(weights)) = 1.0 while keeping all weights within [-0.1, 0.1].
    """
    weights = np.clip(weights, -0.1, 0.1)
    abs_sum = np.sum(np.abs(weights))
    if abs_sum > 0:
        weights /= abs_sum
    weights /= np.sum(np.abs(weights))
    return weights

def calculate_pnl(weights, returns):
    portfolio_returns = np.dot(returns, weights)
    cumulative_pnl = np.nancumsum(portfolio_returns)
    sharpe_ratio = np.nanmean(portfolio_returns) / np.nanstd(portfolio_returns) if np.nanstd(portfolio_returns) != 0 else 0
    max_drawdown = np.nanmax(np.maximum.accumulate(cumulative_pnl) - cumulative_pnl)
    return portfolio_returns, cumulative_pnl, sharpe_ratio, max_drawdown

def compare_strategies(data, lstm_model, scaler, time_steps):
    strategies = {
        "Mean Reversion": mean_reversion_strategy(data),
        "Momentum": momentum_strategy(data),
        "LSTM Forecasting": lstm_strategy(data, lstm_model, scaler, time_steps),
        "Equal Weight": equal_weight_strategy(data)
    }
    results = {}
    returns = data.pct_change().iloc[1:].fillna(0)  # Daily returns
    for name, weights in strategies.items():
        pnl, cum_pnl, sharpe, drawdown = calculate_pnl(weights, returns)
        results[name] = {
            "PnL": pnl,
            "Cumulative PnL": cum_pnl,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": drawdown,
            "Weights": weights
        }
    return results

def save_submission(submission_dict, file_path):
    try:
        with open(file_path, "a") as file:
            file.write(f"{LATEST_RELEASE}: {submission_dict}\n")
        print(f"[DEBUG] Submission successfully saved to {file_path}.")
    except Exception as e:
        print(f"[ERROR] Failed to save submission: {e}")
        raise

def main():
    latest_file_path = f"{DATA_FOLDER}/release_{LATEST_RELEASE}.crypt"
    try:
        # Load and process data
        data = process_data(latest_file_path, PASSWORD)
        data = clean_data(data)
        scaled_data, scaler = scale_data(data)
        X_train, y_train = create_lstm_input(scaled_data, TIME_STEPS)
        print("Data shape is:", data.shape)
        print("y_train shape is:", y_train.shape)

        # Train LSTM model
        input_shape = (TIME_STEPS, data.shape[1])
        lstm_model = build_lstm_model(input_shape)
        lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)

        # Compare strategies
        results = compare_strategies(data, lstm_model, scaler, TIME_STEPS)
        best_strategy = max(results, key=lambda x: results[x]["Sharpe Ratio"])
        best_weights = results[best_strategy]["Weights"]

        print(f"[DEBUG] Best Strategy: {best_strategy}")
        print(f"[DEBUG] Best Weights:\n{best_weights}")

        # Prepare submission
        submission = {
            **{f"strat_{i}": weight for i, weight in enumerate(best_weights)},
            "team_name": TEAM_NAME,
            "passcode": PASSCODE,
            "best_strategy": best_strategy
        }
        save_submission(submission, SUBMISSION_FILE)

    except Exception as e:
        print(f"[ERROR] Main process failed: {e}")

if __name__ == "__main__":
    main()
