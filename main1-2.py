import pandas as pd
import numpy as np
import cryptpandas as crp
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from scipy.optimize import minimize

# Constants
DATA_FOLDER = "./data_releases"
TEAM_NAME = "limoji"
PASSCODE = "014ls434>"
SUBMISSION_FILE = "submissions1.txt"
LATEST_RELEASE = "4059"
PASSWORD = "HkpYXpKituGernrk"

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
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(0.0)
    print(f"[DEBUG] Data after cleaning:\n{data}")
    return data

def scale_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(input_shape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_returns(data, model, scaler, time_steps=10):
    scaled_data, _ = scale_data(data)
    X = []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i])
    X = np.array(X)
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return pd.DataFrame(predictions, columns=data.columns)

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = returns / std
    return -sharpe_ratio

def optimize_portfolio(forecasted_returns):
    mean_returns = forecasted_returns.mean()
    cov_matrix = forecasted_returns.cov()
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1})
    bounds = tuple((-0.1, 0.1) for asset in range(num_assets))
    result = minimize(calculate_portfolio_performance, num_assets*[1./num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def validate_constraints(weights):
    abs_sum = round(np.sum(np.abs(weights)), 8)
    max_abs_position = np.max(np.abs(weights))
    abs_sum_ok = abs_sum == 1.0
    max_abs_ok = max_abs_position <= 0.1
    print(f"[DEBUG] Validation results: abs_sum_ok={abs_sum_ok}, max_abs_ok={max_abs_ok}")
    if not abs_sum_ok:
        print("[ERROR] The absolute sum of weights does not equal 1.0.")
    if not max_abs_ok:
        print("[ERROR] One or more weights exceed the maximum allowed ±0.1.")
    return abs_sum_ok, max_abs_ok

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

        # Clean data
        data = clean_data(data)

        # Scale data and train a forecasting model
        scaled_data, scaler = scale_data(data)
        time_steps = 10
        lstm_model = build_lstm_model((time_steps, data.shape[1]))

        # Train LSTM model (assuming historical data is available)
        # For a full implementation, include historical data loading here
        X_train, y_train = scaled_data[:-1], scaled_data[1:]
        lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)

        # Forecast next period's returns
        forecasted_returns = forecast_returns(data, lstm_model, scaler, time_steps=time_steps)
        print(f"[DEBUG] Forecasted Returns:\n{forecasted_returns}")

        # Optimize portfolio using forecasted data
        optimized_weights = optimize_portfolio(forecasted_returns)
        print(f"[DEBUG] Optimized Weights:\n{optimized_weights}")

        # Validate constraints
        abs_sum_ok, max_abs_ok = validate_constraints(optimized_weights)
        if not abs_sum_ok or not max_abs_ok:
            raise ValueError(
                f"Validation failed: abs_sum_ok={abs_sum_ok}, max_abs_ok={max_abs_ok}. "
                f"Check the weights: {optimized_weights}"
            )

        # Prepare submission dictionary
        submission = {
            **{f"strat_{i}": weight for i, weight in enumerate(optimized_weights)},
            "team_name": TEAM_NAME,
            "passcode": PASSCODE,
        }

        # Output results to terminal
        print(f"Submission: {submission}")
        print(f"Constraints - Abs Sum = 1.0: {abs_sum_ok}, Max Abs Position <= 0.1: {max_abs_ok}")

        # Save submission to file
        save_submission(submission, SUBMISSION_FILE)

    except Exception as e:
        print(f"[ERROR] Main process failed: {e}")

if __name__ == "__main__":
    main()



