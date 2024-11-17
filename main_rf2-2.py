import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cryptpandas as crp
import tensorflow as tf
import cvxpy as cp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from cvxpy import Variable, Problem, Maximize
from sklearn.metrics import mean_squared_error

# Constants
DATA_FOLDER = "./data_releases"
TEAM_NAME = "limoji"
PASSCODE = "014ls434>"
SUBMISSION_FILE = "submissions.txt"
LATEST_RELEASE = "4635"
PASSWORD = "hpuTAsG3v5av6J0D"
MAX_EXPOSURE = 0.1  # Maximum allocation per strategy
Z_SCORE_THRESHOLD = 3
PREDICTION_STEPS = 64
ALLOCATION_CONSTRAINT = 1.0

# Helper Functions
def debug(msg):
    print(f"[DEBUG] {msg}")

def handle_error(msg):
    print(f"[ERROR] {msg}")
    raise ValueError(msg)

def validate_no_nans(data, context):
    """
    Ensures no NaN values are present in the dataset.
    """
    if isinstance(data, pd.DataFrame):
        if data.isnull().any().any():
            debug(f"NaN values detected in DataFrame after {context}:")
            print(data[data.isnull().any(axis=1)])
            handle_error(f"NaN values found in DataFrame after {context}.")
        else:
            debug(f"No NaN values detected in DataFrame after {context}.")
    elif isinstance(data, pd.Series):
        if data.isnull().any():
            debug(f"NaN values detected in Series after {context}: Total NaNs = {data.isnull().sum()}")
            handle_error(f"NaN values found in Series after {context}.")
        else:
            debug(f"No NaN values detected in Series after {context}.")
    else:
        handle_error(f"Unsupported data type for NaN validation after {context}.")

def plot_data(data, title, figsize=(12, 6)):
    """
    Plot the given data for visualization.
    """
    plt.figure(figsize=figsize)
    for column in data.columns:
        plt.plot(data.index, data[column], label=column, alpha=0.7)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Data Cleaning
def load_and_clean_data(file_path, password, near_zero_threshold=1e-6, unique_value_threshold=2):
    """
    Load and clean data, identifying and handling columns with near-zero standard deviation or low unique values.
    """
    try:
        debug("Loading and cleaning data...")
        data = crp.read_encrypted(file_path, password=password)
        debug(f"Data decrypted. Shape: {data.shape}")
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data.fillna(0.0, inplace=True)

        # Calculate standard deviation for all columns
        std_devs = data.std()
        debug("Column standard deviations (raw data):")
        print(std_devs.to_frame(name="Standard Deviation"))

        # Identify near-zero standard deviation columns
        near_zero_std = std_devs[std_devs < near_zero_threshold].index.tolist()

        # Identify columns with very few unique values
        low_unique_cols = data.columns[data.nunique() <= unique_value_threshold].tolist()

        # Combine both criteria
        repeated_columns = list(set(near_zero_std + low_unique_cols))
        debug(f"Columns identified as repeated or near-constant: {repeated_columns}")

        # Fill these columns with 0
        data[repeated_columns] = 0
        debug(f"Repeated or near-constant columns filled with 0.")

        # Normalize and Winsorize data
        data = data.apply(zscore)
        data = data.clip(lower=-Z_SCORE_THRESHOLD, upper=Z_SCORE_THRESHOLD)

        debug(f"Data cleaned and normalized. Shape: {data.shape}")
        validate_no_nans(data, "data cleaning")
        return data, repeated_columns
    except Exception as e:
        handle_error(f"Failed to load and clean data: {e}")

# Feature Engineering
def engineer_features(data, zero_std_columns, clipping_threshold=5):
    """
    Engineer features from the data, including rolling statistics, lagged features,
    and smoothing of anomalies by clipping extreme values.
    """
    try:
        debug("Engineering features...")
        features = pd.DataFrame(index=data.index)

        # Rolling statistics
        features["rolling_mean"] = data.mean(axis=1)
        features["rolling_std"] = data.std(axis=1)

        # Generate lag features
        for lag in range(1, 4):
            lagged_data = data.shift(lag)
            lagged_data.columns = [f"{col}_lag_{lag}" for col in lagged_data.columns]
            features = features.join(lagged_data)

        # Drop rows with NaN values resulting from lagging
        features.dropna(inplace=True)

        # Remove zero-std columns from features
        zero_std_features = [f"{col}_lag_{lag}" for col in zero_std_columns for lag in range(1, 4)]
        debug(f"Removing zero-std features: {zero_std_features}")
        features.drop(columns=zero_std_features, inplace=True, errors='ignore')

        # Normalize features
        scaler = StandardScaler()
        features.iloc[:, :] = scaler.fit_transform(features)

        debug(f"Features normalized. Shape: {features.shape}")
        validate_no_nans(features, "feature normalization")

        # Smoothing anomalies by clipping values
        debug(f"Clipping features to range [-{clipping_threshold}, {clipping_threshold}]")
        features = features.clip(lower=-clipping_threshold, upper=clipping_threshold)

        # Validate after smoothing
        debug(f"Features smoothed. Shape: {features.shape}")
        validate_no_nans(features, "feature smoothing")

        return features
    except Exception as e:
        handle_error(f"Feature engineering failed: {e}")

# Machine Learning: LSTM Model
def create_lstm_model(input_shape):
    """
    Create and compile an LSTM model for multi-step forecasting.
    """
    model = Sequential([
        tf.keras.layers.Input(shape=input_shape),
        LSTM(50, activation='relu', return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(input_shape[1])  # Predict all features
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_model(data):
    """
    Train the LSTM model using the input data.
    """
    debug("Preparing data for LSTM...")
    X, y = [], []
    for i in range(len(data) - PREDICTION_STEPS):
        X.append(data[i:i+PREDICTION_STEPS])
        y.append(data[i+PREDICTION_STEPS])  # Predict the next step's full feature set

    X, y = np.array(X), np.array(y)
    debug(f"Input shape for LSTM: {X.shape}, Target shape: {y.shape}")

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and train the model
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1, validation_data=(X_test, y_test))
    return model

def forecast_lstm(model, data, feature_names):
    """
    Use the LSTM model to forecast the next steps.
    """
    debug("Generating LSTM forecasts...")
    input_data = data[-PREDICTION_STEPS:].reshape(1, PREDICTION_STEPS, data.shape[1])
    predictions = model.predict(input_data)
    predictions_df = pd.DataFrame(predictions, columns=feature_names)
    debug(f"LSTM predictions shape: {predictions_df.shape}")
    return predictions_df

# Portfolio Optimization
def optimize_portfolio(data):
    """
    Optimize portfolio weights to maximize returns with a risk constraint.
    """
    try:
        debug("Optimizing portfolio weights...")
        n = data.shape[1]  # Number of strategies

        # Check if sufficient rows are present for covariance calculation
        if data.shape[0] < 2:
            debug(f"Insufficient rows in data: {data.shape[0]}. Replicating data for covariance estimation.")
            data = np.tile(data, (10, 1))  # Replicate data if insufficient rows (arbitrary 10 rows)

        # Variables for positive and negative weights
        weights_pos = cp.Variable(n)  # Positive weights
        weights_neg = cp.Variable(n)  # Negative weights

        # Portfolio weights are the difference of positive and negative parts
        weights = weights_pos - weights_neg

        # Calculate expected returns and covariance matrix
        expected_returns = np.mean(data, axis=0)  # Average returns of each strategy
        cov_matrix = np.cov(data.T)  # Covariance matrix of returns

        debug(f"Expected returns shape: {expected_returns.shape}")
        debug(f"Covariance matrix shape: {cov_matrix.shape}")

        # Ensure covariance matrix is positive semi-definite
        if cov_matrix.shape[0] != n or cov_matrix.shape[1] != n:
            handle_error(f"Covariance matrix dimensions mismatch: {cov_matrix.shape}")

        # Regularize covariance matrix to ensure numerical stability
        cov_matrix += np.eye(n) * 1e-6

        # Define risk (variance) and return
        portfolio_risk = cp.quad_form(weights, cov_matrix)  # Portfolio risk (variance)
        portfolio_return = weights @ expected_returns  # Portfolio return

        # Define optimization problem: Maximize return for a given risk threshold
        risk_threshold = 0.02  # Adjust this threshold as needed
        objective = cp.Maximize(portfolio_return)
        constraints = [
            weights_pos >= 0,  # Positive weights must be non-negative
            weights_neg >= 0,  # Negative weights must be non-negative
            cp.sum(weights_pos + weights_neg) == ALLOCATION_CONSTRAINT,  # Absolute sum of weights equals 1
            weights <= MAX_EXPOSURE,  # Max exposure to any single strategy
            weights >= -MAX_EXPOSURE,  # Min exposure to any single strategy
            portfolio_risk <= risk_threshold  # Risk constraint
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if weights.value is None:
            handle_error("Optimization failed to find a solution.")

        debug(f"Optimal weights: {weights.value}")
        return weights.value
    except Exception as e:
        handle_error(f"Portfolio optimization failed: {e}")



def evaluate_performance(data, weights):
    """
    Evaluate portfolio performance.
    """
    portfolio_returns = data @ weights
    cumulative_pnl = np.cumsum(portfolio_returns)
    sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
    debug(f"Sharpe Ratio: {sharpe_ratio}, Final Cumulative PnL: {cumulative_pnl[-1]}")
    return portfolio_returns, cumulative_pnl

def plot_allocation(weights):
    """
    Plot the allocation of portfolio weights as a pie chart.
    """
    abs_weights = np.abs(weights)
    labels = [f"{i}" if w >= 0 else f"{i} (short)" for i, w in enumerate(weights)]
    colors = ["green" if w > 0 else "red" for w in weights]

    plt.figure(figsize=(8, 8))
    plt.pie(abs_weights, labels=labels, autopct="%.2f%%", startangle=90, colors=colors)
    plt.title("Portfolio Allocation (Green = Long, Red = Short)")
    plt.show()

def plot_pnl(portfolio_returns, cumulative_pnl):
    """
    Plot per-period PnL and cumulative PnL.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_pnl, label="Cumulative PnL")
    plt.title("Portfolio Cumulative PnL")
    plt.xlabel("Time")
    plt.ylabel("PnL")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_returns, label="Per-Period PnL")
    plt.title("Portfolio Per-Period PnL")
    plt.xlabel("Time")
    plt.ylabel("PnL")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(predictions, historical_data):
    """
    Plot LSTM predictions against historical data.
    """
    plt.figure(figsize=(12, 6))
    for i in range(min(10, predictions.shape[1])):  # Plot the first 10 features (strategies)
        plt.plot(historical_data[-100:, i], label=f"Strategy {i} (Historical)")
        plt.plot(range(len(historical_data) - 100, len(historical_data) - 100 + predictions.shape[0]),
                 predictions[:, i], linestyle="--", label=f"Strategy {i} (Prediction)")

    plt.title("LSTM Predictions vs Historical Data")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.show()


# Main Function
def main():
    try:
        # Load and clean data
        file_path = f"{DATA_FOLDER}/release_{LATEST_RELEASE}.crypt"
        data, zero_std_columns = load_and_clean_data(file_path, PASSWORD)

        # Engineer features
        features = engineer_features(data, zero_std_columns, clipping_threshold=7)

        # Train LSTM model
        lstm_model = train_lstm_model(features.values)

        # Forecast with LSTM
        lstm_predictions = forecast_lstm(lstm_model, features.values, features.columns)

        # Ensure predictions are reshaped and combine with historical data
        lstm_predictions = lstm_predictions.values
        if lstm_predictions.shape[0] == 1:
            lstm_predictions = np.vstack((features.values[-10:], lstm_predictions))  # Append last 10 rows of historical data

        # Optimize portfolio
        weights = optimize_portfolio(lstm_predictions)

        # Evaluate performance
        portfolio_returns, cumulative_pnl = evaluate_performance(features.values, weights)

        # Plot results
        plot_allocation(weights)
        plot_pnl(portfolio_returns, cumulative_pnl)
        plot_predictions(lstm_predictions, features.values)
    except Exception as e:
        handle_error(f"Main process failed: {e}")


if __name__ == "__main__":
    main()
