import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cryptpandas as crp
import cvxpy as cp
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# Constants
DATA_FOLDER = "./data_releases"
TEAM_NAME = "limoji"
PASSCODE = "014ls434>"
SUBMISSION_FILE = "submissions.txt"
LATEST_RELEASE = "4635"
PASSWORD = "hpuTAsG3v5av6J0D"
MAX_EXPOSURE = 0.1  # Maximum allocation per strategy
Z_SCORE_THRESHOLD = 3
ALLOCATION_CONSTRAINT = 1.0
PREDICTION_STEPS = 64

# Helper Functions
def debug(msg):
    print(f"[DEBUG] {msg}")

def handle_error(msg):
    print(f"[ERROR] {msg}")
    raise ValueError(msg)

def validate_no_nans(data, context):
    if data.isnull().any().any():
        debug(f"NaN values detected in DataFrame after {context}:")
        handle_error(f"NaN values found in DataFrame after {context}.")
    else:
        debug(f"No NaN values detected in DataFrame after {context}.")

# Load and Clean Data
def load_and_clean_data(file_path, password):
    try:
        debug("Loading and cleaning data...")
        data = crp.read_encrypted(file_path, password=password)
        debug(f"Data decrypted. Shape: {data.shape}")
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data.fillna(0.0, inplace=True)

        data = data.apply(zscore).clip(lower=-Z_SCORE_THRESHOLD, upper=Z_SCORE_THRESHOLD)
        debug(f"Data cleaned and normalized. Shape: {data.shape}")
        validate_no_nans(data, "data cleaning")
        return data
    except Exception as e:
        handle_error(f"Failed to load and clean data: {e}")

# Feature Engineering
def engineer_features(data):
    try:
        debug("Engineering features...")
        features = pd.DataFrame(index=data.index)

        # Momentum features
        features["rolling_mean"] = data.mean(axis=1)
        features["rolling_std"] = data.std(axis=1)
        for lag in range(1, 4):
            lagged = data.shift(lag)
            lagged.columns = [f"{col}_lag_{lag}" for col in lagged.columns]
            features = features.join(lagged)
        features.dropna(inplace=True)

        # Sharpe Ratio features
        rolling_returns = data.diff().mean(axis=1)
        rolling_volatility = data.std(axis=1)
        features["rolling_sharpe"] = rolling_returns / rolling_volatility

        # Scaling features
        scaler = StandardScaler()
        features.iloc[:, :] = scaler.fit_transform(features)

        debug(f"Features engineered. Shape: {features.shape}")
        validate_no_nans(features, "feature engineering")
        return features
    except Exception as e:
        handle_error(f"Feature engineering failed: {e}")

def train_xgboost(features, target):
    """
    Train an XGBoost model with features and target.
    """
    try:
        debug("Training XGBoost model...")
        debug(f"Features shape: {features.shape}, Target shape: {target.shape}")
        if len(features) != len(target):
            handle_error("Mismatch between features and target lengths!")
        
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(features, target)
        return model
    except Exception as e:
        handle_error(f"XGBoost training failed: {e}")


# Forecast Future Data
def generate_forecasts(model, features):
    debug("Generating forecasts with XGBoost...")
    return pd.DataFrame(model.predict(features[-PREDICTION_STEPS:]), columns=features.columns)

# Optimize Portfolio with Sharpe Ratio
def optimize_portfolio(data):
    try:
        debug("Optimizing portfolio weights with Sharpe Ratio...")
        n = data.shape[1]
        expected_returns = np.mean(data, axis=0)
        cov_matrix = np.cov(data.T) + np.eye(n) * 1e-6

        weights = cp.Variable(n)
        portfolio_return = expected_returns @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix)
        sharpe_ratio = portfolio_return / cp.sqrt(portfolio_risk)

        objective = cp.Maximize(sharpe_ratio)
        constraints = [
            cp.sum(cp.abs(weights)) == ALLOCATION_CONSTRAINT,
            weights >= -MAX_EXPOSURE,
            weights <= MAX_EXPOSURE
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if weights.value is None:
            handle_error("Optimization failed.")
        debug(f"Optimal weights: {weights.value}")
        return weights.value
    except Exception as e:
        handle_error(f"Portfolio optimization failed: {e}")

# Evaluate and Plot Results
def evaluate_performance(data, weights):
    portfolio_returns = data @ weights
    cumulative_pnl = np.cumsum(portfolio_returns)
    sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
    debug(f"Sharpe Ratio: {sharpe_ratio}, Final Cumulative PnL: {cumulative_pnl[-1]}")
    return portfolio_returns, cumulative_pnl, sharpe_ratio

def plot_allocation(weights):
    plt.figure(figsize=(8, 8))
    abs_weights = np.abs(weights)
    labels = [f"Strategy {i}" for i in range(len(weights))]
    plt.pie(abs_weights, labels=labels, autopct="%.2f%%", startangle=90)
    plt.title("Portfolio Allocation")
    plt.show()

def plot_pnl(portfolio_returns, cumulative_pnl):
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_pnl, label="Cumulative PnL")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL")
    plt.title("Portfolio Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main Function
def main():
    try:
        # Load and preprocess data
        file_path = f"{DATA_FOLDER}/release_{LATEST_RELEASE}.crypt"
        data = load_and_clean_data(file_path, PASSWORD)

        # Feature engineering
        features = engineer_features(data)

        # Slice features and targets correctly
        features_train = features[:-PREDICTION_STEPS]
        target_train = data.iloc[PREDICTION_STEPS:len(features)]

        # Validate alignment
        debug(f"Training Features Shape: {features_train.shape}, Training Target Shape: {target_train.shape}")

        # Train model and generate forecasts
        model = train_xgboost(features_train, target_train)
        forecasts = generate_forecasts(model, features[-PREDICTION_STEPS:])

        # Optimize portfolio
        weights = optimize_portfolio(forecasts)

        # Evaluate performance
        portfolio_returns, cumulative_pnl, sharpe_ratio = evaluate_performance(data.iloc[-PREDICTION_STEPS:], weights)

        # Plot results
        plot_allocation(weights)
        plot_pnl(portfolio_returns, cumulative_pnl)
        debug(f"Final Sharpe Ratio: {sharpe_ratio}")
    except Exception as e:
        handle_error(f"Main process failed: {e}")


if __name__ == "__main__":
    main()
