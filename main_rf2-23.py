import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import cvxpy as cp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Constants
TEAM_NAME = "a_great_team_name"
PASSCODE = "a_strong_p4$$c0d3"
MAX_EXPOSURE = 0.1
ALLOCATION_CONSTRAINT = 1.0

# Helper Functions
def debug(msg):
    print(f"[DEBUG] {msg}")

def handle_error(msg):
    print(f"[ERROR] {msg}")
    raise ValueError(msg)

def validate_no_nans(data, context):
    """
    Ensure no NaN or infinite values in the data.
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

# Data Cleaning and Exploration
def load_and_clean_data(file_path):
    """
    Load and clean the dataset.
    """
    try:
        debug("Loading data...")
        data = pd.read_csv(file_path)
        debug(f"Data loaded. Shape: {data.shape}")
        
        # Handle missing values and outliers
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0, inplace=True)
        validate_no_nans(data, "data cleaning")
        
        # Normalize data (Z-score normalization)
        debug("Normalizing data...")
        data = (data - data.mean()) / data.std()
        
        # Clip extreme outliers
        data = data.clip(lower=-3, upper=3)
        return data
    except Exception as e:
        handle_error(f"Failed to load and clean data: {e}")

def explore_data(data):
    """
    Perform exploratory data analysis (EDA).
    """
    debug("Exploring data...")
    
    # Correlation matrix and heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # Distribution of returns for each strategy
    data.hist(bins=30, figsize=(15, 10), grid=False)
    plt.suptitle("Strategy Return Distributions")
    plt.show()

# Feature Engineering
def engineer_features(data):
    """
    Create lagged features and rolling metrics for each strategy.
    """
    debug("Engineering features...")
    features = pd.DataFrame(index=data.index)
    
    # Generate lagged features
    for lag in range(1, 4):
        lagged = data.shift(lag)
        lagged.columns = [f"{col}_lag_{lag}" for col in data.columns]
        features = pd.concat([features, lagged], axis=1)
    
    # Rolling metrics (mean, std)
    rolling_mean = data.rolling(window=5).mean()
    rolling_std = data.rolling(window=5).std()
    features = pd.concat([features, rolling_mean.add_suffix("_roll_mean"), rolling_std.add_suffix("_roll_std")], axis=1)
    
    # Drop rows with NaNs due to lagging or rolling
    features.dropna(inplace=True)
    debug(f"Features engineered. Shape: {features.shape}")
    return features

# Predictive Modeling
def train_xgboost_model(features, target):
    """
    Train an XGBoost regressor for predicting future returns.
    """
    debug("Training XGBoost model...")
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    # Evaluate model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    debug(f"Model evaluation: MSE={mse:.4f}, MAE={mae:.4f}")
    return model

def forecast_returns(model, features):
    """
    Forecast future returns using the trained model.
    """
    debug("Forecasting returns...")
    return model.predict(features)

# Portfolio Optimization
def optimize_portfolio(expected_returns):
    """
    Optimize portfolio weights under constraints.
    """
    debug("Optimizing portfolio weights...")
    n = len(expected_returns)
    weights = cp.Variable(n)
    
    # Objective: Maximize expected returns
    objective = cp.Maximize(expected_returns @ weights)
    
    # Constraints
    constraints = [
        cp.sum(cp.abs(weights)) == ALLOCATION_CONSTRAINT,  # Absolute sum equals 1
        weights <= MAX_EXPOSURE,  # Maximum weight per strategy
        weights >= -MAX_EXPOSURE  # Minimum weight per strategy
    ]
    
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    if weights.value is None:
        handle_error("Portfolio optimization failed.")
    
    optimized_weights = weights.value
    debug(f"Optimal weights: {optimized_weights}")
    return optimized_weights

# Submission Preparation
def prepare_submission(weights, strategies):
    """
    Prepare the submission dictionary.
    """
    weights_dict = {f"strat_{i}": w for i, w in enumerate(weights)}
    submission = {
        **weights_dict,
        "team_name": TEAM_NAME,
        "passcode": PASSCODE,
    }
    debug(f"Submission prepared: {submission}")
    return submission

# Main Function
def main():
    try:
        # Load and clean data
        file_path = "your_data.csv"  # Update with actual data file path
        data = load_and_clean_data(file_path)
        
        # Explore data
        explore_data(data)
        
        # Engineer features
        features = engineer_features(data)
        target = data.iloc[features.index]  # Align target with feature indices
        
        # Train model
        model = train_xgboost_model(features, target)
        
        # Forecast future returns
        forecasts = forecast_returns(model, features)
        
        # Optimize portfolio
        weights = optimize_portfolio(forecasts)
        
        # Prepare submission
        submission = prepare_submission(weights, data.columns)
        print(submission)
    except Exception as e:
        handle_error(f"Main process failed: {e}")

if __name__ == "__main__":
    main()
