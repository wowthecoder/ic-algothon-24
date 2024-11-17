import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cryptpandas as crp
import cvxpy as cp
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# Constants
DATA_FOLDER = "./data_releases"
TEAM_NAME = "limoji"
PASSCODE = "014ls434>"
SUBMISSION_FILE = "submissions.txt"
LATEST_RELEASE = "4635"
PASSWORD = "hpuTAsG3v5av6J0D"
MAX_EXPOSURE = 0.1  # Maximum allowed allocation to any single strategy
Z_SCORE_THRESHOLD = 3  # Threshold for Z-score-based outlier detection

def debug(msg):
    print(f"[DEBUG] {msg}")

def handle_error(msg):
    print(f"[ERROR] {msg}")
    raise ValueError(msg)

def validate_no_nans(data, context):
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
            print(data[data.isnull()])
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


def main():
    try:
        file_path = f"{DATA_FOLDER}/release_{LATEST_RELEASE}.crypt"
        data, zero_std_columns = load_and_clean_data(file_path, PASSWORD)

        debug("Inspecting loaded data:")
        print(data.head())

        # Plot raw loaded data
        plot_data(data, title="Raw Loaded Data")

        # Engineer features with smoothing applied
        features = engineer_features(data, zero_std_columns, clipping_threshold=7)

        debug("Inspecting smoothed and engineered features:")
        print(features.head())

        # Plot feature-engineered and smoothed data
        plot_data(features, title="Feature-Engineered and Smoothed Data")

        # Further steps for model training and optimization can be added here
    except Exception as e:
        handle_error(f"Main process failed: {e}")

if __name__ == "__main__":
    main()
