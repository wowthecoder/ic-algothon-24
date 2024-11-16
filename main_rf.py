import pandas as pd
import numpy as np
import cryptpandas as crp
import riskfolio as rp
import logging
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

# Constants
DATA_FOLDER = "./data_releases"
TEAM_NAME = "limoji"
PASSCODE = "014ls434>"
SUBMISSION_FILE = "submissions1.txt"
LATEST_RELEASE = "4507"
PASSWORD = "Zp4NnKkrC6OT0xms"

def load_and_decrypt_data(file_path, password):
    """
    Load and decrypt the dataset.
    """
    try:
        decrypted_data = crp.read_encrypted(file_path, password=password)
        logging.debug(f"Successfully decrypted data from {file_path}.")
        return decrypted_data
    except Exception as e:
        logging.error(f"Failed to decrypt data: {e}")
        raise

def clean_data(data):
    """
    Clean the dataset by handling missing, infinite, and extreme outlier values.
    """
    try:
        logging.debug("Cleaning data...")

        # Replace infinite values with NaN
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        logging.debug("Replaced infinite values with NaN.")

        # Check for columns with all NaN values
        all_nan_columns = data.columns[data.isna().all()].tolist()
        if all_nan_columns:
            logging.warning(f"Columns with all NaN values: {all_nan_columns}")
            data.drop(columns=all_nan_columns, inplace=True)
            logging.debug(f"Dropped columns with all NaN values.")

        # Replace remaining NaN with column median
        data.fillna(data.median(), inplace=True)
        logging.debug("Replaced remaining NaN with column medians.")

        # Handle outliers using IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        data = data.clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR, axis=1)
        logging.debug("Handled outliers using IQR method.")

        return data
    except Exception as e:
        logging.error(f"Data cleaning failed: {e}")
        raise

def calculate_returns(data):
    """
    Calculate percentage returns from adjusted prices.
    """
    try:
        returns = data.pct_change().dropna()
        logging.debug("Calculated percentage returns.")
        return returns
    except Exception as e:
        logging.error(f"Return calculation failed: {e}")
        raise

def validate_data(data):
    """
    Validate the dataset to ensure it doesn't contain inf or NaN values.
    """
    if data.isnull().values.any():
        logging.error("Data contains NaN values.")
        raise ValueError("Data contains NaN values.")
    if np.isinf(data.values).any():
        logging.error("Data contains infinite values.")
        raise ValueError("Data contains infinite values.")
    logging.debug("Data validation passed.")

def optimize_portfolio(returns):
    """
    Optimize portfolio weights using Riskfolio-Lib.
    """
    try:
        logging.debug("Optimizing portfolio using Riskfolio-Lib...")
        port = rp.Portfolio(returns=returns)

        # Estimate statistics
        port.assets_stats(method_mu="hist", method_cov="hist")
        logging.debug("Calculated portfolio statistics.")

        # Optimize portfolio for maximum Sharpe ratio
        weights = port.optimization(
            model="Classic",
            rm="MV",
            obj="Sharpe",
            rf=0,
            l=0
        )
        logging.debug(f"Optimized weights:\n{weights}")
        return weights
    except Exception as e:
        logging.error(f"Portfolio optimization failed: {e}")
        raise

def evaluate_performance(weights, returns):
    """
    Evaluate portfolio performance: Sharpe Ratio, Max Drawdown, Cumulative PnL.
    """
    try:
        portfolio_returns = (weights.T @ returns.T).sum()
        cumulative_pnl = portfolio_returns.cumsum()
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() != 0 else 0
        max_drawdown = (cumulative_pnl.max() - cumulative_pnl.min()) / cumulative_pnl.max() if cumulative_pnl.max() != 0 else 0

        logging.debug("Evaluated portfolio performance.")
        logging.debug(f"Sharpe Ratio: {sharpe_ratio}")
        logging.debug(f"Max Drawdown: {max_drawdown}")
        return portfolio_returns, cumulative_pnl, sharpe_ratio, max_drawdown
    except Exception as e:
        logging.error(f"Performance evaluation failed: {e}")
        raise

def save_submission(submission_dict, file_path):
    """
    Save submission to a file.
    """
    try:
        with open(file_path, "a") as file:
            file.write(f"{LATEST_RELEASE}: {submission_dict}\n")
        logging.debug(f"Submission successfully saved to {file_path}.")
    except Exception as e:
        logging.error(f"Failed to save submission: {e}")
        raise

def main():
    try:
        # File path
        latest_file_path = f"{DATA_FOLDER}/release_{LATEST_RELEASE}.crypt"

        # Step 1: Load and decrypt data
        data = load_and_decrypt_data(latest_file_path, PASSWORD)

        # Step 2: Clean the data
        data = clean_data(data)
        logging.debug(f"Data after cleaning:\n{data.head()}")

        # Step 3: Calculate returns
        returns = calculate_returns(data)
        logging.debug(f"Returns after cleaning:\n{returns.head()}")

        # Step 4: Validate returns data
        validate_data(returns)

        # Step 5: Optimize portfolio
        weights = optimize_portfolio(returns)

        # Step 6: Evaluate performance
        portfolio_returns, cumulative_pnl, sharpe_ratio, max_drawdown = evaluate_performance(weights, returns)

        # Step 7: Prepare submission
        submission = {
            **weights.to_dict(),
            "team_name": TEAM_NAME,
            "passcode": PASSCODE,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }
        logging.debug(f"Submission:\n{submission}")

        # Step 8: Save submission
        save_submission(submission, SUBMISSION_FILE)
    except Exception as e:
        logging.error(f"Main process failed: {e}")

if __name__ == "__main__":
    main()
