import pandas as pd
import numpy as np
import cryptpandas as crp
import riskfolio.Portfolio as pf
from riskfolio.Params import Constraints

# Constants
DATA_FOLDER = "./data_releases"
TEAM_NAME = "limoji"
PASSCODE = "014ls434>"
SUBMISSION_FILE = "submissions1.txt"
LATEST_RELEASE = "4507"
PASSWORD = "Zp4NnKkrC6OT0xms"

def process_data(file_path, password):
    """
    Load and process the latest dataset from the given file path.
    """
    try:
        decrypted_data = crp.read_encrypted(file_path, password=password)
        print(f"[DEBUG] Successfully decrypted data from {file_path}.")
        return decrypted_data
    except Exception as e:
        print(f"[ERROR] Failed to process data from {file_path}: {e}")
        raise

def clean_data(data):
    """
    Clean the dataset by handling NaN and inf values.
    """
    print("[DEBUG] Cleaning data...")
    data = data.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
    data = data.fillna(0.0)  # Replace NaN with 0.0
    print(f"[DEBUG] Data after cleaning:\n{data}")
    return data

def optimize_portfolio_riskfolio(data):
    """
    Optimize portfolio weights using Riskfolio-Lib under specified constraints.
    """
    print("[DEBUG] Optimizing portfolio using Riskfolio-Lib...")
    
    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Initialize portfolio object
    port = pf.Portfolio(returns=returns)

    # Constraints: Absolute sum = 1, weights within [-0.1, 0.1]
    constraints = Constraints()
    constraints.set_constraints(ineq=0.1, eq=1.0, kind='abs')  # Custom constraints
    
    # Set objective to maximize Sharpe Ratio
    port.assets_stats(method_mu="hist", method_cov="hist")
    weights = port.optimization(model="Classic", rm="Sharpe", constraints=constraints)
    
    print(f"[DEBUG] Optimized Weights:\n{weights}")
    return weights

def validate_constraints(weights):
    """
    Validate that the constraints are met:
    - The abs sum of the positions must be 1.0
    - The largest abs position must be <= 0.1
    """
    abs_sum = round(weights.abs().sum(), 8)
    max_abs_position = weights.abs().max()

    abs_sum_ok = abs_sum == 1.0
    max_abs_ok = max_abs_position <= 0.1

    print(f"[DEBUG] Validation results: abs_sum_ok={abs_sum_ok}, max_abs_ok={max_abs_ok}")
    if not abs_sum_ok:
        print("[ERROR] The absolute sum of weights does not equal 1.0.")
    if not max_abs_ok:
        print("[ERROR] One or more weights exceed the maximum allowed ±0.1.")
    return abs_sum_ok, max_abs_ok

def evaluate_performance(weights, data):
    """
    Evaluate portfolio performance metrics: Sharpe Ratio, Max Drawdown, Cumulative PnL.
    """
    returns = data.pct_change().dropna()
    portfolio_returns = (weights.T @ returns.T).sum()
    cumulative_pnl = portfolio_returns.cumsum()
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() != 0 else 0
    max_drawdown = (cumulative_pnl.max() - cumulative_pnl.min()) / cumulative_pnl.max() if cumulative_pnl.max() != 0 else 0

    print(f"[DEBUG] Portfolio Performance:")
    print(f"Sharpe Ratio: {sharpe_ratio}, Max Drawdown: {max_drawdown}")
    return portfolio_returns, cumulative_pnl, sharpe_ratio, max_drawdown

def save_submission(submission_dict, file_path):
    """
    Save the submission dictionary to a file in append mode.
    """
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

        # Optimize portfolio using Riskfolio-Lib
        optimized_weights = optimize_portfolio_riskfolio(data)

        # Validate constraints
        abs_sum_ok, max_abs_ok = validate_constraints(optimized_weights)
        if not abs_sum_ok or not max_abs_ok:
            raise ValueError(
                f"Validation failed: abs_sum_ok={abs_sum_ok}, max_abs_ok={max_abs_ok}. "
                f"Check the weights: {optimized_weights}"
            )

        # Evaluate performance
        portfolio_returns, cumulative_pnl, sharpe_ratio, max_drawdown = evaluate_performance(optimized_weights, data)

        # Prepare submission dictionary
        submission = {
            **{f"strat_{i}": weight for i, weight in enumerate(optimized_weights)},
            "team_name": TEAM_NAME,
            "passcode": PASSCODE,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }

        # Output results to terminal
        print(f"Submission: {submission}")

        # Save submission to file
        save_submission(submission, SUBMISSION_FILE)

    except Exception as e:
        print(f"[ERROR] Main process failed: {e}")

if __name__ == "__main__":
    main()
