import pandas as pd
import numpy as np
import cryptpandas as crp
import matplotlib.pyplot as plt

# Constants
DATA_FOLDER = "./data_releases"
TEAM_NAME = "limoji"
PASSCODE = "014ls434>"
SUBMISSION_FILE = "submissions.txt"
LATEST_RELEASE = "8475"
PASSWORD = "IToOiV72S4vRSmcn"
MAX_EXPOSURE = 0.1
ALLOCATION_CONSTRAINT = 1.0

# Helper Functions
def debug(msg):
    print(f"[DEBUG] {msg}")

def handle_error(msg):
    print(f"[ERROR] {msg}")
    raise ValueError(msg)

def validate_no_nans(data, context):
    if data.isnull().any().any():
        handle_error(f"NaN values found in DataFrame after {context}.")
    else:
        debug(f"No NaN values detected in DataFrame after {context}.")

def load_and_clean_data(file_path, password):
    """
    Loads encrypted data and performs basic cleaning.
    """
    debug("Loading and cleaning data...")
    try:
        data = crp.read_encrypted(file_path, password=password)
        debug(f"Data decrypted. Shape: {data.shape}")
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0.0, inplace=True)
        validate_no_nans(data, "data cleaning")
        return data
    except Exception as e:
        handle_error(f"Failed to load and clean data: {e}")

def evaluate_sharpe(data):
    """
    Evaluates Sharpe ratios for all strategies.
    """
    debug("Evaluating Sharpe ratios...")
    mean_returns = data.mean(axis=0)
    volatilities = data.std(axis=0)
    sharpe_ratios = mean_returns / volatilities
    return sharpe_ratios, mean_returns, volatilities

def allocate_positions(sharpe_ratios):
    """
    Allocates positions based on Sharpe ratios:
    - Long 0.1 on top 5 performers.
    - Short -0.1 on worst 5 performers.
    """
    debug("Allocating positions...")
    sorted_sharpe = sharpe_ratios.sort_values(ascending=False)
    top_5 = sorted_sharpe.index[:5]
    worst_5 = sorted_sharpe.index[-5:]

    weights = pd.Series(0, index=sharpe_ratios.index)
    weights[top_5] = MAX_EXPOSURE
    weights[worst_5] = -MAX_EXPOSURE

    # Validate allocation constraint
    if not np.isclose(weights.abs().sum(), ALLOCATION_CONSTRAINT, atol=1e-6):
        handle_error("Total allocation does not meet the constraint of 1.0.")
    
    debug(f"Final allocated weights: {weights}")
    return weights

def evaluate_portfolio(data, weights):
    """
    Evaluates portfolio performance: Sharpe Ratio and PnL.
    """
    debug("Evaluating portfolio performance...")
    portfolio_returns = data @ weights
    cumulative_pnl = portfolio_returns.cumsum()
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0

    debug(f"Portfolio Sharpe Ratio: {sharpe_ratio}")
    debug(f"Final Cumulative PnL: {cumulative_pnl.iloc[-1]}")
    return sharpe_ratio, cumulative_pnl, portfolio_returns

def evaluate_portfolio(data, weights):
    """
    Evaluates portfolio performance: Sharpe Ratio and PnL.
    """
    debug("Evaluating portfolio performance...")
    portfolio_returns = data @ weights
    cumulative_pnl = portfolio_returns.cumsum()
    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0

    # Print numerical results
    print(f"Final Portfolio Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Final Portfolio Cumulative PnL: {cumulative_pnl.iloc[-1]:.4f}")

    debug(f"Portfolio Sharpe Ratio: {sharpe_ratio}")
    debug(f"Final Cumulative PnL: {cumulative_pnl.iloc[-1]}")
    return sharpe_ratio, cumulative_pnl, portfolio_returns

def save_submission(weights, filename="submission.txt"):
    """
    Save submission in the required format with weights, team_name, and passcode.
    """
    debug(f"Saving submission to {filename}...")
    submission = {f"strat_{i}": weight for i, weight in enumerate(weights)}
    submission["team_name"] = TEAM_NAME
    submission["passcode"] = PASSCODE

    # Save to file
    with open(filename, "w") as f:
        f.write(str(submission))

    debug("Submission saved to submission.txt")
    # Print the submission for display
    print("Submission:", submission)

def main():
    try:
        # Load and preprocess data
        file_path = f"{DATA_FOLDER}/release_{LATEST_RELEASE}.crypt"
        data = load_and_clean_data(file_path, PASSWORD)

        # Evaluate Sharpe ratios
        sharpe_ratios, mean_returns, volatilities = evaluate_sharpe(data)

        # Allocate positions
        weights = allocate_positions(sharpe_ratios)

        # Evaluate portfolio performance
        sharpe_ratio, cumulative_pnl, portfolio_returns = evaluate_portfolio(data, weights)

        # Save submission
        save_submission(weights)

        # Print formatted numerical results
        print(f"Final Portfolio Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Final Portfolio Cumulative PnL: {cumulative_pnl.iloc[-1]:.4f}")

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_pnl, label="Cumulative PnL", color="blue")
        plt.xlabel("Time")
        plt.ylabel("PnL")
        plt.title("Portfolio Cumulative PnL")
        plt.legend()
        plt.grid(True)
        plt.savefig("cumulative_pnl.png")
        debug("Cumulative PnL plot saved as cumulative_pnl.png")

        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_returns, label="Portfolio Returns", color="green")
        plt.xlabel("Time")
        plt.ylabel("Returns")
        plt.title("Per-Period Portfolio Returns")
        plt.legend()
        plt.grid(True)
        plt.savefig("portfolio_returns.png")
        debug("Portfolio returns plot saved as portfolio_returns.png")
    except Exception as e:
        handle_error(f"Main process failed: {e}")

if __name__ == "__main__":
    main()
