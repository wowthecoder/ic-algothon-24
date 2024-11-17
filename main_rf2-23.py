import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cryptpandas as crp
import cvxpy as cp
from scipy.stats import zscore
import seaborn as sns

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

def debug_shape(data, context):
    debug(f"Shape of data during {context}: {data.shape}")

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

# Plot Correlation Matrix
def plot_correlation_matrix(data, filename="correlation_matrix.png"):
    debug("Plotting correlation matrix...")
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=False, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix")
    plt.savefig(filename)
    plt.close()
    debug(f"Correlation matrix saved as {filename}")

def plot_selected_correlation(correlation_matrix, weights, filename="selected_correlation.png"):
    """
    Plot the correlation matrix of the selected strategies based on non-zero weights.
    """
    selected_indices = np.nonzero(weights)[0]
    selected_correlation = correlation_matrix[np.ix_(selected_indices, selected_indices)]
    plt.figure(figsize=(8, 6))
    sns.heatmap(selected_correlation, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix of Selected Strategies")
    plt.savefig(filename)
    plt.close()
    debug(f"Selected correlation matrix saved as {filename}")

# Strategy Evaluation
def evaluate_strategies(data):
    debug("Evaluating strategies...")
    mean_returns = data.mean(axis=0).values
    volatility = data.std(axis=0).values
    sharpe_ratios = mean_returns / volatility
    debug(f"Mean returns: {mean_returns}")
    debug(f"Volatility: {volatility}")
    debug(f"Sharpe ratios: {sharpe_ratios}")
    return mean_returns, volatility, sharpe_ratios

# Optimize Portfolio
def optimize_portfolio(mean_returns, cov_matrix, risk_tolerance=0.01):
    try:
        debug("Optimizing portfolio weights...")
        n = len(mean_returns)
        
        # Positive and negative weights
        weights_pos = cp.Variable(n)
        weights_neg = cp.Variable(n)
        weights = weights_pos - weights_neg
        
        # Portfolio return and risk
        portfolio_return = mean_returns @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix)

        # Objective: Maximize return for given risk tolerance
        objective = cp.Maximize(portfolio_return - risk_tolerance * portfolio_risk)
        constraints = [
            weights_pos >= 0,  # Positive weights non-negative
            weights_neg >= 0,  # Negative weights non-negative
            cp.sum(weights_pos + weights_neg) == ALLOCATION_CONSTRAINT,  # Absolute weight constraint
            weights <= MAX_EXPOSURE,  # Max positive exposure
            weights >= -MAX_EXPOSURE,  # Max negative exposure
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
    """
    Evaluate the portfolio performance given data and weights.
    """
    try:
        debug("Evaluating portfolio performance...")
        # Compute portfolio returns
        portfolio_returns = data @ weights
        # Compute log-factor cumulative PnL
        log_returns = np.log1p(portfolio_returns)  # Logarithmic factor
        cumulative_pnl = log_returns.cumsum()

        # Calculate Sharpe Ratio
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0

        # Debug information
        if len(cumulative_pnl) > 0:
            debug(f"Portfolio Returns: {portfolio_returns}")
            debug(f"Cumulative PnL: {cumulative_pnl}")
            debug(f"Sharpe Ratio: {sharpe_ratio}, Final Cumulative PnL: {cumulative_pnl.iloc[-1]}")
        else:
            debug("Cumulative PnL is empty!")

        return portfolio_returns, cumulative_pnl, sharpe_ratio
    except Exception as e:
        handle_error(f"Performance evaluation failed: {e}")

def plot_allocation(weights, filename="allocation_pie_chart.png"):
    plt.figure(figsize=(8, 8))
    abs_weights = np.abs(weights)
    labels = [f"Strategy {i}" for i in range(len(weights))]
    plt.pie(abs_weights, labels=labels, autopct="%.2f%%", startangle=90)
    plt.title("Portfolio Allocation")
    plt.savefig(filename)
    plt.close()
    debug(f"Portfolio allocation saved as {filename}")

def plot_pnl(portfolio_returns, cumulative_pnl, filename_cumulative="cumulative_pnl.png", filename_period="period_pnl.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_pnl, label="Cumulative PnL", color="blue")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL")
    plt.title("Portfolio Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename_cumulative)
    plt.close()
    debug(f"Cumulative PnL plot saved as {filename_cumulative}")

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_returns, label="Per-Period PnL", color="green")
    plt.xlabel("Time")
    plt.ylabel("PnL")
    plt.title("Per-Period Portfolio Returns")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename_period)
    plt.close()
    debug(f"Per-period PnL plot saved as {filename_period}")

# Optimize Portfolio with Correlation Hedging
def optimize_portfolio_with_sharpe_momentum_and_hedging(
    mean_returns,
    cov_matrix,
    correlation_matrix,
    momentum_scores,
    target_return=0.01,
    risk_tolerance=0.01,
    prev_weights=None,
):
    """
    Optimize portfolio by balancing Sharpe ratio, Momentum, and Hedging constraints.
    """
    try:
        debug("Optimizing portfolio with Sharpe, Momentum, and Hedging constraints...")
        n = len(mean_returns)

        # Variables for weights
        weights = cp.Variable(n)

        # Objective Components
        portfolio_return = mean_returns @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix)

        # Reformulated Sharpe objective
        sharpe_loss = -portfolio_return + risk_tolerance * portfolio_risk

        # Momentum Loss: Penalize deviation from ideal momentum scores
        target_momentum = momentum_scores / np.sum(np.abs(momentum_scores))
        momentum_loss = cp.norm(weights - target_momentum, 1)

        # Correlation Hedging Penalty
        correlation_penalty = cp.norm(correlation_matrix @ weights, 1)

        # Transaction Cost Penalty: Penalize large deviations from previous weights
        transaction_cost = (
            cp.norm(weights - prev_weights, 1) if prev_weights is not None else 0
        )

        # Final Objective
        objective = cp.Minimize(
            sharpe_loss
            + 0.1 * momentum_loss
            + 0.01 * correlation_penalty
            + 0.005 * transaction_cost
        )

        # Constraints
        constraints = [
            weights >= -MAX_EXPOSURE,  # Allow shorting
            weights <= MAX_EXPOSURE,  # Max positive exposure
            cp.sum(cp.abs(weights)) <= ALLOCATION_CONSTRAINT,  # Absolute weight constraint
            portfolio_return >= target_return,  # Target return constraint
        ]

        # Solve Optimization Problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if weights.value is None:
            handle_error("Optimization failed.")
        debug(f"Optimized weights with hedging: {weights.value}")
        return weights.value
    except Exception as e:
        handle_error(f"Optimization with Sharpe, Momentum, and Hedging failed: {e}")

# Submission Output
def save_allocation_submission(weights, filename="submission.txt"):
    """
    Save allocation weights to a submission file with 'team_name' and 'passcode'.
    """
    debug(f"Preparing submission to {filename}...")
    non_zero_weights = {f"Strategy_{i}": weight for i, weight in enumerate(weights) if weight != 0}
    submission = {
        "team_name": TEAM_NAME,
        "passcode": PASSCODE,
        **non_zero_weights
    }
    with open(filename, "w") as f:
        for key, value in submission.items():
            f.write(f"{key}: {value}\n")
    debug(f"Submission saved to {filename}")

def calculate_momentum_scores(data):
    """
    Calculate momentum scores for each strategy based on cumulative returns.
    """
    debug("Calculating momentum scores...")
    momentum_scores = data.cumsum().iloc[-1].values  # Final cumulative return as momentum
    debug(f"Momentum scores: {momentum_scores}")
    return momentum_scores

import numpy as np

def redistribute_excess_weights(weights, max_exposure=0.1, allocation_constraint=1.0):
    """
    Adjusts portfolio weights to ensure they meet the specified constraints.

    Parameters:
    - weights: np.ndarray of portfolio weights.
    - max_exposure: Maximum allowed absolute weight for any asset.
    - allocation_constraint: Total allowed sum of absolute weights.

    Returns:
    - np.ndarray of adjusted weights.
    """
    # Step 1: Clip weights that exceed MAX_EXPOSURE
    clipped_weights = np.clip(weights, -max_exposure, max_exposure)

    # Step 2: Calculate the total absolute weight after clipping
    total_abs_weight = np.sum(np.abs(clipped_weights))

    # Step 3: If the total absolute weight exceeds the allocation constraint, scale down
    if total_abs_weight > allocation_constraint:
        scaling_factor = allocation_constraint / total_abs_weight
        adjusted_weights = clipped_weights * scaling_factor
    else:
        adjusted_weights = clipped_weights

    # Step 4: Validate the adjusted weights
    total_allocation = np.sum(np.abs(adjusted_weights))
    max_weight = np.max(np.abs(adjusted_weights))

    if not np.isclose(total_allocation, allocation_constraint, atol=1e-6) or max_weight > max_exposure:
        raise ValueError("Adjusted weights do not satisfy the constraints.")

    return adjusted_weights


def calculate_dynamic_risk_tolerance(sharpe_ratios):
    """
    Adjust risk tolerance dynamically based on Sharpe ratio volatility.
    """
    return max(0.01, 0.5 / (1 + np.std(sharpe_ratios)))

def validate_constraints(weights):
    """
    Validate constraints:
    - Sum of absolute weights equals ALLOCATION_CONSTRAINT (with rounding).
    - All weights are within [-MAX_EXPOSURE, MAX_EXPOSURE] (with rounding).
    """
    total_allocation = np.round(np.sum(np.abs(weights)), decimals=4)
    max_weight = np.round(np.max(np.abs(weights)), decimals=4)

    constraints_met = (
        np.isclose(total_allocation, ALLOCATION_CONSTRAINT, atol=1e-4)
        and max_weight <= MAX_EXPOSURE
    )

    debug(f"Total allocation: {total_allocation}, Max weight: {max_weight}")
    if constraints_met:
        debug("All constraints are satisfied.")
        return True
    else:
        debug("Constraints not satisfied.")
        return False

# Updated Main Function
def main():
    try:
        # Load and preprocess data
        file_path = f"{DATA_FOLDER}/release_{LATEST_RELEASE}.crypt"
        data = load_and_clean_data(file_path, PASSWORD)
        debug_shape(data, "after loading and cleaning")

        # Correlation matrix
        plot_correlation_matrix(data)
        correlation_matrix = data.corr().values  # Extract correlation matrix as a numpy array

        # Evaluate strategies
        mean_returns, volatility, sharpe_ratios = evaluate_strategies(data)
        cov_matrix = np.cov(data.T) + np.eye(data.shape[1]) * 1e-6

        # Calculate momentum scores
        momentum_scores = calculate_momentum_scores(data)

        # Previous weights (Optional for transaction cost penalty)
        prev_weights = None  # Replace with actual previous weights if available

        # Dynamic risk tolerance
        risk_tolerance = calculate_dynamic_risk_tolerance(sharpe_ratios)

        # Optimize portfolio
        optimized_weights = optimize_portfolio_with_sharpe_momentum_and_hedging(
            mean_returns, cov_matrix, correlation_matrix, momentum_scores,
            target_return=0.01, risk_tolerance=risk_tolerance, prev_weights=prev_weights
        )

        # Redistribute weights to meet strict constraints
        weights = redistribute_excess_weights(
            optimized_weights, max_exposure=MAX_EXPOSURE, allocation_constraint=ALLOCATION_CONSTRAINT
        )

        # Validate constraints after redistribution
        if not validate_constraints(weights):
            handle_error("Final weights do not meet constraints.")

        debug(f"Final optimized weights: {weights}")

        # Save allocation submission
        save_allocation_submission(weights)

        # Evaluate performance
        portfolio_returns, cumulative_pnl, sharpe_ratio = evaluate_performance(data, weights)

        # Plot results
        plot_allocation(weights)
        plot_pnl(portfolio_returns, cumulative_pnl)
        plot_selected_correlation(correlation_matrix, weights)

        debug(f"Final Sharpe Ratio: {sharpe_ratio}")
        debug(f"Final Portfolio Allocation: {weights}")
    except Exception as e:
        handle_error(f"Main process failed: {e}")


if __name__ == "__main__":
    main()
