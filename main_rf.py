import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cryptpandas as crp

# Constants
DATA_FOLDER = "./data_releases"
TEAM_NAME = "limoji"
PASSCODE = "014ls434>"
SUBMISSION_FILE = "submissions.txt"
LATEST_RELEASE = "4635"
PASSWORD = "hpuTAsG3v5av6J0D"

# Data Cleaning and Feature Engineering
def process_data(file_path, password):
    try:
        data = crp.read_encrypted(file_path, password=password)
        print(f"[DEBUG] Successfully decrypted data from {file_path}. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to process data from {file_path}: {e}")
        raise

def clean_data(data):
    """
    Clean PnL data by removing NaNs, infinities, and capping extreme outliers.
    """
    print("[DEBUG] Cleaning data...")
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0.0, inplace=True)
    
    # Capping outliers to +/- 5 standard deviations
    std = data.std(axis=0)  # Specify axis=0 for column-wise standard deviation
    mean = data.mean(axis=0)  # Specify axis=0 for column-wise mean
    data = data.clip(lower=mean - 5 * std, upper=mean + 5 * std, axis=1)  # Explicit axis=1 for columns
    print("[DEBUG] Data cleaned. Shape:", data.shape)
    return data

# Weight Calculations
def calculate_sharpe_weights(data):
    print("[DEBUG] Calculating Sharpe ratio weights...")
    expected_returns = data.mean()
    volatilities = data.std()
    sharpe_ratios = expected_returns / volatilities
    weights = sharpe_ratios / sharpe_ratios.abs().sum()
    print("[DEBUG] Initial Sharpe ratio weights:\n", weights)
    return redistribute_weights(weights)

def calculate_momentum_weights(data):
    print("[DEBUG] Calculating momentum weights...")
    momentum_scores = data.cumsum().iloc[-1]
    weights = momentum_scores / momentum_scores.abs().sum()
    print("[DEBUG] Initial momentum weights:\n", weights)
    return redistribute_weights(weights)

def redistribute_weights(weights):
    """
    Dynamically redistribute weights to meet constraints.
    """
    max_iterations = 100
    for iteration in range(max_iterations):
        over_allocated = weights[weights > 0.1]
        under_allocated = weights[weights < -0.1]
        if over_allocated.empty and under_allocated.empty:
            break
        for idx in over_allocated.index:
            excess = weights[idx] - 0.1
            weights[idx] = 0.1
            eligible = weights[(weights > 0) & (weights < 0.1)].index
            if not eligible.empty:
                redistribution = excess / len(eligible)
                weights.loc[eligible] += redistribution
        for idx in under_allocated.index:
            excess = weights[idx] + 0.1
            weights[idx] = -0.1
            eligible = weights[(weights < 0) & (weights > -0.1)].index
            if not eligible.empty:
                redistribution = excess / len(eligible)
                weights.loc[eligible] += redistribution
        weights /= weights.abs().sum()
    print("[DEBUG] Final redistributed weights:\n", weights)
    return weights

# Performance Evaluation
def evaluate_performance(data, weights):
    portfolio_returns = data @ weights
    cumulative_pnl = portfolio_returns.cumsum()
    mean_return = portfolio_returns.mean()
    std_dev = portfolio_returns.std()
    sharpe_ratio = mean_return / std_dev if std_dev > 0 else 0
    max_drawdown = (cumulative_pnl - cumulative_pnl.cummax()).min()
    
    print("[METRICS] Sharpe Ratio:", sharpe_ratio)
    print("[METRICS] Mean Return:", mean_return)
    print("[METRICS] Standard Deviation:", std_dev)
    print("[METRICS] Max Drawdown:", max_drawdown)
    print("[METRICS] Final Cumulative PnL:", cumulative_pnl.iloc[-1])
    return portfolio_returns, cumulative_pnl

# Visualization
def plot_pnl(portfolio_returns, cumulative_pnl):
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_pnl, label="Cumulative PnL", color="blue")
    plt.title("Portfolio Cumulative PnL")
    plt.xlabel("Time")
    plt.ylabel("PnL")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_returns, label="Per-Period PnL", color="green")
    plt.title("Portfolio Per-Period PnL")
    plt.xlabel("Time")
    plt.ylabel("PnL")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_allocation(weights):
    abs_weights = weights.abs()
    labels = [f"{idx} (short)" if w < 0 else idx for idx, w in weights.items()]
    colors = ["red" if w < 0 else "green" for w in weights]

    plt.figure(figsize=(8, 8))
    plt.pie(abs_weights, labels=labels, autopct="%.1f%%", startangle=90, colors=colors, wedgeprops={"edgecolor": "black"})
    plt.title("Portfolio Allocation (Red = Short, Green = Long)")
    plt.show()

# Main Function
def main():
    file_path = f"{DATA_FOLDER}/release_{LATEST_RELEASE}.crypt"
    try:
        data = process_data(file_path, PASSWORD)
        data = clean_data(data)
        
        sharpe_weights = calculate_sharpe_weights(data)
        momentum_weights = calculate_momentum_weights(data)
        
        combined_weights = (sharpe_weights + momentum_weights) / 2
        portfolio_returns, cumulative_pnl = evaluate_performance(data, combined_weights)
        
        # Visualization
        plot_pnl(portfolio_returns, cumulative_pnl)
        plot_allocation(combined_weights)
        
        # Save submission
        submission = {**combined_weights.to_dict(), "team_name": TEAM_NAME, "passcode": PASSCODE}
        print("[DEBUG] Submission:\n", submission)
    except Exception as e:
        print(f"[ERROR] Main process failed: {e}")

if __name__ == "__main__":
    main()
