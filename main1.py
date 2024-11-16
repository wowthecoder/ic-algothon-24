import pandas as pd
import numpy as np
import cryptpandas as crp

# Constants
DATA_FOLDER = "./data_releases"
TEAM_NAME = "limoji"
PASSCODE = "014ls434>"
SUBMISSION_FILE = "submissions1.txt"
LATEST_RELEASE = "4123"
PASSWORD = "WM5xrwsJiBCo4Unp"

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
    data = data.fillna(0.0)  # Replace NaN with 0.0\

    #print(round(data))
    print(f"[DEBUG] Data after cleaning:\n{data}")
    return data

def redistribute_excess_dynamically(weights):
    """
    Dynamically redistribute excess weight from strategies exceeding ±0.1
    to ensure all constraints are met.
    """
    max_iterations = 100  # Avoid infinite loops
    print("Starting dynamic redistribution...")
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}:")
        print(f"Current weights:\n{weights}")

        # Identify strategies exceeding ±0.1
        over_allocated = weights[weights > 0.1]
        under_allocated = weights[weights < -0.1]

        # Break loop if no strategy exceeds ±0.1
        if over_allocated.empty and under_allocated.empty:
            print("No excess detected. Redistribution complete.")
            break

        # Redistribute positive excess
        for index, value in over_allocated.items():
            excess = value - 0.1  # Calculate positive excess
            weights[index] = 0.1  # Clip to max 0.1

            # Redistribute excess to the next best-performing strategies
            eligible_indices = weights[(weights > 0) & (weights < 0.1)].index
            if not eligible_indices.empty:
                redistribution_factors = weights.loc[eligible_indices] / weights.loc[eligible_indices].sum()
                weights.loc[eligible_indices] += redistribution_factors * excess

        # Redistribute negative excess
        for index, value in under_allocated.items():
            excess = value + 0.1  # Calculate negative excess
            weights[index] = -0.1  # Clip to min -0.1

            # Redistribute excess to the next worst-performing strategies
            eligible_indices = weights[(weights < 0) & (weights > -0.1)].index
            if not eligible_indices.empty:
                redistribution_factors = weights.loc[eligible_indices] / weights.loc[eligible_indices].sum()
                weights.loc[eligible_indices] += redistribution_factors * excess

        # Normalize weights to ensure abs(sum(weights)) = 1.0
        weights /= weights.abs().sum()

    print("Final redistributed weights:\n", weights)
    return weights

def calculate_weights_with_constraints(data):
    """
    Calculate strategy weights, clip excess, and dynamically redistribute to meet constraints.
    """
    try:
        # Clean the dataset
        data = clean_data(data)

        cumulative_data = data.cumsum()
        final_values = cumulative_data.iloc[-1]
        print(f"[DEBUG] Final values for strategies:\n{final_values}")

        # Initial normalization
        weights = final_values / final_values.abs().sum()
        print(f"[DEBUG] Initial normalized weights:\n{weights}")

        # Dynamically clip and redistribute excess weights
        weights = redistribute_excess_dynamically(weights)
        print(f"[DEBUG] Final redistributed weights:\n{weights}")

        return weights.dropna()
    except Exception as e:
        print(f"[ERROR] Failed to calculate weights with constraints: {e}")
        raise

def validate_constraints(weights):
    """
    Validate that the constraints are met:
    - The abs sum of the positions must be 1.0
    - The largest abs position must be <= 0.1
    """
    abs_sum = round(weights.abs().sum(), 8)  # Ensure precision for floating-point errors
    max_abs_position = weights.abs().max()

    abs_sum_ok = abs_sum == 1.0
    max_abs_ok = max_abs_position <= 0.1

    print(f"[DEBUG] Validation results: abs_sum_ok={abs_sum_ok}, max_abs_ok={max_abs_ok}")
    if not abs_sum_ok:
        print("[ERROR] The absolute sum of weights does not equal 1.0.")
    if not max_abs_ok:
        print("[ERROR] One or more weights exceed the maximum allowed ±0.1.")
    return abs_sum_ok, max_abs_ok


def save_submission(submission_dict, file_path):
    """
    Save the submission dictionary to a file in append mode.
    Each submission includes the release identifier for clarity.
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

        # Calculate weights
        weights = calculate_weights_with_constraints(data)

        # Validate constraints
        abs_sum_ok, max_abs_ok = validate_constraints(weights)
        if not abs_sum_ok or not max_abs_ok:
            raise ValueError(
                f"Validation failed: abs_sum_ok={abs_sum_ok}, max_abs_ok={max_abs_ok}. "
                f"Check the weights: {weights.to_dict()}"
            )

        # Prepare submission dictionary
        submission = {
            **weights.to_dict(),
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


