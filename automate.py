import pandas as pd
import numpy as np
import cryptpandas as crp
import json
from flask import Flask, request, jsonify
from main1 import process_data, calculate_weights_with_constraints, validate_constraints, save_submission

app = Flask(__name__)

# Constants
DATA_FOLDER = "./data_releases"
TEAM_NAME = "limoji"
PASSCODE = "014ls434>"
SUBMISSION_FILE = "submissions1.txt"
SLACK_CLIENT_ID = "8020284472341.8039893431250"
SLACK_CLIENT_SECRET = "1ac9f7fe408aa41eabcf2267caecbbb1"
LOCALHOST_ADDRESS = "http://172.26.157.241:7879/message"

#setup
latest_release = 3867
latest_password = "1vA9LaAZDTEKPePs"
with open("algothon_google_api.json") as f:
    google_api_credentials = json.load(f)["installed"]

@app.route('/')
def test():
    return "hello world"

@app.route("/message", methods=['POST'])
def main():
    data = request.get_json()
    if "challenge" in data:
        return data["challenge"]
    else:
        print(data)
        # Process the latest file
        latest_file_path = f"{DATA_FOLDER}/release_{latest_release}.crypt"
        data = process_data(latest_file_path, latest_password)

        # Calculate weights with constraints
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

        # Increment latest release
        latest_release += 64