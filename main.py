import pandas as pd
import cryptpandas as crp
import matplotlib.pyplot as plt
import verify_submission
from scipy.optimize import minimize
import implemeted
import main1

TEAM_NAME = "limoji"
PASSCODE = "014ls434>"

decrypted_df = crp.read_encrypted(path='data_releases/release_4763.crypt', password='YaYVHkUbVZkzNItu')
# decrypted_df = crp.read_encrypted(path='release_3611.crypt', password='GMJVDf4WWzsV1hfL')
decrypted_df = decrypted_df.dropna()

weights = implemeted.get_weights(decrypted_df)

print(weights)

# Validate constraints
abs_sum_ok, max_abs_ok = main1.validate_constraints(weights)
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
print("\n\n\n")
print(f"Submission: {submission}")