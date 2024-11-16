import pandas as pd
import cryptpandas as crp
import matplotlib.pyplot as plt
import verify_submission
from scipy.optimize import minimize
import implemeted
import main1
import riskfolio as rp
import numpy as np

TEAM_NAME = "limoji"
PASSCODE = "014ls434>"

decrypted_df = crp.read_encrypted(path='data_releases/release_4955.crypt', password='sL7KZHNduMf2Cy5D')
# decrypted_df = crp.read_encrypted(path='release_3611.crypt', password='GMJVDf4WWzsV1hfL')

cumulative_data = decrypted_df.cumsum()
plt.plot(cumulative_data["strat_3"])
plt.show()


weights = implemeted.get_weights(decrypted_df)

print(weights)

ax = rp.plot_pie(w=weights, title = "Optimum Portfolio", others = 0.05, cmap = 'tab20')
plt.show()


# Prepare submission dictionary
submission = {
    **weights.to_dict(),
    "team_name": TEAM_NAME,
    "passcode": PASSCODE,
}

# Output results to terminal
print("\n\n\n")
print(f"Submission: {submission}")