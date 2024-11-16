import pandas as pd
import cryptpandas as crp
import matplotlib.pyplot as plt
import verify_submission
from scipy.optimize import minimize


decrypted_df = crp.read_encrypted(path='data_releases/release_3739.crypt', password='1vA9LaAZDTEKPePs')
# decrypted_df = crp.read_encrypted(path='release_3611.crypt', password='GMJVDf4WWzsV1hfL')

print("df")
print(decrypted_df)

#release_3611.crypt' the passcode is 'GMJVDf4WWzsV1hfL'
# plot decrypted_df
plt.plot(decrypted_df)
plt.show()

cumulative_data = decrypted_df.cumsum()
plt.plot(cumulative_data)
plt.show()

# Calculate the final cumulative value for each strategy
final_values = cumulative_data.iloc[-1]

print(final_values)
