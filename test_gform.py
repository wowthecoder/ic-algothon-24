import requests 
from selenium import webdriver
from selenium.webdriver.common.by import By
import json

# GOOGLE_FORM_URL = "https://docs.google.com/forms/u/0/d/e/1FAIpQLSeUYMkI5ce18RL2aF5C8I7mPxF7haH23VEVz7PQrvz0Do0NrQ/formResponse"
# weights = {'weights': {'strat_0': 0.006381321943660828, 'strat_1': 0.06727097269072646, 'strat_2': 1.793672853492162e-06, 'strat_3': 0.0006476542050530071, 'strat_4': 0.1508090967531575, 'strat_5': 0.13862179839705754, 'strat_6': 0.07749009529728924, 'strat_7': 5.735804434520436e-06, 'strat_8': 0.009221647920815645, 'strat_9': 0.10895501684642905, 'strat_10': 1.1045876655116052e-06, 'strat_11': 7.0627892021710625e-06, 'strat_12': 0.0005009058897479787, 'strat_13': 1.5046738039496306e-05, 'strat_14': 0.0001953354142721439, 'strat_15': 0.029489923283291405, 'strat_16': 0.10839254964131491, 'strat_17': 5.052358974651055e-05, 'strat_18': 0.0051976380126376245, 'strat_19': 4.118071798426025e-06, 'strat_20': 1.4015848589568877e-05, 'strat_21': 4.148740388452795e-05, 'strat_22': 1.925850976787125e-06, 'strat_23': 2.2596083826031726e-05, 'strat_24': 8.864677150040104e-05, 'strat_25': 1.5080827635702046e-05, 'strat_26': 3.870006658986266e-06, 'strat_27': 0.01385531815310859, 'strat_28': 0.0004642619730967113, 'strat_29': 9.471919768993334e-05, 'strat_30': 0.04127969381387015, 'strat_31': 0.00019074960766395264, 'strat_32': 0.0008977003754621607, 'strat_33': 4.81257966633433e-05, 'strat_34': 0.01670628227370584, 'strat_35': 0.0002425953048305316, 'strat_36': 0.00837144734310745, 'strat_37': 2.4569967552581388e-05, 'strat_38': 7.48780932401483e-05, 'strat_39': 0.004365682865357878, 'strat_40': 1.4129010819258632e-05, 'strat_41': 0.10559541605379873, 'strat_42': 3.36821197326486e-06, 'strat_43': 0.10432409761579407}, 'team_name': 'asdfasdfasd', 'passcode': '8734gf'}
# form_data = {
#     "entry.1985358237": weights, 
#     "emailAddress": "jherng365@gmail.com"
# }

GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSeUYMkI5ce18RL2aF5C8I7mPxF7haH23VEVz7PQrvz0Do0NrQ/viewform"
weights = {'weights': {'strat_0': 0.006381321943660828, 'strat_1': 0.06727097269072646, 'strat_2': 1.793672853492162e-06, 'strat_3': 0.0006476542050530071, 'strat_4': 0.1508090967531575, 'strat_5': 0.13862179839705754, 'strat_6': 0.07749009529728924, 'strat_7': 5.735804434520436e-06, 'strat_8': 0.009221647920815645, 'strat_9': 0.10895501684642905, 'strat_10': 1.1045876655116052e-06, 'strat_11': 7.0627892021710625e-06, 'strat_12': 0.0005009058897479787, 'strat_13': 1.5046738039496306e-05, 'strat_14': 0.0001953354142721439, 'strat_15': 0.029489923283291405, 'strat_16': 0.10839254964131491, 'strat_17': 5.052358974651055e-05, 'strat_18': 0.0051976380126376245, 'strat_19': 4.118071798426025e-06, 'strat_20': 1.4015848589568877e-05, 'strat_21': 4.148740388452795e-05, 'strat_22': 1.925850976787125e-06, 'strat_23': 2.2596083826031726e-05, 'strat_24': 8.864677150040104e-05, 'strat_25': 1.5080827635702046e-05, 'strat_26': 3.870006658986266e-06, 'strat_27': 0.01385531815310859, 'strat_28': 0.0004642619730967113, 'strat_29': 9.471919768993334e-05, 'strat_30': 0.04127969381387015, 'strat_31': 0.00019074960766395264, 'strat_32': 0.0008977003754621607, 'strat_33': 4.81257966633433e-05, 'strat_34': 0.01670628227370584, 'strat_35': 0.0002425953048305316, 'strat_36': 0.00837144734310745, 'strat_37': 2.4569967552581388e-05, 'strat_38': 7.48780932401483e-05, 'strat_39': 0.004365682865357878, 'strat_40': 1.4129010819258632e-05, 'strat_41': 0.10559541605379873, 'strat_42': 3.36821197326486e-06, 'strat_43': 0.10432409761579407}, 'team_name': 'asdfasdfasd', 'passcode': '8734gf'}

options = webdriver.ChromeOptions()
# options.add_argument('--headless')
driver = webdriver.Chrome(options=options)

# Open the form
driver.get(GOOGLE_FORM_URL)

# Check the response
textarea = driver.find_element(By.XPATH, '//textarea[@aria-label="Your answer"]')

# Interact with the textarea (e.g., input text)
textarea.send_keys(json.dumps(weights))

driver.find_element(By.CSS_SELECTOR, "div[role='button']").click()

print("Yay succeeded")

# Close the browser
driver.quit()