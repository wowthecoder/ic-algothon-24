import riskfolio as rp
import matplotlib.pyplot as plt
import pandas as pd


import pandas as pd
import cryptpandas as crp
import numpy as np

import main1

import warnings
warnings.filterwarnings("ignore")



DATA_FOLDER = "./data_releases"
TEAM_NAME = "limoji"
PASSCODE = "014ls434>"
SUBMISSION_FILE = "submissions1.txt"
LATEST_RELEASE = "4635"
PASSWORD = "hpuTAsG3v5av6J0D"




def get_weights(data):

    asset_classes = pd.read_csv("asset_classes.csv").sort_values(by = ["Assets"])
    asset_classes = asset_classes.sort_values(by=['Assets'])

    assets = list(asset_classes["Assets"])


    constraints = pd.read_csv("constraints.csv")
    constraints


    returns = data.pct_change().dropna()
    returns

    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    method_mu = 'hist'
    method_cov = 'hist'
    hist = True
    model = 'Classic'
    rm = 'MDD'
    obj = 'Sharpe'
    rf = 0
    l  = 0

    """
    ’MV’: Standard Deviation.
    ’MAD’: Mean Absolute Deviation.
    ’MSV’: Semi Standard Deviation.
    ’FLPM’: First Lower Partial Moment (Omega Ratio).
    ’SLPM’: Second Lower Partial Moment (Sortino Ratio).
    ’CVaR’: Conditional Value at Risk.
    ’EVaR’: Entropic Value at Risk.
    ’WR’: Worst Realization (Minimax)
    ’MDD’: Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio).
    ’ADD’: Average Drawdown of uncompounded cumulative returns.
    ’DaR’: Drawdown at Risk of uncompounded cumulative returns.
    ’CDaR’: Conditional Drawdown at Risk of uncompounded cumulative returns.
    ’EDaR’: Entropic Drawdown at Risk of uncompounded cumulative returns.
    ’UCI’: Ulcer Index of uncompounded cumulative returns.
    """




    A, B = rp.assets_constraints(constraints, asset_classes)

    port = rp.Portfolio(returns = returns)

    port.assets_stats(method_mu=method_mu, method_cov=method_cov)

    port.upperlng = 0.1  # Setting the upper limit for long positions

    # Perform optimization
    weights = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)


    return weights


    # ax = rp.plot_pie(w=w, title = "Optimum Portfolio", others = 0.05, cmap = 'tab20')
    # plt.show()