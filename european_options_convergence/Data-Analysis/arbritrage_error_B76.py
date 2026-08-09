import black76 as bs_cf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BASE = Path(__file__).resolve().parent


spot = 120.0
future = 115.0
strike = 100.0

rate = 0.04
dividend = 0.02
volatility = 0.25
maturity = 2.0
multiplier = 4.0




dfC = pd.read_csv(BASE / "../Excels/Black-76-Call.csv")
dfP = pd.read_csv(BASE / "../Excels/Black-76-Put.csv")

N = np.array(dfC["N"])
binomial_tree_call = np.array(dfC["Binomial Tree"])
trinomial_tree_call = np.array(dfC["Trinomial Tree"])
finite_differences_call = np.array(dfC["Finite Differences"])

binomial_tree_put = np.array(dfP["Binomial Tree"])
trinomial_tree_put = np.array(dfP["Trinomial Tree"])
finite_differences_put = np.array(dfP["Finite Differences"])

log_N = np.log(N)

log_arb_error_BT = np.empty(len(N))
log_arb_error_BT[:] = np.log(np.abs(binomial_tree_call[:] - binomial_tree_put[:] - np.exp(-rate*maturity) * (future - strike)))

log_arb_error_TT = np.empty(len(N))
log_arb_error_TT[:] = np.log(np.abs(trinomial_tree_call[:] - trinomial_tree_put[:] - np.exp(-rate*maturity) * (future - strike)))

log_arb_error_FDM = np.empty(len(N))
log_arb_error_FDM[:] = np.log(np.abs(finite_differences_call[:] - finite_differences_put[:] - np.exp(-rate*maturity) * (future - strike)))

fig, ax = plt.subplots()
ax.plot(log_N, log_arb_error_BT, label = "Binomial Tree", color="blue")
ax.plot(log_N, log_arb_error_TT, label = "Trinomial Tree", color="red")
ax.plot(log_N, log_arb_error_FDM, label = "Finite Differences", color="black")

ax.set_title("Arbritrage Error Graph for Black-76 Options")
ax.set_xlabel("log N")
ax.set_ylabel("log Arbitrage Error")

ax.legend(
    loc="upper right"
)

plt.show()
