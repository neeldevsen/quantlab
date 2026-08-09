import black_scholes as bs_cf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
EXCELS = BASE / "../Excels"


spot = 120.0
future = 115.0
strike = 100.0

rate = 0.04
dividend = 0.02
volatility = 0.25
maturity = 2.0
multiplier = 4.0


bs_cf_put = bs_cf.put(
    spot, strike, rate, volatility, maturity
)




df = pd.read_csv(EXCELS / "Black-Scholes-Put.csv")

N = np.array(df["N"])
binomial_tree = np.array(df["Binomial Tree"])
trinomial_tree = np.array(df["Trinomial Tree"])
finite_differences = np.array(df["Finite Differences"])

log_N = np.log(N)

log_abs_error_BT = np.empty(len(N))
log_abs_error_BT[:] = np.log(np.abs(binomial_tree[:] - bs_cf_put))

log_abs_error_TT = np.empty(len(N))
log_abs_error_TT[:] = np.log(np.abs(trinomial_tree[:] - bs_cf_put))

log_abs_error_FDM = np.empty(len(N))
log_abs_error_FDM[:] = np.log(np.abs(finite_differences[:] - bs_cf_put))

fig, ax = plt.subplots()
ax.plot(log_N, log_abs_error_BT, label = "Binomial Tree", color="blue")
ax.plot(log_N, log_abs_error_TT, label = "Trinomial Tree", color="red")
ax.plot(log_N, log_abs_error_FDM, label = "Finite Differences", color="black")

ax.set_title("Convergence Graph for Black-Scholes Puts")
ax.set_xlabel("log N")
ax.set_ylabel("log Absolute Error")

ax.legend(
    loc="upper right"
)

plt.show()
