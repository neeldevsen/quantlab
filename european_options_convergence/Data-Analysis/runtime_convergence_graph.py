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


bs_cf_call = bs_cf.call(
    spot, strike, rate, volatility, maturity
)


df_err = pd.read_csv(EXCELS / "Black-Scholes-Call.csv")
df_runtimes = pd.read_csv(EXCELS / "Black-Scholes-Call-Runtimes"
".csv")
N = np.array(df_err["N"])
binomial_tree_error = np.array(df_err["Binomial Tree"])
trinomial_tree_error = np.array(df_err["Trinomial Tree"])
finite_differences_error = np.array(df_err["Finite Differences"])

binomial_tree_runtimes = np.array(df_runtimes["Binomial Tree"])
trinomial_tree_runtimes = np.array(df_runtimes["Trinomial Tree"])
finite_differences_runtimes = np.array(df_runtimes["Finite Differences"])

log_N = np.log(N)

log_abs_error_BT = np.empty(len(N))
log_abs_error_BT[:] = np.log(np.abs(binomial_tree_error[:]- bs_cf_call))

log_abs_error_TT = np.empty(len(N))
log_abs_error_TT[:] = np.log(np.abs(trinomial_tree_error[:] - bs_cf_call))

log_abs_error_FDM = np.empty(len(N))
log_abs_error_FDM[:] = np.log(np.abs(finite_differences_error[:] - bs_cf_call))

log_BT_runtimes = np.empty(len(N))
log_BT_runtimes[:] = np.log(np.abs(binomial_tree_runtimes[:]))
z = np.polyfit(log_BT_runtimes, log_abs_error_BT, 1)
best_line_fit_BT= np.poly1d(z)
slope_BT, intercept_BT = best_line_fit_BT

log_TT_runtimes = np.empty(len(N))
log_TT_runtimes[:] = np.log(np.abs(trinomial_tree_runtimes[:]))
z = np.polyfit(log_TT_runtimes, log_abs_error_TT, 1)
best_line_fit_TT = np.poly1d(z)
slope_TT, intercept_TT = best_line_fit_TT

log_FDM_runtimes = np.empty(len(N))
log_FDM_runtimes[:] = np.log(np.abs(finite_differences_runtimes[:]))
z = np.polyfit(log_FDM_runtimes, log_abs_error_FDM, 1)
best_line_fit_FDM = np.poly1d(z)
slope_FDM, intercept_FDM = best_line_fit_FDM


plt.scatter(log_BT_runtimes, log_abs_error_BT, color="blue", marker="x")
plt.scatter(log_TT_runtimes, log_abs_error_TT, color="red", marker="x")
plt.scatter(log_FDM_runtimes, log_abs_error_FDM, color="black", marker="x")

plt.plot(log_BT_runtimes, best_line_fit_BT(log_BT_runtimes), label=f"Binomial Tree: y = {slope_BT:.4f}x + {intercept_BT:.4f}", color="blue", linestyle="--")
plt.plot(log_TT_runtimes, best_line_fit_TT(log_TT_runtimes), label=f"Trinomial Tree: y = {slope_TT:.4f}x + {intercept_TT:.4f}", color="red", linestyle="--")
plt.plot(log_FDM_runtimes, best_line_fit_FDM(log_FDM_runtimes),label = f"Finite Differences: y = {slope_FDM:.4f}x + {intercept_FDM:.4f}", color="black", linestyle="--")


plt.title("Empirical Computational Complexity for Black Scholes Calls")
plt.xlabel("log Runtime (s)")
plt.ylabel("log Absolute Error")

plt.legend(
    loc="lower left",
)

plt.show()
