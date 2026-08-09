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




df = pd.read_csv(EXCELS / "Black-Scholes-Call-Runtimes.csv")

N = np.array(df["N"])
binomial_tree = np.array(df["Binomial Tree"])
trinomial_tree = np.array(df["Trinomial Tree"])
finite_differences = np.array(df["Finite Differences"])

log_N = np.log(N)
mask = log_N >= 4

log_abs_error_BT = np.empty(len(N))
log_abs_error_BT[:] = np.log(np.abs(binomial_tree[:]))
z = np.polyfit(log_N[mask], log_abs_error_BT[mask], 1)
best_line_fit_BT = np.poly1d(z)
slope_BT, intercept_BT = best_line_fit_BT

log_abs_error_TT = np.empty(len(N))
log_abs_error_TT[:] = np.log(np.abs(trinomial_tree[:]))
z = np.polyfit(log_N[mask], log_abs_error_TT[mask], 1)
best_line_fit_TT = np.poly1d(z)
slope_TT, intercept_TT = best_line_fit_TT

log_abs_error_FDM = np.empty(len(N))
log_abs_error_FDM[:] = np.log(np.abs(finite_differences[:]))
z = np.polyfit(log_N[mask], log_abs_error_FDM[mask], 1)
best_line_fit_FDM = np.poly1d(z)
slope_FDM, intercept_FDM = best_line_fit_FDM

plt.scatter(log_N[mask], log_abs_error_BT[mask], color="blue", marker="x")
plt.scatter(log_N[mask], log_abs_error_TT[mask], color="red", marker="x")
plt.scatter(log_N[mask], log_abs_error_FDM[mask], color="black", marker="x")

plt.plot(log_N, best_line_fit_BT(log_N), label=f"Binomial Tree: y = {slope_BT:.4f}x + {intercept_BT:.4f}", color="blue", linestyle="--")
plt.plot(log_N, best_line_fit_TT(log_N), label=f"Trinomial Tree: y = {slope_TT:.4f}x + {intercept_TT:.4f}", color="red", linestyle="--")
plt.plot(log_N, best_line_fit_FDM(log_N),label = f"Finite Differences: y = {slope_FDM:.4f}x + {intercept_FDM:.4f}", color="black", linestyle="--")

plt.title("Empirical Computational Complexity Scatter Plot for Black Scholes Calls")
plt.xlabel("log N")
plt.ylabel("log Runtime (s)")

plt.legend(
    loc="upper left"
)

plt.show()
