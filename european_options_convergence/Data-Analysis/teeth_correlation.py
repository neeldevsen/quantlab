import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import black_scholes as bs_cf
from scipy.stats import pearsonr, spearmanr


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


df_distances = pd.read_csv(EXCELS / "S-K-Distance.csv")
df_errors = pd.read_csv(EXCELS / "Black-Scholes-Call.csv")


binomial_tree_err = np.array(np.abs((df_errors["Binomial Tree"]) - bs_cf_call))
trinomial_tree_err = np.array(np.abs((df_errors["Trinomial Tree"])- bs_cf_call))
finite_differences_err = np.array(np.abs((df_errors["Finite Differences"])- bs_cf_call))

N = np.array((df_errors["N"]))

residual_binomial_tree_err = np.empty(len(N))
residual_trinomial_tree_err = np.empty(len(N))
residual_finite_differences_err = np.empty(len(N))

z = np.polyfit(N, binomial_tree_err, 1)
best_line_fit_BT = np.poly1d(z)
slope_BT, intercept_BT = best_line_fit_BT

z = np.polyfit(N, trinomial_tree_err, 1)
best_line_fit_TT = np.poly1d(z)
slope_TT, intercept_TT = best_line_fit_TT

z = np.polyfit(N, finite_differences_err, 1)
best_line_fit_FDM = np.poly1d(z)
slope_FDM, intercept_FDM = best_line_fit_FDM

residual_binomial_tree_err[:] = np.abs(slope_BT * N[:] + intercept_BT - binomial_tree_err[:])
residual_trinomial_tree_err[:] = np.abs(slope_TT * N[:] + intercept_TT - trinomial_tree_err[:])
residual_finite_differences_err[:] = np.abs(slope_FDM * N[:] + intercept_FDM - finite_differences_err[:])


binomial_tree_dist = np.array(df_distances["Binomial Tree"])
trinomial_tree_dist = np.array(df_distances["Trinomial Tree"])
finite_differences_dist = np.array(df_distances["Finite Differences"])

print("Binomial Tree")
print("Pearson:", pearsonr(
    binomial_tree_dist,
    residual_binomial_tree_err
))
print("Spearman:", spearmanr(
    binomial_tree_dist,
    residual_binomial_tree_err
))

print("Trinomial Tree")
print("Pearson:", pearsonr(
    trinomial_tree_dist,
    residual_trinomial_tree_err
))
print("Spearman:", spearmanr(
    trinomial_tree_dist,
    residual_trinomial_tree_err
))

print("Finite Differences")
print("Pearson:", pearsonr(
    finite_differences_dist,
    residual_finite_differences_err
))
print("Spearman:", spearmanr(
    finite_differences_dist,
    residual_finite_differences_err
))

plt.scatter(binomial_tree_dist, residual_binomial_tree_err, label = "Binomial Tree", color="blue")
plt.scatter(trinomial_tree_dist, residual_trinomial_tree_err, label = "Trinomial Tree", color="red")
plt.scatter(finite_differences_dist, residual_finite_differences_err, label = "Finite Differences", color="black")

plt.title("Relative Gird Alignment vs Residual in Error")
plt.xlabel("Closest distance of simulated Stock to Strike")
plt.ylabel("Residual in error")

plt.legend(
    loc="upper left"
)

plt.show()