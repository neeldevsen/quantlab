import black76 as b76_cf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
EXCELS = BASE / "../Excels"


df = pd.read_csv(EXCELS / "Black-Scholes-Call-Runtimes.csv")

N = np.array(df["N"])
binomial_tree = np.array(df["Binomial Tree"])
trinomial_tree = np.array(df["Trinomial Tree"])
finite_differences = np.array(df["Finite Differences"])

size_of_type = 8
log_mem_BT = np.log((2*N-1) * size_of_type)
log_mem_TT = np.log(4*N * size_of_type)
log_mem_FDM = np.log((4*N-2) * size_of_type)

log_abs_error_BT = np.empty(len(N))
log_abs_error_BT[:] = np.log(np.abs(binomial_tree[:]))

log_abs_error_TT = np.empty(len(N))
log_abs_error_TT[:] = np.log(np.abs(trinomial_tree[:]))

log_abs_error_FDM = np.empty(len(N))
log_abs_error_FDM[:] = np.log(np.abs(finite_differences[:]))

fig, ax = plt.subplots()
ax.plot(log_mem_BT, log_abs_error_BT, label = "Binomial Tree", color="blue")
ax.plot(log_mem_TT, log_abs_error_TT, label = "Trinomial Tree", color="red")
ax.plot(log_mem_FDM, log_abs_error_FDM, label = "Finite Differences", color="black")

ax.set_title("Memory--Runtime Graph for Black Scholes Calls")
ax.set_xlabel("log Memory (Bytes)")
ax.set_ylabel("log Runtime (s)")

ax.legend(
    loc="upper left",
)

plt.show()
