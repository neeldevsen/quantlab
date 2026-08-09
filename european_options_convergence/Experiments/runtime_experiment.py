import pandas as pd
import numpy as np
import time

import bs_binomial as bs_bin
import b76_binomial as b76_bin
import merton_binomial as merton_bin

import bs_trinomial as bs_tri
import b76_trinomial as b76_tri
import merton_trinomial as merton_tri

import bs_fdm
import b76_fdm
import merton_fdm

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

n = np.unique(np.logspace(1, 4.5, 40, dtype=int))


# Black-Scholes
bs_bin_call_runtime = np.empty(len(n))
bs_bin_put_runtime = np.empty(len(n))
bs_tri_call_runtime = np.empty(len(n))
bs_tri_put_runtime = np.empty(len(n))
bs_fdm_call_runtime = np.empty(len(n))
bs_fdm_put_runtime = np.empty(len(n))

# Black-76
b76_bin_call_runtime = np.empty(len(n))
b76_bin_put_runtime = np.empty(len(n))
b76_tri_call_runtime = np.empty(len(n))
b76_tri_put_runtime = np.empty(len(n))
b76_fdm_call_runtime = np.empty(len(n))
b76_fdm_put_runtime = np.empty(len(n))

# Merton
merton_bin_call_runtime = np.empty(len(n))
merton_bin_put_runtime = np.empty(len(n))
merton_tri_call_runtime = np.empty(len(n))
merton_tri_put_runtime = np.empty(len(n))
merton_fdm_call_runtime = np.empty(len(n))
merton_fdm_put_runtime = np.empty(len(n))


# ============================================================
# Black-Scholes
# ============================================================

for i in range(0,len(n)):
    N = n[i]
    start = time.perf_counter()
    bs_bin.call(
        spot, strike, rate, volatility, maturity, N
    )
    end = time.perf_counter()
    bs_bin_call_runtime[i] = end - start

    start = time.perf_counter()
    bs_bin.put(
            spot, strike, rate, volatility, maturity, N
        )
    end = time.perf_counter()
    bs_bin_put_runtime[i] = end - start

    start = time.perf_counter()
    bs_tri.call(
                spot, strike, rate, volatility, maturity, N
            )
    end = time.perf_counter()
    bs_tri_call_runtime[i] = end - start

    start = time.perf_counter()
    bs_tri.put(
                spot, strike, rate, volatility, maturity, N
            )
    end = time.perf_counter()
    bs_tri_put_runtime[i] = end - start

    start = time.perf_counter()
    bs_fdm.call(
                    spot, strike, rate, volatility, maturity, N, N
                )
    end = time.perf_counter()
    bs_fdm_call_runtime[i] = end - start

    start = time.perf_counter()
    bs_fdm.put(
                spot, strike, rate, volatility, maturity, N, N
            )
    end = time.perf_counter()
    bs_fdm_put_runtime[i] = end - start

    # ============================================================
    # Black-76
    # ============================================================


    start = time.perf_counter()
    b76_bin.call(
            future, strike, rate, volatility, maturity, N
        )
    end = time.perf_counter()
    b76_bin_call_runtime[i] = end - start
    
    start = time.perf_counter()
    b76_bin.put(
                future, strike, rate, volatility, maturity, N
            )
    end = time.perf_counter()
    b76_bin_put_runtime[i] = end - start
    
    start = time.perf_counter()
    b76_tri.call(
                    future, strike, rate, volatility, maturity, N
                )
    end = time.perf_counter()
    b76_tri_call_runtime[i] = end - start
    
    start = time.perf_counter()
    b76_tri.put(
                    future, strike, rate, volatility, maturity, N
                )
    end = time.perf_counter()
    b76_tri_put_runtime[i] = end - start
    
    start = time.perf_counter()
    b76_fdm.call(
                        future, strike, rate, volatility, maturity, N, N
                    )
    end = time.perf_counter()
    b76_fdm_call_runtime[i] = end - start
    
    start = time.perf_counter()
    b76_fdm.put(
                    future, strike, rate, volatility, maturity, N, N
                )
    end = time.perf_counter()
    b76_fdm_put_runtime[i] = end - start

    # ============================================================
    # Merton
    # ============================================================


    start = time.perf_counter()
    merton_bin.call(
            spot, strike, rate, dividend, volatility, maturity, N
        )
    end = time.perf_counter()
    merton_bin_call_runtime[i] = end - start
    
    start = time.perf_counter()
    merton_bin.put(
                spot, strike, rate, dividend, volatility, maturity, N
            )
    end = time.perf_counter()
    merton_bin_put_runtime[i] = end - start
    
    start = time.perf_counter()
    merton_tri.call(
                    spot, strike, rate, dividend, volatility, maturity, N
                )
    end = time.perf_counter()
    merton_tri_call_runtime[i] = end - start
    
    start = time.perf_counter()
    merton_tri.put(
                    spot, strike, rate, dividend, volatility, maturity, N
                )
    end = time.perf_counter()
    merton_tri_put_runtime[i] = end - start
    
    start = time.perf_counter()
    merton_fdm.call(
                        spot, strike, rate, dividend, volatility, maturity, N, N
                    )
    end = time.perf_counter()
    merton_fdm_call_runtime[i] = end - start
    
    start = time.perf_counter()
    merton_fdm.put(
                    spot, strike, rate, dividend, volatility, maturity, N, N
                )
    end = time.perf_counter()
    merton_fdm_put_runtime[i] = end - start

data_BS_Call = {
    "N": n,
    "Binomial Tree": bs_bin_call_runtime,
    "Trinomial Tree": bs_tri_call_runtime,
    "Finite Differences": bs_fdm_call_runtime
}

data_BS_Put = {
    "N": n,
    "Binomial Tree": bs_bin_put_runtime,
    "Trinomial Tree": bs_tri_put_runtime,
    "Finite Differences": bs_fdm_put_runtime
}

data_B76_Call = {
    "N": n,
    "Binomial Tree": b76_bin_call_runtime,
    "Trinomial Tree": b76_tri_call_runtime,
    "Finite Differences": b76_fdm_call_runtime
}

data_B76_Put = {
    "N": n,
    "Binomial Tree": b76_bin_put_runtime,
    "Trinomial Tree": b76_tri_put_runtime,
    "Finite Differences": b76_fdm_put_runtime
}

data_Merton_Call = {
    "N": n,
    "Binomial Tree": merton_bin_call_runtime,
    "Trinomial Tree": merton_tri_call_runtime,
    "Finite Differences": merton_fdm_call_runtime
}

data_Merton_Put = {
    "N": n,
    "Binomial Tree": merton_bin_put_runtime,
    "Trinomial Tree": merton_tri_put_runtime,
    "Finite Differences": merton_fdm_put_runtime
}


df_BS_Call = pd.DataFrame(data_BS_Call)
df_BS_Put = pd.DataFrame(data_BS_Put)
df_B76_Call = pd.DataFrame(data_B76_Call)
df_B76_Put = pd.DataFrame(data_B76_Put)
df_Merton_Call = pd.DataFrame(data_Merton_Call)
df_Merton_Put = pd.DataFrame(data_Merton_Put)

df_BS_Call.to_csv(EXCELS / "Black-Scholes-Call-Runtimes.csv", index=False) 
df_BS_Put.to_csv(EXCELS / "Black-Scholes-Put-Runtimes.csv", index=False)  
df_B76_Call.to_csv(EXCELS / "Black-76-Call-Runtimes.csv", index=False) 
df_B76_Put.to_csv(EXCELS / "Black-76-Put-Runtimes.csv", index=False) 
df_Merton_Call.to_csv(EXCELS / "Merton-Call-Runtimes.csv", index=False) 
df_Merton_Put.to_csv(EXCELS / "Merton-Put-Runtimes.csv", index=False)  

