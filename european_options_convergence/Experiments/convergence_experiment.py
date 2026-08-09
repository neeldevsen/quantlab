import pandas as pd
import numpy as np

import black_scholes as bs_cf
import black76 as b76_cf
import merton as merton_cf

import bs_binomial as bs_bin
import b76_binomial as b76_bin
import merton_binomial as merton_bin

import bs_trinomial as bs_tri
import b76_trinomial as b76_tri
import merton_trinomial as merton_tri

import bs_fdm
import b76_fdm
import merton_fdm


spot = 120.0
future = 115.0
strike = 100.0

rate = 0.04
dividend = 0.02
volatility = 0.25
maturity = 2.0
multiplier = 4.0

n = np.unique(np.logspace(1, 4.5, 40, dtype=int))

bs_cf_call = bs_cf.call(
    spot, strike, rate, volatility, maturity
)
print(bs_cf_call)
bs_cf_put = bs_cf.put(
    spot, strike, rate, volatility, maturity
)

b76_cf_call = b76_cf.call(
    future, strike, rate, volatility, maturity
)

b76_cf_put = b76_cf.put(
    future, strike, rate, volatility, maturity
)

merton_cf_call = merton_cf.call(
    spot, strike, rate, dividend, volatility, maturity
)

merton_cf_put = merton_cf.put(
    spot, strike, rate, dividend, volatility, maturity
)

# Black-Scholes
bs_bin_call = np.empty(len(n))
bs_bin_put = np.empty(len(n))
bs_tri_call = np.empty(len(n))
bs_tri_put = np.empty(len(n))
bs_fdm_call = np.empty(len(n))
bs_fdm_put = np.empty(len(n))

# Black-76
b76_bin_call = np.empty(len(n))
b76_bin_put = np.empty(len(n))
b76_tri_call = np.empty(len(n))
b76_tri_put = np.empty(len(n))
b76_fdm_call = np.empty(len(n))
b76_fdm_put = np.empty(len(n))

# Merton
merton_bin_call = np.empty(len(n))
merton_bin_put = np.empty(len(n))
merton_tri_call = np.empty(len(n))
merton_tri_put = np.empty(len(n))
merton_fdm_call = np.empty(len(n))
merton_fdm_put = np.empty(len(n))


# ============================================================
# Black-Scholes
# ============================================================

for i in range(0,len(n)):
    N = n[i]
    bs_bin_call[i] = bs_bin.call(
        spot, strike, rate, volatility, maturity, N
    )

    bs_bin_put[i] = bs_bin.put(
        spot, strike, rate, volatility, maturity, N
    )

    bs_tri_call[i] = bs_tri.call(
        spot, strike, rate, volatility, maturity, N
    )

    bs_tri_put[i] = bs_tri.put(
        spot, strike, rate, volatility, maturity, N
    )

    bs_fdm_call[i] = bs_fdm.call(
        spot, strike, rate, volatility, maturity,
        N, N, multiplier
    )

    bs_fdm_put[i] = bs_fdm.put(
        spot, strike, rate, volatility, maturity,
        N, N, multiplier
    )

    # ============================================================
    # Black-76
    # ============================================================


    b76_bin_call[i] = b76_bin.call(
        future, strike, rate, volatility, maturity, N
    )

    b76_bin_put[i] = b76_bin.put(
        future, strike, rate, volatility, maturity, N
    )

    b76_tri_call[i] = b76_tri.call(
        future, strike, rate, volatility, maturity, N
    )

    b76_tri_put[i] = b76_tri.put(
        future, strike, rate, volatility, maturity, N
    )

    b76_fdm_call[i] = b76_fdm.call(
        future, strike, rate, volatility, maturity,
        N, N, multiplier
    )

    b76_fdm_put[i] = b76_fdm.put(
        future, strike, rate, volatility, maturity,
        N, N, multiplier
    )

    # ============================================================
    # Merton
    # ============================================================


    merton_bin_call[i] = merton_bin.call(
        spot, strike, rate, dividend, volatility, maturity, N
    )
    merton_bin_put[i] = merton_bin.put(
        spot, strike, rate, dividend, volatility, maturity, N
    )

    merton_tri_call[i] = merton_tri.call(
        spot, strike, rate, dividend, volatility, maturity, N
    )

    merton_tri_put[i] = merton_tri.put(
        spot, strike, rate, dividend, volatility, maturity, N
    )

    merton_fdm_call[i] = merton_fdm.call(
        spot, strike, rate, dividend, volatility, maturity,
        N, N, multiplier
    )

    merton_fdm_put[i] = merton_fdm.put(
        spot, strike, rate, dividend, volatility, maturity,
        N, N, multiplier
    )

data_BS_Call = {
    "N": n,
    "Binomial Tree": bs_bin_call,
    "Trinomial Tree": bs_tri_call,
    "Finite Differences": bs_fdm_call
}

data_BS_Put = {
    "N": n,
    "Binomial Tree": bs_bin_put,
    "Trinomial Tree": bs_tri_put,
    "Finite Differences": bs_fdm_put
}

data_B76_Call = {
    "N": n,
    "Binomial Tree": b76_bin_call,
    "Trinomial Tree": b76_tri_call,
    "Finite Differences": b76_fdm_call
}

data_B76_Put = {
    "N": n,
    "Binomial Tree": b76_bin_put,
    "Trinomial Tree": b76_tri_put,
    "Finite Differences": b76_fdm_put
}

data_Merton_Call = {
    "N": n,
    "Binomial Tree": merton_bin_call,
    "Trinomial Tree": merton_tri_call,
    "Finite Differences": merton_fdm_call
}

data_Merton_Put = {
    "N": n,
    "Binomial Tree": merton_bin_put,
    "Trinomial Tree": merton_tri_put,
    "Finite Differences": merton_fdm_put
}


df_BS_Call = pd.DataFrame(data_BS_Call)
df_BS_Put = pd.DataFrame(data_BS_Put)
df_B76_Call = pd.DataFrame(data_B76_Call)
df_B76_Put = pd.DataFrame(data_B76_Put)
df_Merton_Call = pd.DataFrame(data_Merton_Call)
df_Merton_Put = pd.DataFrame(data_Merton_Put)

df_BS_Call.to_csv("Black-Scholes-Call.csv", index=False) 
df_BS_Put.to_csv("Black-Scholes-Put.csv", index=False)  
df_B76_Call.to_csv("Black-76-Call.csv", index=False) 
df_B76_Put.to_csv("Black-76-Put.csv", index=False) 
df_Merton_Call.to_csv("Merton-Call.csv", index=False) 
df_Merton_Put.to_csv("Merton-Put.csv", index=False)  

