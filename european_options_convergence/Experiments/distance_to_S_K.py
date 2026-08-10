import numpy as np
import pandas as pd
from pathlib import Path


BASE = Path(__file__).resolve().parent
EXCELS = BASE / "../Excels"


def binomial_tree_min(S,K,r,sigma,T,N):
    k = np.arange(N+1)
    terminal_S = np.empty(N+1)
    S_K_distance = np.empty(N+1)
    u = np.exp(sigma * np.sqrt(T/N))
    d = 1 / u
    terminal_S[:] = S * u**k[:] * d**(N-k[:])
    S_K_distance[:] = np.abs(terminal_S[:] - K)
    nearest_index = np.argmin(S_K_distance)
    if nearest_index == 0:
        spacing = np.abs(terminal_S[1] - terminal_S[0])
    elif nearest_index == len(terminal_S) - 1:
        spacing = np.abs(terminal_S[-1] - terminal_S[-2])
    else:
        spacing = np.abs(terminal_S[nearest_index + 1] - terminal_S[nearest_index - 1]) / 2
    
    return np.min(S_K_distance) / spacing

def trinomial_tree_min(S,K,r,sigma,T,N):
    k = np.arange(-N, N+1)
    terminal_S = np.empty(2*N+1)
    S_K_distance = np.empty(2*N+1)
    u = np.exp(2* sigma * np.sqrt(T/N))
    terminal_S[:] = S * u**k[:]
    S_K_distance[:] = np.abs(terminal_S[:] - K)
    nearest_index = np.argmin(S_K_distance)
    if nearest_index == 0:
        spacing = np.abs(terminal_S[1] - terminal_S[0])
    elif nearest_index == len(terminal_S) - 1:
        spacing = np.abs(terminal_S[-1] - terminal_S[-2])
    else:
        spacing = np.abs(terminal_S[nearest_index + 1] - terminal_S[nearest_index - 1]) / 2

    return np.min(S_K_distance) / spacing

def finite_differences_min(S,K,r,sigma,T,N,multiplier):
    Smax = multiplier * max(S,K)
    deltaS = Smax / N
    i = np.arange(N+1)
    S_grid = np.empty(N+1)
    S_K_distance = np.empty(N+1)
    S_grid[:] = i[:] * deltaS
    S_K_distance[:] =  np.abs(S_grid[:] - K)
    return np.min(S_K_distance) / deltaS

n = np.unique(np.logspace(1, 4.5, 40, dtype=int))

distance_S_K_BT = np.empty(len(n))
distance_S_K_TT = np.empty(len(n))
distance_S_K_FDM = np.empty(len(n))


spot = 120.0
future = 115.0
strike = 100.0

rate = 0.04
dividend = 0.02
volatility = 0.25
maturity = 2.0
multiplier = 4.0


for i in range(0,len(n)):
    N = n[i]
    distance_S_K_BT[i] = binomial_tree_min(spot,strike,rate,volatility,maturity,N)
    distance_S_K_TT[i] = trinomial_tree_min(spot,strike,rate,volatility,maturity,N)
    distance_S_K_FDM[i] = finite_differences_min(spot,strike,rate,volatility,maturity,N, multiplier)

data = {
    "N" : n,
    "Binomial Tree": distance_S_K_BT,
    "Trinomial Tree": distance_S_K_TT,
    "Finite Differences": distance_S_K_FDM
}

df = pd.DataFrame(data)

df.to_csv(EXCELS / "S-K-Distance.csv", index=False)

