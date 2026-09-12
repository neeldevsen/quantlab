import matplotlib.pyplot as plt
import numpy as np

def MC_w_stop_loss(S0, r, vol, T, M, N, stop_loss):
    stocks = np.zeros(M)
    returns = np.zeros(M)
    dt = T/ N
    for i in range(0, M):
        S = S0
        for j in range(0, N):
            Z = np.random.standard_normal()
            S *= np.exp(((r-0.5*vol**2)*dt + vol * np.sqrt(dt) * Z))
            if S < S0 * (1 - stop_loss):
                break
        stocks[i] = S
    returns[:] = np.log(stocks[:] / S0)

    return returns

def MC_wo_stop_loss(S0, r, vol, T, M, N):
    stocks = np.zeros(M)
    returns = np.zeros(M)
    dt = T/ N
    for i in range(0, M):
        S = S0
        for j in range(0, N):
            Z = np.random.normal(loc=0.0, scale=np.sqrt(dt))
            S *= np.exp(((r-0.5*vol**2)*dt + vol * Z))
        stocks[i] = S
    returns[:] = np.log(stocks[:] / S0)

    return returns

S = 100
r = 0.01
vol = 0.3
T = 1
M = 50000
N = 252
stop_loss = 0.1

returns_w = MC_w_stop_loss(S, r, vol, T, M, N, stop_loss)
returns_wo = MC_wo_stop_loss(S, r, vol, T, M, N)
    
fig, ax = plt.subplots()

ax.hist(returns_w, color="blue", bins=50, alpha=0.5, label="log Returns with Stop Loss", density=True)
ax.hist(returns_wo, color="red", bins=50, alpha=0.5, label="log Returns without Stop Loss", density=True)
ax.legend()

plt.show()