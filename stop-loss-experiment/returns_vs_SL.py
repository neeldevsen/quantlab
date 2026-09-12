import matplotlib.pyplot as plt
import numpy as np

def MC_w_stop_loss(S0, r, vol, T, M, N, Z):
    stocks = np.zeros((M,N))
    returns = np.zeros((99,))
    dt = T/ N
    time = np.arange(1, N+1) * dt

    stocks[:,:] = S0 * np.exp((r - 0.5*vol**2)*time + vol * np.sqrt(dt) * np.cumsum(Z, axis=1))

    for i in range(1, 100):
        h = np.zeros((M,N))
        h[:,:] = stocks[:,:] < S0 * (1 - i/100)
        did_hit = np.any(h, axis=1)
        first_hit = np.argmax(h, axis=1)
        final_prices = stocks[:, -1].copy()

        rows = np.arange(M)

        final_prices[did_hit] = stocks[rows[did_hit], first_hit[did_hit]]
        returns[i-1] = np.mean(np.log(final_prices / S0))

    return returns




S = 100
r = 0.01
vol = 0.3
T = 1
M = 20000
N = 252

times = np.linspace(0, 1, 99)

returns_w = np.zeros((99, len(times)))

Z = np.random.standard_normal((M, N))

for i in range(0, 99):
    returns_w[: , i] =  MC_w_stop_loss(S, r, vol, times[i], M, N, Z)

stop_loss = np.linspace(1, 99, 99)

fig = plt.figure(facecolor="black")
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor("black")


x , y = np.meshgrid(stop_loss, times, indexing="ij")

ax.set_xlabel('Stop loss (%)')
ax.set_ylabel('Time (year)')
ax.set_zlabel('log Returns')

ax.xaxis.set_pane_color((0, 0, 0, 1))
ax.yaxis.set_pane_color((0, 0, 0, 1))
ax.zaxis.set_pane_color((0, 0, 0, 1))

ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.zaxis.label.set_color("white")

ax.tick_params(colors="white")


ax.plot_surface(x, y, returns_w, alpha=0.5, cmap="viridis", rstride=1, cstride=1, antialiased=True)

plt.show()