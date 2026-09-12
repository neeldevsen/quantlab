import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

def heston_naive_euler_call(data, M, N):
    dt = data.maturity / N

    # Current state of all M paths
    stock = np.full(M, data.spot, dtype=float)
    variance = np.full(M, data.variance, dtype=float)

    # Store full trajectories
    stock_paths = np.empty((N + 1, M))
    variance_paths = np.empty((N + 1, M))

    stock_paths[0] = stock
    variance_paths[0] = variance

    for n in range(N):
        Z1 = np.random.normal(size=M)
        Z3 = np.random.normal(size=M)

        # Correlated Brownian shock for variance
        Z2 = (
            data.rho * Z1
            + np.sqrt(1.0 - data.rho**2) * Z3
        )

        stock_next = (
            stock
            + data.rate * stock * dt
            + stock * np.sqrt(variance * dt) * Z1
        )

        variance_next = (
            variance
            + data.kappa * (data.theta - variance) * dt
            + data.xi * np.sqrt(variance * dt) * Z2
        )

        stock = stock_next
        variance = variance_next

        stock_paths[n + 1] = stock
        variance_paths[n + 1] = variance

    payoff = np.maximum(stock - data.strike, 0.0)

    price = (
        np.exp(-data.rate * data.maturity)
        * np.mean(payoff)
    )

    return price, stock_paths, variance_paths

def heston_fte_call(data, M, N):
    dt = data.maturity / N

    # Current state of all M paths
    stock = np.full(M, data.spot, dtype=float)
    variance = np.full(M, data.variance, dtype=float)

    # Store full trajectories
    stock_paths = np.empty((N + 1, M))
    variance_paths = np.empty((N + 1, M))

    stock_paths[0] = stock
    variance_paths[0] = variance

    for n in range(N):
        Z1 = np.random.normal(size=M)
        Z2 = np.random.normal(size=M)

        Zv = (
            data.rho * Z1
            + np.sqrt(1.0 - data.rho**2) * Z2
        )

        variance_positive = np.maximum(variance, 0.0)

        stock_next = (
            stock
            + data.rate * stock * dt
            + stock
            * np.sqrt(variance_positive * dt)
            * Z1
        )

        variance_next = (
            variance
            + data.kappa
            * (data.theta - variance_positive)
            * dt
            + data.xi
            * np.sqrt(variance_positive * dt)
            * Zv
        )

        stock = stock_next
        variance = variance_next

        stock_paths[n + 1] = stock
        variance_paths[n + 1] = variance

    payoff = np.maximum(stock - data.strike, 0.0)

    price = (
        np.exp(-data.rate * data.maturity)
        * np.mean(payoff)
    )

    return price, stock_paths, variance_paths

def heston_milstein_call(data, M, N):
    dt = data.maturity / N

    # Current state of all M paths
    stock = np.full(M, data.spot, dtype=float)
    variance = np.full(M, data.variance, dtype=float)

    # Store full trajectories
    stock_paths = np.empty((N + 1, M))
    variance_paths = np.empty((N + 1, M))

    stock_paths[0] = stock
    variance_paths[0] = variance

    for n in range(N):

        Z1 = np.random.normal(size=M)
        Z3 = np.random.normal(size=M)

        # Correlated normal for variance process
        Z2 = (
            data.rho * Z1
            + np.sqrt(1.0 - data.rho**2) * Z3
        )

        variance_positive = np.maximum(variance, 0.0)

        # Important: use old stock/variance values for both updates
        stock_next = (
            stock
            + data.rate * stock * dt
            + stock * np.sqrt(variance_positive * dt) * Z1
            + 0.5
            * stock
            * variance_positive
            * dt
            * (Z1**2 - 1.0)
        )

        variance_next = (
            variance
            + data.kappa
            * (data.theta - variance_positive)
            * dt
            + data.xi
            * np.sqrt(variance_positive * dt)
            * Z2
            + 0.25
            * data.xi**2
            * dt
            * (Z2**2 - 1.0)
        )

        stock = stock_next
        variance = variance_next

        stock_paths[n + 1] = stock
        variance_paths[n + 1] = variance

    payoff = np.maximum(stock - data.strike, 0.0)

    price = (
        np.exp(-data.rate * data.maturity)
        * np.mean(payoff)
    )

    return price, stock_paths, variance_paths

def heston_qe_call(data, M, N, threshold=1.5, gamma1=0.5, gamma2=0.5):
    dt = data.maturity / N

    exp_factor = np.exp(-data.kappa * dt)

    gamma1_dt = gamma1 * dt
    gamma2_dt = gamma2 * dt

    K0 = -data.rho * data.kappa * data.theta * dt / data.xi

    K1 = (
        gamma1_dt * (data.kappa * data.rho / data.xi - 0.5)
        - data.rho / data.xi
    )

    K2 = (
        gamma2_dt * (data.kappa * data.rho / data.xi - 0.5)
        + data.rho / data.xi
    )

    K3 = gamma1_dt * (1.0 - data.rho**2)
    K4 = gamma2_dt * (1.0 - data.rho**2)

    # Current state of all M paths
    X = np.full(M, np.log(data.spot), dtype=float)
    variance = np.full(M, data.variance, dtype=float)

    # Store every timestep for every path
    stock_paths = np.empty((N + 1, M))
    variance_paths = np.empty((N + 1, M))

    stock_paths[0] = data.spot
    variance_paths[0] = data.variance

    for n in range(N):

        s_squared = (
            variance
            * data.xi**2
            * exp_factor
            * (1.0 - exp_factor)
            / data.kappa
            +
            data.theta
            * (data.xi * (1.0 - exp_factor))**2
            / (2.0 * data.kappa)
        )

        m = (
            data.theta
            + (variance - data.theta) * exp_factor
        )

        phi = s_squared / m**2

        Z1 = np.random.normal(size=M)
        Z2 = np.random.normal(size=M)

        variance_next = np.empty(M)

        # -------------------------
        # Quadratic branch
        # -------------------------

        quadratic = phi < threshold

        phi_q = phi[quadratic]
        m_q = m[quadratic]

        b_squared = (
            2.0 / phi_q
            - 1.0
            + np.sqrt(
                (2.0 / phi_q)
                * (2.0 / phi_q - 1.0)
            )
        )

        a = m_q / (1.0 + b_squared)

        variance_next[quadratic] = (
            a
            * (np.sqrt(b_squared) + Z1[quadratic])**2
        )

        # -------------------------
        # Exponential branch
        # -------------------------

        exponential = ~quadratic

        phi_e = phi[exponential]
        m_e = m[exponential]

        p = (phi_e - 1.0) / (phi_e + 1.0)
        beta = (1.0 - p) / m_e

        u = np.random.uniform(size=np.sum(exponential))

        variance_next[exponential] = np.where(
            u <= p,
            0.0,
            np.log((1.0 - p) / (1.0 - u)) / beta
        )

        # -------------------------
        # Log-stock update
        # -------------------------

        X += (
            data.rate * dt
            + K0
            + K1 * variance
            + K2 * variance_next
            + np.sqrt(
                np.maximum(
                    K3 * variance + K4 * variance_next,
                    0.0
                )
            ) * Z2
        )

        variance = variance_next

        # Save this timestep
        stock_paths[n + 1] = np.exp(X)
        variance_paths[n + 1] = variance

    terminal_stock = stock_paths[-1]

    payoff = np.maximum(
        terminal_stock - data.strike,
        0.0
    )

    price = (
        np.exp(-data.rate * data.maturity)
        * np.mean(payoff)
    )

    return price, stock_paths, variance_paths

@dataclass
class HestonData:
    spot: float
    strike: float
    rate: float
    maturity: float
    variance: float
    kappa: float
    theta: float
    xi: float
    rho: float

data = HestonData(
    spot=100.0,
    strike=100.0,
    rate=0.05,
    maturity=1.0,
    variance=0.04,
    kappa=2.0,
    theta=0.04,
    xi=0.6,
    rho=-0.7
)

M = 1000000
N = 252

price_euler, S_euler, v_euler = heston_naive_euler_call(data, M, N)
price_fte, S_fte, v_fte = heston_fte_call(data, M, N)
price_milstein, S_milstein, v_milstein = heston_milstein_call(data, M, N)
price_qe, S_qe, v_qe = heston_qe_call(
    data,
    M,
    N,
    threshold=1.5,
    gamma1=0.5,
    gamma2=0.5
)

time = np.linspace(0.0, data.maturity, N + 1)[:, None]


fig, ax = plt.subplots()

plt.plot(time, np.mean(v_euler, axis=1), color="lime", label="Euler Scheme")
plt.plot(time, np.mean(v_fte, axis=1), color="cyan", label="Full Truncation Euler")
plt.plot(time, np.mean(v_milstein, axis=1), color="red", label="Milstein Model")
plt.plot(time, np.mean(v_qe, axis=1), color="orange", label="Quadratic-Exponential")
plt.axhline(y=0.04, color="white", linestyle="--", linewidth=1.5, label=r"$\theta = 0.04$")

fig.patch.set_facecolor("black")
ax.set_facecolor("black")
plt.style.use("dark_background")

ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Variance")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")

for spine in ax.spines.values():
    spine.set_color("white")


plt.legend()
plt.show()

plt.figure(figsize=(12, 6))

plt.hist(
    v_qe[-1],
    bins=150,
    density=True,
    alpha=0.7,
    label="Quadractic-Exponential",
    color="yellow"
)

plt.axvline(
    x=0.04,
    color="white",
    linestyle="--",
    label=r"$\theta = 0.04$"
)

plt.xlabel("Terminal Variance")
plt.ylabel("Density")
plt.title("QE Terminal Variance Distribution")


plt.show()

plt.figure(figsize=(12, 6))

plt.hist(
    v_milstein[-1],
    bins=150,
    density=True,
    alpha=0.7,
    label="Milstein Model",
    color="red"
)

plt.axvline(
    x=0.04,
    color="white",
    linestyle="--",
    label=r"$\theta = 0.04$"
)

plt.xlabel("Terminal Variance")
plt.ylabel("Density")
plt.title("Milstein Model Terminal Variance Distribution")

plt.show()

plt.figure(figsize=(12, 6))

plt.hist(
    v_euler[-1],
    bins=150,
    density=True,
    alpha=0.7,
    label="Euler Scheme",
    color="lime"
)

plt.axvline(
    x=0.04,
    color="white",
    linestyle="--",
    label=r"$\theta = 0.04$"
)

plt.xlabel("Terminal Variance")
plt.ylabel("Density")
plt.title("Euler Scheme Terminal Variance Distribution")

plt.show()

plt.figure(figsize=(12, 6))

plt.hist(
    v_fte[-1],
    bins=150,
    density=True,
    alpha=0.7,
    label="Full Truncation Euler",
    color="cyan"
)

plt.axvline(
    x=0.04,
    color="white",
    linestyle="--",
    label=r"$\theta = 0.04$"
)


plt.xlabel("Terminal Variance")
plt.ylabel("Density")
plt.title("Full Truncation Euler Terminal Variance Distribution")

plt.legend()
plt.show()