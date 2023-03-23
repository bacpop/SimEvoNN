import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#### Deterministic SIR model.

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def simulate_deterministic_SIR(days, N, beta, gamma, I):
    t=np.linspace(0, days, days)
    I0, R0 = 1, 0
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return t, S, I, R


def plot_deterministic_SIR(t, S, I, R):
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    #ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(True)
    plt.show()


plot_deterministic_SIR(*simulate_deterministic_SIR(days=160, N=1000, beta=.3, gamma=.1, I=2))

### Stochastic SIR model
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7094774/


def infection_prob(beta, N, I):
    return beta*I/N


def recovery_prob(gamma):
    return gamma


def infection(n_infected, n_susceptible, beta, N):
    return np.random.binomial(n_susceptible, infection_prob(beta, N, n_infected))


def recovery(n_infected, gamma):
    return np.random.binomial(n_infected, recovery_prob(gamma))


def simulate(n_days, n_infected, beta, gamma, N):
    ##Initialize
    n_recovered = 0
    n_susceptible = N - n_infected - n_recovered
    infected, susceptible, recovered = [n_infected], [n_susceptible], [n_recovered]
    t = np.linspace(0, n_days, n_days)

    for i in range(1, n_days):
        recovering = recovery(n_infected, gamma)
        n_recovered += recovering
        n_infected += infection(n_infected, n_susceptible, beta, N) - recovering
        n_susceptible = N - n_infected - n_recovered
        if n_infected + n_susceptible + n_recovered != N:
            raise ValueError('Population size not conserved')

        if n_susceptible < 0: n_susceptible = 0
        infected.append(n_infected)
        susceptible.append(n_susceptible)
        recovered.append(n_recovered)

        if n_infected <= 0:
            break

    return infected, susceptible, recovered


def plot_simulation(infected, susceptible, recovered):
    days = range(len(infected))
    plt.plot(days, infected, label='Infected')
    plt.plot(days, susceptible, label='Susceptible')
    plt.plot(days, recovered, label='Recovered')
    plt.legend()
    plt.show()


def stochastic_SIR(n_days, n_infected, beta, gamma, N):
    plot_simulation(*simulate(n_days, n_infected, beta, gamma, N))


stochastic_SIR(n_days=160, n_infected=2, beta=.3, gamma=.1, N=1000)
