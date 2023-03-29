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


def simulate_deterministic_SIR(n_days,n_infected, N, beta, gamma):
    t=np.linspace(0, n_days, n_days)
    I0, R0 = n_infected, 0
    S0 = N - I0 - R0
    y0 = S0, I0, R0
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return t, S, I, R


def plot_SIR(t, S, I, R):
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


if __name__ == "__main__":
    plot_SIR(*simulate_deterministic_SIR(n_days=160, N=1000, beta=.3, gamma=.1, n_infected=2))

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


def simulate_stochastic_SIR(n_days, n_infected, beta, gamma, N, batch_size=None, random_state=None):
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

        if n_infected < 0:
            n_infected = 0
            #break

    return t, np.asarray(susceptible), np.asarray(infected), np.asarray(recovered)


def plot_simulation(infected, susceptible, recovered):
    days = range(len(infected))
    plt.plot(days, infected, label='Infected')
    plt.plot(days, susceptible, label='Susceptible')
    plt.plot(days, recovered, label='Recovered')
    plt.legend()
    plt.show()


def stochastic_SIR(n_days, n_infected, beta, gamma, N):
    plot_SIR(*simulate_stochastic_SIR(n_days, n_infected, beta, gamma, N))


if __name__ == "__main__":
    stochastic_SIR(n_days=160, n_infected=2, beta=.3, gamma=.1, N=50)


#### Deterministic SIR model with FW alleles and selection
"""
## Infectivity of a pathogen related the Allele that it carries
## generation of the pathogens should be selected from the infected population
## to initiate, let's say we have 2 alleles, A,B. A is more infective than the allele B.

infectivity_matrix = np.asarray([[0.4], [0.1]]) ## Infectivity of allele A and B
## This also means we need to separate the infected population, the one infected with A, and other B.
#(We may also want to include the population that is infected by both, and by new alleles (mutation))

beta = 0.8
gamma = 0.02
n_infected_A = 20
n_infected_B = 30
n_infected = n_infected_A + n_infected_B
infective_efficient_A = 0.9
infective_efficient_B = 0.1
N=1000
n_recovered_A = n_recovered_B = n_recovered = 0
n_susceptible_A = n_susceptible_B = n_susceptible = N - n_infected_A - n_infected_B
p_infectivity_A = beta*n_infected_A*infective_efficient_A/N
p_infectivity_B = beta*n_infected_B*infective_efficient_B/N
n_days = 365

new_infected_A = np.random.binomial(n_susceptible, p_infectivity_A)
new_infected_B = np.random.binomial(n_susceptible, p_infectivity_B)

infected, susceptible, recovered = [n_infected], [n_susceptible], [n_recovered]

for day in range(n_days):
    recovering_A = np.random.binomial(n_infected_A, gamma)
    recovering_B = np.random.binomial(n_infected_B, gamma)
    n_recovered_A += recovering_A
    n_recovered_B += recovering_B
    new_infected_A = np.random.binomial(n_susceptible_A, p_infectivity_A)
    new_infected_B = np.random.binomial(n_susceptible_B, p_infectivity_B)
    n_infected_A += new_infected_A - recovering_A
    n_infected_B += new_infected_B - recovering_B
    ###Add interference between strains
    n_susceptible_A = N - n_infected_A - recovering_A
    n_susceptible_B = N - n_infected_B - recovering_B
    #n_susceptible = N - n_infected_B - n_infected_A - recovering_A - recovering_B

    #if n_infected_A + n_infected_B + n_susceptible + n_recovered != N:
    #    raise ValueError('Population size not conserved')

    if n_susceptible < 0: n_susceptible = 0
    infected.append(n_infected_A + n_infected_B)
    susceptible.append(n_susceptible)
    recovered.append(n_recovered)


plot_simulation(infected, susceptible, recovered)

## Later we want to include allele A and B selections from infected population."""

## Try genetic diversity SIR paper:
## Same guy who introduces ELFI
### https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2653725/

