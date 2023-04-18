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


def plot_SIR(t, S, I, R, N=1000):
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel(f'Number ({N}s)')
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


def simulate_stochastic_SIR(n_days, n_infected, beta, gamma, N, lose_immunity=False,gain_immunity=False, batch_size=None, random_state=None):
    ##Initialize
    n_recovered = 0
    n_susceptible = N - n_infected - n_recovered
    infected, susceptible, recovered = [n_infected], [n_susceptible], [n_recovered]
    t = np.linspace(0, n_days, n_days)
    n_lost_immunity = 0
    n_gain_immunity = 0

    for i in range(1, n_days):
        recovering = recovery(n_infected, gamma)
        if lose_immunity:
            n_lost_immunity = np.random.binomial(n_recovered, beta/10)
        if gain_immunity:
            n_gain_immunity = np.random.binomial(n_susceptible, gamma*10)
        n_recovered += recovering - n_lost_immunity + n_gain_immunity
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


if __name__ == "__main__":
    plot_SIR(*simulate_stochastic_SIR(
        n_days=160,
        n_infected=4,
        beta=.3,
        gamma=.02,
        N=100,
        #lose_immunity=True,
        #gain_immunity=True
    ), N=100)


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

### Vectroized SIR model

def vectorised_SIR_simulator(beta: np.array, gamma: np.array, n_init_infected, N, n_days, n_obs=1, batch_size=1,
                             random_state=None):
    # Simulate model
    #### INITIAL CONDITIONS ####
    sir_matrix = np.ones([n_days, 3, n_obs]) * np.nan  ## days, [infected, recovered, susceptible], n_obs
    sir_matrix = sir_matrix.astype(np.int8)
    sir_matrix[0, 0, :] = n_init_infected
    sir_matrix[0, 1, :] = 0
    sir_matrix[0, 2, :] = N - n_init_infected

    random_state = random_state or np.random
    if len(beta) < n_obs or len(gamma) < n_obs:
        beta = np.repeat(beta, n_obs)
        gamma = np.repeat(gamma, n_obs)

    # obs = np.zeros([batch_size, n_obs]).astype(np.int16)

    for i in range(1, n_days):
        for j in range(n_obs):
            ## Deterministic model
            # sir_matrix[i,0,j] = sir_matrix[i-1,0,j] + beta*sir_matrix[i-1,0,j]*sir_matrix[i-1,2,j]/N - gamma*sir_matrix[i-1,0,j]
            # sir_matrix[i,1,j] = sir_matrix[i-1,1,j] + gamma*sir_matrix[i-1,0,j]
            # sir_matrix[i,2,j] = sir_matrix[i-1,2,j] - beta*sir_matrix[i-1,0,j]*sir_matrix[i-1,2,j]/N

            ## Stochastic model
            p_IR = 1 - math.e ** -gamma[j]
            p_SI = 1 - math.e ** (-beta[j] * sir_matrix[i - 1, 0, j] / N)
            recovering = random_state.binomial(sir_matrix[i - 1, 0, j], p_IR)
            sir_matrix[i, 0, j] = sir_matrix[i - 1, 0, j] + random_state.binomial(sir_matrix[i - 1, 2, j],
                                                                                  p_SI) - recovering
            sir_matrix[i, 1, j] = sir_matrix[i - 1, 1, j] + recovering
            sir_matrix[i, 2, j] = N - sir_matrix[i, 0, j] - sir_matrix[i, 1, j]

            assert sir_matrix[i, 1, j] + sir_matrix[i, 0, j] + sir_matrix[i, 2, j] == N, "Population size not conserved"

    return sir_matrix


def plot_sir_matrix(sir_matrix, separate_plots=False):
    plt.figure(figsize=(10, 5))
    plt.xlabel('Days')
    plt.ylabel('Number of people')

    if separate_plots:
        for i in range(sir_matrix.shape[2]):
            ax = plt.subplot(2, 3, i + 1)
            ax.plot(sir_matrix[:, 0, i], label='infected')
            ax.plot(sir_matrix[:, 1, i], label='recovered')
            ax.plot(sir_matrix[:, 2, i], label='susceptible')
    else:
        ## Plot all in one
        plt.plot(sir_matrix[:, 0, :], label='infected', c='r', alpha=0.5)
        plt.plot(sir_matrix[:, 1, :], label='recovered', c='g', alpha=0.5)
        plt.plot(sir_matrix[:, 2, :], label='susceptible', c='b', alpha=0.5)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()



## Try genetic diversity SIR paper:
## Same guy who introduces ELFI
### https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2653725/

import math


class SIR_Simulator:
    ##https://mrc-ide.github.io/odin.dust/articles/sir_models.html

    def __init__(self, beta, gamma, N, n_days, n_init_infected,):
        self.beta = beta
        self.gamma = gamma
        self.N = N
        self.n_days = n_days
        self.n_recovered = 0
        self.n_susceptible = N - n_init_infected
        self.n_infected = n_init_infected
        self.step = 0
        self.dt = 0

    def infection_event(self):
        return np.random.binomial(self.n_susceptible, 1-math.e**(-self.beta*self.dt*self.n_infected/self.N))

    def recovery_event(self):
        return np.random.binomial(self.n_infected, 1-math.e**(-self.gamma*self.dt))

    def next_event(self):
        recovering = self.recovery_event()
        self.n_recovered += recovering
        self.n_infected += self.infection_event() - recovering
        self.n_susceptible = self.N - self.n_infected - self.n_recovered

    def time_step(self):
        self.dt = np.random.exponential(scale=1/self.N)
        self.step += 1

    def simulate(self):
        for day in range(self.n_days):
            self.time_step()
            self.next_event()