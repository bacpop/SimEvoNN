"""
This script contains simulator class for FW and SIR models combined.
"""

import numpy as np
import math


class Simulator:

    def __init__(self,
                 beta, gamma, population_size,
                 n_days, n_init_infected,
                 n_alleles,
                 random_state=None):

        ## Time parameters
        self.n_days = n_days
        self.step = 0

        ## Other parameters
        self.random_state = random_state or np.random

        ## SIR model parameters
        self.beta = beta
        self.gamma = gamma
        self.population_size = population_size
        self.n_recovered = 0
        self.n_susceptible = population_size - n_init_infected
        self.n_infected = n_init_infected
        self.SIR_matrix = self._init_SIR()

        ## FW model parameters
        self.n_alleles = n_alleles
        self.allele_dist_func=None
        self.allele_freq_matrix = self._init_FW()


    def _init_SIR(self):
        sir_matrix = np.zeros([self.n_days, 3]).astype(np.int8)  ## days, [infected, recovered, susceptible], ##n_obs
        sir_matrix[0, 0] = self.n_infected
        sir_matrix[0, 1] = self.n_recovered
        sir_matrix[0, 2] = self.population_size - self.n_infected - self.n_recovered
        return sir_matrix

    def _init_FW(self):
        self.allele_dist_func = self.random_state.multinomial if self.n_alleles > 2 else self.random_state.binomial
        allele_fraction_matrix = np.zeros([self.n_days, self.n_alleles]).astype(np.float64)
        allele_fraction_matrix[0, :] = 1 / self.n_alleles
        return allele_fraction_matrix

    def update_SIR(self):
        new_recovered = self.recovery_event()
        new_infected = self.infection_event()
        self.n_infected = self.n_infected + new_infected - new_recovered
        self.n_recovered = self.n_recovered + new_recovered
        self.n_susceptible = self.population_size - self.n_infected - self.n_recovered

        assert self.n_recovered + self.n_susceptible + self.n_infected == self.population_size, "Population size not conserved"

        self.SIR_matrix[self.step, 0] = self.n_infected
        self.SIR_matrix[self.step, 1] = self.n_recovered
        self.SIR_matrix[self.step, 2] = self.n_susceptible

    def _get_p_IR(self, recovery_param=1):
        return (1 - math.e ** -self.gamma) * recovery_param

    def _get_p_SI(self, infectivity_param=1):
        return (1 - math.e ** (-self.beta * self.n_infected / self.population_size)) * infectivity_param

    def recovery_event(self):
        return self.random_state.binomial(self.n_infected, self._get_p_IR())

    def infection_event(self):
        return self.random_state.binomial(self.n_susceptible, self._get_p_SI())

    def update_FW(self):
        self.allele_freq_matrix[self.step, :] = self.select_new_generation() / self.n_infected
        assert math.isclose(1, np.sum(self.allele_freq_matrix[self.step, :]), ), "Allele frequencies do not add up to 1"

    def select_new_generation(self):
        return self.allele_dist_func(self.n_infected, self._get_allele_probability())

    def _get_allele_probability(self,mutation_coefficient=None,selection_coefficient=None):
        return self.allele_freq_matrix[self.step-1, :]

    def simulate(self):
        for day in range(1, self.n_days):
            self.step = day
            self.update_FW()
            self.update_SIR()
            if self.n_infected == 0:
                break

from FW_model import plot_multiallele
from SIR_model import plot_sir_matrix

n_days = 60
n_alleles = 4

my_sim = Simulator(
    beta=0.7,
    gamma=0.1,
    population_size=100,
    n_days=n_days,
    n_init_infected=5,
    n_alleles=n_alleles
)

my_sim.simulate()

plot_multiallele(my_sim.allele_freq_matrix.reshape(n_days,1,n_alleles))

plot_sir_matrix(my_sim.SIR_matrix.reshape(n_days,3,1))
