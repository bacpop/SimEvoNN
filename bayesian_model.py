import numpy as np
import matplotlib.pyplot as plt
from SIR_model import simulate_stochastic_SIR, simulate_deterministic_SIR, plot_simulation

### Try to simulate transmisson (SIR) model with Bayesian inference


## Set-up ABC model - manually

n_particles = 10
distance_threshold = 50
n_par_estimated = 1  # Number of parameters to estimate

beta = 0.3
gamma = 0.18

n_infected = 2
s_observed, i_observed, r_observed = simulate_deterministic_SIR(n_days=160, n_infected=n_infected, beta=beta, gamma=gamma, N=1000)[1:]

res = np.zeros((n_particles, n_par_estimated+1)) # model parameter(s) and distance

par_accepted_counter = 0 # count of accepted parameters
par_all_counter = 0 # count of all proposed parameters

while par_accepted_counter < n_particles:
    # Generate random parameters
    #beta = np.random.uniform(0, 1)
    gamma = np.random.uniform(0, 1)
    #n_infected = np.random.randint(0, 1000)
    # Simulate model ### prior distribution
    s_simulated, i_simulated, r_simulated = simulate_stochastic_SIR(n_days=160, n_infected=n_infected, beta=beta, gamma=gamma, N=1000)[1:]
    # Calculate distance
    distance = np.linalg.norm(i_observed - i_simulated)
    if distance < distance_threshold:
        res[par_accepted_counter, :] = np.asarray([gamma,distance])
        par_accepted_counter += 1
        print(f'Accepted {par_accepted_counter} of {n_particles}. Distance: {distance}')

    par_all_counter += 1

def plot_distances(res):
    #plt.hist(res[:,1])
    y=1/res[:,1] # inverse distance
    plt.scatter(res[:,0], y)
    plt.ylabel('1/Distance')
    plt.xlabel('Gamma')
    plt.show()

plot_distances(res)