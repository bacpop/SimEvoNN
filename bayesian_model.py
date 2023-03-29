import numpy as np
import matplotlib.pyplot as plt
from SIR_model import simulate_stochastic_SIR, simulate_deterministic_SIR, plot_simulation

### Try to simulate transmisson (SIR) model with Bayesian inference
### https://researchonline.lshtm.ac.uk/id/eprint/4654745/1/1-s2.0-S175543651930026X-main.pdf

## Set-up ABC model - manually
def ABC_SIR(n_days, n_infected, beta, gamma, N, n_particles, distance_threshold,
            simulate_gamma=None, simulate_n_infected=None, simulate_beta=None, simulate_N=None):
    """

    :param n_days:
    :param n_infected:
    :param beta:
    :param gamma:
    :param N:
    :param n_particles:
    :param distance_threshold:
    :return: res: array of shape (n_particles, n_par_estimated+1)
    """
    par_to_estimate = {
        "simulate_gamma":simulate_gamma, 
        'simulate_n_infected':simulate_n_infected, 
        'simulate_beta':simulate_beta, 
        'simulate_N':simulate_N
    }
    n_par_estimated = len(par_to_estimate) # number of estimated parameters
    s_observed, i_observed, r_observed = simulate_deterministic_SIR(n_days=n_days, n_infected=n_infected, beta=beta, gamma=gamma, N=N)[1:]

    res = np.zeros((n_particles, n_par_estimated, 2)) # model parameter(s) and distance

    key_idx = 0
    for key, item in par_to_estimate.items():
        if not item: continue
        par_accepted_counter = 0  # count of accepted parameters
        par_all_counter = 0  # count of all proposed parameters
        while par_accepted_counter < n_particles:
            # Generate random parameters
            beta = np.random.uniform(0, 1) if key == 'simulate_beta' else beta
            gamma = np.random.uniform(0, 1) if key == 'simulate_gamma' else gamma
            n_infected = np.random.randint(1, 1000) if key == 'simulate_n_infected' else n_infected
            N = np.random.randint(10, 1000) if key == 'simulate_N' else N
            n_days = np.random.randint(10, 300) if key == 'simulate_n_days' else n_days
            # Simulate model ### prior distribution  ###TODO: how to implement summary statistics here?
            s_simulated, i_simulated, r_simulated = simulate_stochastic_SIR(n_days=n_days, n_infected=n_infected, beta=beta, gamma=gamma, N=N)[1:]
            # Calculate distance
            distance = np.linalg.norm(i_observed - i_simulated) ## Euclidean distance
            if distance < distance_threshold:
                par_value = beta if key == 'simulate_beta' else gamma if key == 'simulate_gamma' else n_infected if key == 'simulate_n_infected' else N
                res[par_accepted_counter, key_idx, :] = np.asarray([par_value,distance])
                par_accepted_counter += 1
                print(f'Accepted {par_accepted_counter} of {n_particles} for {key}. Distance: {distance}')

            par_all_counter += 1
        key_idx += 1
    return res, par_to_estimate

def plot_distances(res, par_to_estimate):
    #plt.hist(res[:,1])
    #plt.scatter(res[:,0], y)
    plt.figure(figsize=(18, 21))
    plt.subplots_adjust(hspace=0.5)
    idx = 0
    for parameter, bool_value in par_to_estimate.items():
        if not bool_value: continue
        # x --> parameter value
        # y --> 1/distance
        ax = plt.subplot(3,3 , idx+1)
        x, y = res[:,idx,0], 1 / res[:, idx, 1]  # inverse distance
        ax.scatter(x,y)
        ax.set_ylabel("1/Distance")
        ax.set_xlabel(f"{parameter} values")
        idx += 1
    plt.show()


if __name__ == "__main__":

    """plot_distances(*ABC_SIR(
        n_particles=100,
        distance_threshold = 500,
        beta = 0.3, #true values
        gamma = 0.1, #true values
        n_infected = 3, #true values
        N = 1000, #true values --> population size
        n_days = 150, #true values
        simulate_beta = True,
        simulate_gamma=True,
        simulate_n_infected=True,
        simulate_N=True
    ))"""

### Try ELFI ### only works with Python 3.7
import elfi
import scipy.stats as ss

####TODO: could not implement this function to ELFI. I could not understand vectorization of the model. May require some help to code.
def elfi_simulator(beta, gamma, n_infected, N, n_days, batch_size=1, random_state=None):
    # Simulate model
    #### INITIAL CONDITIONS ####
    gamma_vec = np.asanyarray(gamma).reshape(-1, 1)
    beta_vec = np.asanyarray(beta).reshape(-1, 1)
    n_recov_init = 0
    n_recov_vec = np.asanyarray(n_recov_init).reshape(-1, 1)
    n_infected_vec = np.asanyarray(n_infected)
    n_susceptible_vec = np.asanyarray(N - n_infected_vec - n_recov_vec).reshape(-1, 1)
    N_vec = np.asanyarray(N).reshape(-1, 1)

    random_state = random_state or np.random
    w = random_state.randn(batch_size, n_days)

    # define relationship between parameters --> deterministic
    #n_recov_vec = gamma_vec * n_infected_vec
    #n_infected_vec = beta_vec*n_susceptible_vec*n_infected_vec/N_vec - n_recov_vec
    #n_susceptible_vec = N_vec - n_infected_vec

    # Simulate model --> stochastic
    n_infected_vec = np.random.binomial(n_susceptible_vec, beta_vec*n_infected_vec/N_vec)
    n_recov_vec = np.random.binomial(n_infected_vec, gamma_vec)
    n_susceptible_vec = N_vec - n_infected_vec - n_recov_vec
    #s_simulated, i_simulated, r_simulated = simulate_stochastic_SIR(n_days=n_days, n_infected=n_infected, beta=beta, gamma=gamma, N=N)[1:]
    #ris_simulated = np.asarray([s_simulated, i_simulated, r_simulated])

    return np.asarray([n_susceptible_vec, n_infected_vec, n_recov_vec])

"""
print(elfi_simulator(
    beta=0.3,
    gamma=0.01,
    n_infected=9,
    N=1000,
    n_days=150
       )
    )"""
def elfi_parameter_trial():
    gamma = elfi.Prior('uniform', 0, 1)
    beta = elfi.Prior('uniform', 0, 1)

    gamma_init = 0.2
    beta_init = 0.4

    y_obs = simulate_stochastic_SIR(160, 5, beta_init,gamma_init, 1000)


    # Add the simulator node and observed data to the model
    sim = elfi.Simulator(simulate_stochastic_SIR, 160, 5, beta, gamma, 1000, observed=y_obs)

    # Add summary statistics to the model
    S1 = elfi.Summary(gamma, sim, name="S1")
    S2 = elfi.Summary(beta, sim, name="S2")

    # Specify distance as euclidean between summary vectors (S1, S2) from simulated and
    # observed data
    d = elfi.Distance('euclidean', S1, S2)

    # Plot the complete model (requires graphviz)
    elfi.draw(d, "elfi_SIR")

    # Run the rejection sampler
    rej = elfi.Rejection(d, batch_size=10000, seed=30052017)
    res = rej.sample(1000, threshold=.5)
    print(res)

    res.plot_marginals()
    plt.show()

#elfi_parameter_trial()