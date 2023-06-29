import math

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from SIR_model import simulate_stochastic_SIR, simulate_deterministic_SIR, vectorised_SIR_simulator


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
        "simulate_gamma": simulate_gamma,
        'simulate_n_infected': simulate_n_infected,
        'simulate_beta': simulate_beta,
        'simulate_N': simulate_N
    }
    n_par_estimated = len(par_to_estimate)  # number of estimated parameters
    s_observed, i_observed, r_observed = simulate_deterministic_SIR(n_days=n_days, n_infected=n_infected, beta=beta,
                                                                    gamma=gamma, N=N)[1:]

    res = np.zeros((n_particles, n_par_estimated, 2))  # model parameter(s) and distance

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
            s_simulated, i_simulated, r_simulated = simulate_stochastic_SIR(n_days=n_days, n_infected=n_infected,
                                                                            beta=beta, gamma=gamma, N=N)[1:]
            # Calculate distance
            distance = np.linalg.norm(i_observed - i_simulated)  ## Euclidean distance
            if distance < distance_threshold:
                par_value = beta if key == 'simulate_beta' else gamma if key == 'simulate_gamma' else n_infected if key == 'simulate_n_infected' else N
                res[par_accepted_counter, key_idx, :] = np.asarray([par_value, distance])
                par_accepted_counter += 1
                print(f'Accepted {par_accepted_counter} of {n_particles} for {key}. Distance: {distance}')

            par_all_counter += 1
        key_idx += 1
    return res, par_to_estimate


def plot_distances(res, par_to_estimate):
    # plt.hist(res[:,1])
    # plt.scatter(res[:,0], y)
    plt.figure(figsize=(18, 21))
    plt.subplots_adjust(hspace=0.5)
    idx = 0
    for parameter, bool_value in par_to_estimate.items():
        if not bool_value: continue
        # x --> parameter value
        # y --> 1/distance
        ax = plt.subplot(3, 3, idx + 1)
        x, y = res[:, idx, 0], 1 / res[:, idx, 1]  # inverse distance
        ax.scatter(x, y)
        ax.set_ylabel("1/Distance")
        ax.set_xlabel(f"{parameter} values")
        idx += 1
    plt.show()

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

# print(vectorised_SIR_simulator(0.3, 0.1, 3, 1000, 10, 2))

from SIR_model import plot_sir_matrix


# plot_sir_matrix(vectorised_SIR_simulator([0.3,0.5,0.9], [0.01,0.2,0.7], 5, 50, 100, 3), separate_plots=True)


def elfi_parameter_trial(gamma_true, beta_true, n_infected_init=10, N=1000, n_days=128, number_of_batches=1,
                         rejection_threshold=0.5, n_samples=1000):
    gamma = elfi.Prior(ss.uniform, 0, 1)
    beta = elfi.Prior(ss.uniform, 0, 1)

    # gamma_init = np.array([0.2, 0.2, 0.2])
    gamma_true = [gamma_true]
    # beta_init = np.array([0.4, 0.4, 0.4])
    beta_true = [beta_true]

    y_obs = vectorised_SIR_simulator(
        beta=beta_true,
        gamma=gamma_true,
        n_init_infected=n_infected_init,
        N=N,
        n_days=n_days,
        n_obs=number_of_batches
    )

    # Add the simulator node and observed data to the model
    sim = elfi.Simulator(vectorised_SIR_simulator, beta, gamma, n_infected_init, N, n_days, number_of_batches,
                         observed=y_obs)

    # Add summary statistics to the model
    S1 = elfi.Summary(total_infected, sim, number_of_batches, name="total_infected_summary")
    S2 = elfi.Summary(peak_infected, sim, number_of_batches, name="peak_infected_summary")

    # Specify distance as euclidean between summary vectors (S1, S2) from simulated and
    # observed data
    d = elfi.Distance('euclidean', S1, S2)
    # d = elfi.Distance('seuclidean', S1, S2)

    # Plot the complete model (requires graphviz)
    # elfi.draw(d, filename="./elfi_SIR")

    # Run the rejection sampler
    rej = elfi.Rejection(d, batch_size=number_of_batches, seed=1)

    elfi.set_client('multiprocessing')
    res = rej.sample(n_samples, threshold=rejection_threshold)

    # np.save('gamma_data.npy', res.samples['gamma'])
    # np.save('beta_data.npy', res.samples['beta'])
    ## Tries to create different thresholds for each batch

    print(res.method_name, f"\nAcceptance rate: {res.meta['accept_rate']}\n Means:\n{res.sample_means_and_95CIs}", )

    res.plot_marginals()
    plt.show()


def summarise_recovery(x, n_batches=1):
    rv = np.array([sigmoid(-x[:, 1, j]) for j in range(n_batches)])
    return rv


def summarise_infected_sigmoid(x, n_batches=1):
    rv = np.array([sigmoid(x[:, 0, j]) for j in range(n_batches)])
    return rv


def summarise_infected(x, n_batches=1):
    r_array = np.zeros([n_batches, x.shape[0]])
    for j in range(n_batches):
        var_infected = np.var(x[:, 0, j])
        mean_infected = np.mean(x[:, 0, j])
        g = gaussian(x[:, 0, j], sigma=np.sqrt(var_infected), mu=mean_infected)
        r_array[j] = g

    return r_array


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def total_infected(x, n_batches=1):
    rv = np.array([np.sum(x[:, 0, j]) for j in range(n_batches)])
    return rv


def peak_infected(x, n_batches=1):
    rv = np.array([np.max(x[:, 0, j]) for j in range(n_batches)])
    return rv


def recovery_variance(x, n_batches=1):
    rv = np.array([np.var(x[:, 1, j]) for j in range(n_batches)])
    return rv


"""elfi_parameter_trial(gamma_true=0.1, beta_true=0.6, number_of_batches=3,
                     rejection_threshold=0.5, n_samples=100,
                     n_days=32, N=100, n_infected_init=3)"""


## Write BOLFI (Bayesian Optimisation LFI) code here

def create_elfi_model_SIR(
        gamma_true, beta_true, n_infected_init=3, N=100, n_days=32, number_of_batches=1
):
    gamma = elfi.Prior(ss.uniform, 0, 1)
    beta = elfi.Prior(ss.uniform, 0, 1)

    # gamma_init = np.array([0.2, 0.2, 0.2])
    gamma_true = [gamma_true]
    # beta_init = np.array([0.4, 0.4, 0.4])
    beta_true = [beta_true]

    y_obs = vectorised_SIR_simulator(
        beta=beta_true,
        gamma=gamma_true,
        n_init_infected=n_infected_init,
        N=N,
        n_days=n_days,
        n_obs=number_of_batches,
        deterministic=False
    )

    # Add the simulator node and observed data to the model
    sim = elfi.Simulator(vectorised_SIR_simulator, beta, gamma, n_infected_init, N, n_days, number_of_batches,
                         observed=y_obs, name="SIR_simulator")

    # Add summary statistics to the model
    # S1 = elfi.Summary(summarise_recovery, sim, number_of_batches, name="recovered_summary")
    # S2 = elfi.Summary(summarise_infected, sim, number_of_batches, name="infected_summary")

    S1 = elfi.Summary(total_infected, sim, number_of_batches, name="total_infected_summary")
    S2 = elfi.Summary(peak_infected, sim, number_of_batches, name="peak_infected_summary")
    S3 = elfi.Summary(recovery_variance, sim, number_of_batches, name="recovery_variance_summary")

    # Specify distance as euclidean between summary vectors (S1, S2) from simulated and
    # observed data
    d = elfi.Distance('euclidean', S1, S2, S3)

    return d


def bolfi_sir_trial():
    # Set an arbitrary global seed to keep the randomly generated quantities the same
    seed = 1
    np.random.seed(seed)
    elfi.set_client('multiprocessing')
    model = create_elfi_model_SIR(gamma_true=0.02, beta_true=0.3,n_days=160,n_infected_init=4, number_of_batches=1)
    #elfi.draw(model, filename="./master_bolfi_SIR")

    ## Take the log of the distance to reduce the influence of outliers on Gaussian Process
    log_distance = elfi.Operation(np.log, model)

    # Set up the inference method
    bolfi = elfi.BOLFI(
        log_distance, batch_size=1, initial_evidence=2000,
        update_interval=2, bounds={'beta': (0, 1), 'gamma': (0, 0.1)},
        seed=seed)

    # Fit the surrogate model
    post = bolfi.fit(n_evidence=2000)
    post2 = bolfi.extract_posterior(-1.)

    print(bolfi.target_model)

    # Plot the results
    bolfi.plot_state()
    bolfi.plot_discrepancy()
    #plt.show()
    post.plot(logpdf=True)
    post2.plot(logpdf=True)

    ### Sample from the posterior
    result_BOLFI = bolfi.sample(3000, algorithm='metropolis')

    print(result_BOLFI)

    result_BOLFI.plot_traces()
    result_BOLFI.plot_marginals()
    plt.show()


# bolfi_sir_trial()

## Fisher Wright model bolfi trial
from FW_model import fisher_wright_simulator


def bolfi_FW_trial():
    seed = 1
    np.random.seed(seed)
    elfi.set_client('multiprocessing')
    model = create_elfi_model_FW(mutation_true=0.1, n_individuals=100, n_generations=50,
                                 number_of_batches=1, n_alleles=1)

    # elfi.draw(model, filename="./bolfi_FW")

    ## Take the log of the distance to reduce the influence of outliers on Gaussian Process
    log_distance = elfi.Operation(np.log, model)

    # Set up the inference method
    bolfi = elfi.BOLFI(
        log_distance, batch_size=1, initial_evidence=500,
        update_interval=1, #bounds={'mutation': (0.0, 0.5)},
        seed=seed, )#acq_noise_var=[0.0, 0.0])

    # Fit the surrogate model
    post = bolfi.fit(n_evidence=1200)
    post2 = bolfi.extract_posterior(-1.)

    print(bolfi.target_model)

    # Plot the results
    #bolfi.plot_state()
    bolfi.plot_discrepancy()
    plt.show()
    post.plot(logpdf=True)
    post2.plot(logpdf=True)

    ### Sample from the posterior
    result_BOLFI = bolfi.sample(2000, algorithm='metropolis')

    print(result_BOLFI)

    result_BOLFI.plot_traces()
    result_BOLFI.plot_marginals()
    plt.show()


def create_elfi_model_FW(
        mutation_true=0.3, n_individuals=100, n_generations=100,
        number_of_batches=1, n_alleles=3,
        random_state=None, batch_size=None
):
    mutation = elfi.Prior(ss.uniform, 0, 1)
    mutation_true = [mutation_true]
    y_obs = fisher_wright_simulator(
        n_alleles=n_alleles,
        n_repeats=number_of_batches,
        n_generations=n_generations,
        n_individuals=n_individuals,
        mutation_rates=mutation_true,
        plot=False,
        set_allele_freq_equal=True,
    )

    ##n_repeats, n_generations, n_individuals, n_alleles=None,alleles=None, mutation_rates:np.array=None,
    # Add the simulator node and observed data to the model
    sim = elfi.Simulator(
        fisher_wright_simulator,
        number_of_batches, n_generations, n_individuals, n_alleles, None, mutation,
        observed=y_obs, name="FW_simulator"
    )

    # Add summary statistics to the model
    S1 = elfi.Summary(sumarise_mean, sim, n_alleles, number_of_batches, name="allele_freq_mean_summary")
    S2 = elfi.Summary(summarise_variance, sim, n_alleles, number_of_batches, name="allele_freq_var_summary")
    # Specify distance as euclidean between summary vectors (S1, S2) from simulated and
    # observed data
    d = elfi.Distance('euclidean', S1, S2)
    #elfi.draw(d, filename="./bolfi_mut_only_FW")
    return d


def summarise_variance(allele_freqs, n_alleles, n_batches=1):
    rv = [np.array([np.var(allele_freqs[:, b, i]) for i in range(n_alleles)]) for b in range(n_batches)]
    return rv


def sumarise_mean(allele_freqs, n_alleles, n_batches=1):
    rv = [np.array([np.mean(allele_freqs[:, b, i]) for i in range(n_alleles)]) for b in range(n_batches)]
    return rv


#bolfi_FW_trial()


from typing import Callable, Dict, List, Optional, Tuple, Union
import tempfile


class BOLFI4WF:

    def __init__(self, simulator_function: Callable, function_params: List, prior_params: Dict,
                 number_of_batches=1, observed_data=None, random_state=None, model=None,
                 filter_allele_freq_below:float=0.0, bounds=None, summary_stats=None, ss_indices=None,
                 initial_evidence=30, update_interval=1,n_evidence=100, acq_noise_var=0.0, n_post_samples=10,
                 _seed=1, distance_name="euclidean", sampling_algorithm="metropolis",
                 work_dir = None, save_simulations=False):

        ###Others
        self._seed = _seed
        np.random.seed(self._seed)
        self._work_dir = work_dir if work_dir is not None else tempfile.mkdtemp()
        self.save_simulations = save_simulations
        #elfi.set_client("multiprocessing")

        ##Simulator model
        self.simulator_function = simulator_function
        self.function_params = function_params
        self.prior_params = prior_params
        self.priors = self.create_priors()
        self.filter_allele_freq_below = filter_allele_freq_below

        ### Wright-Fisher Parameters to be estimated
        # self.Ne = Ne # Effective population size,
        # self.mutation_rate = mutation_rate

        ### Observed/Real data to be used
        self.observed_data = observed_data
        self.summary_stats = summary_stats
        self.ss_indices = ss_indices

        ### ELFI tool parameters
        self.number_of_batches = number_of_batches
        self.random_state = random_state or np.random

        ### BOLFI Model to be used, if None then create a new one
        self.model = model if model is not None else self.create_elfi_model()
        self.summaries = None
        self.distance_name = distance_name
        self.sampling_algorithm = sampling_algorithm
        self.distance = None

        ##Bolfi inference params
        self.initial_evidence = initial_evidence
        self.update_interval = update_interval
        self.n_evidence = n_evidence
        self.acq_noise_var = acq_noise_var
        self.bounds = bounds
        self.n_post_samples:int = n_post_samples

    def run_bolfi(self):
        self.get_elfi_summary_statistics(stats_to_summarise=self.summary_stats)
        self.get_elfi_distance()
        self.get_bolfi_inference_model()

    def create_priors(self):
        prior_objs: List[elfi.Prior] = []
        for param_name, param in self.prior_params.items():
            prior_objs.append(elfi.Prior(param["distribution"], param["min"], param["max"], name=param_name))
        return prior_objs

    def create_elfi_model(self, name=None):

        ##n_repeats, n_generations, n_individuals, n_alleles=None,alleles=None, mutation_rates:np.array=None,
        # Add the simulator node and observed data to the model
        return elfi.Simulator(
            self.simulator_function,
            *self.priors,
            *self.function_params,
            self._work_dir,
            self.save_simulations,
            self.filter_allele_freq_below,
            self.number_of_batches,
            self.ss_indices,
            observed=self.observed_data,
            name=name,
        )

    def get_elfi_summary_statistics(self, stats_to_summarise=None):
        from config import SS_INDICES
        sumstats = list(SS_INDICES.keys()) if stats_to_summarise is None else stats_to_summarise
        self.summaries = [elfi.Summary(self._get_summary_column, self.model, stat_key, self.ss_indices, name=stat_key) for stat_key in
                          sumstats]

    def get_elfi_distance(self):
        self.distance = elfi.Distance(self.distance_name, *self.summaries)
        #elfi.draw(self.distance, filename='Wright-Fisher_ELFI_model')

    def get_bolfi_inference_model(self):
        log_distance = self._log_distance()

        bounds = self.bounds or self._get_bounds_from_priors()
        # Set up the inference method
        bolfi = elfi.BOLFI(#self.model,
            log_distance,
            batch_size=self.number_of_batches,
            initial_evidence=self.initial_evidence,
            update_interval=self.update_interval, bounds=bounds,
            seed=self._seed, acq_noise_var=self.acq_noise_var
        )

        # Fit the surrogate model
        post = bolfi.fit(n_evidence=self.n_evidence)
        post2 = bolfi.extract_posterior(-1.)

        print(bolfi.target_model)

        # Plot the results
        bolfi.plot_state()
        bolfi.plot_discrepancy()
        plt.show()
        post.plot(logpdf=True)
        post2.plot(logpdf=True)

        ### Sample from the posterior
        result_BOLFI = bolfi.sample(self.n_post_samples, algorithm=self.sampling_algorithm)

        print(result_BOLFI)
        ### TODO: add methods for saving plots and model fit
        result_BOLFI.plot_traces()
        result_BOLFI.plot_marginals()
        plt.show()
        elfi.ElfiModel.save(self.model,self._work_dir)

    def _get_bounds_from_priors(self):
        return {p_name: (p["min"], p["max"]) for p_name, p in self.prior_params.items()}

    def _log_distance(self):
        ## Take the log of the distance to reduce the influence of outliers on Gaussian Process
        return elfi.Operation(np.log, self.distance)

    def _get_observed_data(self, **kwargs):
        return self.simulator_function(**kwargs)

    @staticmethod
    def _get_summary_column(y, column, ss_indices=None):
        from config import SS_INDICES
        ss_indices = SS_INDICES if ss_indices is None else ss_indices
        #indices_dict = {"max_H" : 0, "min_H": 1, "a_BL_mean": 2, "a_BL_median": 3}
        #return np.array([y[0][column]])
        y = y.reshape([len(ss_indices)])
        return y[ss_indices[column]]

"""
##True inputs
#{"initial_allele_seq": ["AAGTTCAAAGTGT"], "n_individuals": 100, "n_generations": 20, "mutation_rates": 0.6, "max_mutation_size": 100}

##Outputs
{"max_H": 0.2352274456166919, "min_H": 0.0, "a_BL_mean": 0.08888538418754201, "a_BL_median": 0.08632864864667722}

import json
tru_f = '/Users/berk/Projects/jlees/data/simulations/20230428-1653/Sim_99/tree_stats.json'
with open(tru_f) as fh:
    tru_dict = json.load(fh)

observed = np.zeros([1, len(list(tru_dict.values()))])
observed[0,:] = np.array(list(tru_dict.values()))
observed = np.array([0.2352274456166919, 0.0, 0.08888538418754201, 0.08632864864667722], dtype=
[
    ("max_H", np.float64),
    ("min_H", np.float64),
    ("a_BL_mean", np.float64),
    ("a_BL_median", np.float64),
])"""

#import gzip
#with gzip.open("/Users/berk/Projects/jlees/trial/pneumoniea_5_results.npy.gz", "rb") as f:
#    observed = np.frombuffer(f.read(), dtype=np.float64)

observed = np.load("/Users/berk/Projects/jlees/vs_codon/pneumoniea/pneumoniea_1_trial_results.npy")
from lib.simulator import simulator
from config import DATA_PATH
import os
import time
workdir = os.path.join(DATA_PATH, "BOLFI", "simulations",  str(time.strftime("%Y%m%d-%H%M")))

prior_params = {
    "Ne": {"distribution": "uniform", "min": 10, "max": 1000},
    "mutation_rate": {"distribution": "uniform", "min": 0.0, "max": 1.0},
}

##### input_fasta, n_generations, max_mutations
#function_params = ["/Users/berk/Projects/jlees/data/WF_input.fasta", 50, 500]
function_params = ["/Users/berk/Projects/jlees/data/Streptococcus_pneumoniae.fasta", 50, 5000]
batch_size = 1

import pandas as pd

clean_df = pd.read_csv("/Users/berk/Documents/cleaned_data.csv")
ss_list = list(clean_df.columns)[:-2]
ss_indices = {ss: i for i, ss in enumerate(ss_list)}

observed_ss = np.array(clean_df.iloc[0,:-2])
trues = clean_df.iloc[0,-2:]
print(trues)
bolfi_wf = BOLFI4WF(
    ## Simulator params
    simulator_function=elfi.tools.vectorize(simulator),
    function_params=function_params,
    prior_params=prior_params,
    filter_allele_freq_below=None,
    observed_data=observed_ss,
    summary_stats=ss_list,
    ss_indices=ss_indices,
    ## Bolfi params
    initial_evidence=100,
    update_interval=5,
    n_evidence=400,
    acq_noise_var=0.0,
    bounds=None,
    n_post_samples=800,
    ###Misc
    number_of_batches=batch_size,
    work_dir=workdir,
    save_simulations=False,
)

bolfi_wf.run_bolfi()