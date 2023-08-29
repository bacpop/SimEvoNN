from lib.bayesian_model import BOLFI4WF
from lib.simulator import simulator
import elfi
import pandas as pd
import numpy as np


def run_bolfi(args):

    ## Prepare truth data to bolfi class
    clean_df = pd.read_csv(args.observed_data)
    ss_list = list(clean_df.columns)[:-2] ## Remove last two columns (parameters)
    ss_indices = {ss: i for i, ss in enumerate(ss_list)}
    observed_ss = np.array(clean_df.iloc[0:args.batch_size, :-2])


    ## Prepare simulator params to bolfi class
    func_params = [args.fasta, args.n_generations, args.max_mutations]
    prior_params = {
    "Ne": {"distribution": str(args.ne_distribution), "min": args.n_individuals_min, "max": args.n_individuals_max},
    "mutation_rate": {"distribution": str(args.mu_distribution), "min": args.mutation_rate_min, "max": args.mutation_rate_max},
}

    bolfi = BOLFI4WF(
        ## Simulator params
        simulator_function=elfi.tools.vectorize(simulator),
        function_params=func_params,
        prior_params=prior_params,
        filter_allele_freq_below=args.filter_below,
        observed_data=observed_ss,
        summary_stats=ss_list,
        ss_indices=ss_indices,
        ## Bolfi params
        initial_evidence=args.initial_evidence,
        update_interval=args.update_interval,
        n_evidence=args.n_evidence,
        acq_noise_var=args.acq_noise_var,
        bounds=None,
        n_post_samples=args.n_post_samples,
        ###Misc
        number_of_batches=args.batch_size,
        work_dir=args.workdir,
        save_simulations=False,
    )

    bolfi.run_bolfi()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Bayesian Optimisation for Wright-Fisher simulator for Effective population size and mutation rate parameters')
    parser.add_argument('--observed_data', type=str, help='Path to observed data, csv format with summary statistics and last two columns contain Ne and Mu parameters', required=True)
    parser.add_argument('--fasta', type=str, help='Path to fasta file', required=True)
    parser.add_argument('--n_generations', type=int, help='Number of generations to simulate', required=False, default=50)
    parser.add_argument('--n_individuals_min', type=int, help='Minimum number of individuals to simulate', required=False, default=100)#default=5e+5)
    parser.add_argument('--n_individuals_max', type=int, help='Maximum number of individuals to simulate', required=False, default=5000)#default=5e+8)
    parser.add_argument('--mutation_rate_min', type=float, help='Minimum mutation rate to simulate', required=False, default=0.0001)
    parser.add_argument('--mutation_rate_max', type=float, help='Maximum mutation rate to simulate', required=False, default=0.1)
    parser.add_argument('--mu_distribution', type=str, help='Mutation rate distribution', required=False, default="uniform")
    parser.add_argument('--ne_distribution', type=str, help='Effective population size distribution', required=False, default="loguniform")
    parser.add_argument('--max_mutations', type=int, help='Maximum number of mutations to simulate', required=False, default=5000)
    parser.add_argument('--batch_size', type=int, help='Batch size to simulate', required=False, default=1)
    parser.add_argument('--filter_below', type=float, help='Filter allele frequency below this value', required=False, default=None)
    parser.add_argument('--workdir', type=str, help='Work directory to simulate', required=False, default=None)
    parser.add_argument('--initial_evidence', type=int, help='Initial evidence for BOLFI', required=False, default=300)
    parser.add_argument('--update_interval', type=int, help='Update interval of the acquisition function', required=False, default=1)
    parser.add_argument('--n_evidence', type=int, help='Number of evidence for BOLFI', required=False, default=1200)
    parser.add_argument('--acq_noise_var', type=float, help='Acquisition noise variance', required=False, default=0.00)
    parser.add_argument('--n_post_samples', type=int, help='Number of posterior samples', required=False, default=2000)
    args = parser.parse_args()

    run_bolfi(args)
