import logging
import time
import os
logger = logging.getLogger(f'WFsim_{time.strftime("%Y%m%d-%H%M")}')
logger.setLevel(logging.ERROR)

def save_results(resulting_matrix, output_dir):
    import numpy as np
    import pandas as pd
    from config import ALL_INDICES

    from utils import call_subprocess
    out_path = os.path.join(output_dir, f"{args.sim_name}_results")
    numpy_path = f"{out_path}.npy"
    pd_path = f"{out_path}.csv"
    np.save(numpy_path, resulting_matrix)
    pd.DataFrame(resulting_matrix, columns=ALL_INDICES.keys()).to_csv(pd_path, index=False)
    if args.compress:
        call_subprocess("gzip", [numpy_path, "--force"])
        call_subprocess("gzip", [pd_path, "--force"])

def run_simulator(args):
    from lib.simulator import simulator
    from config import FASTA_IN, DATA_PATH


    ## Check input arguments
    input_fasta = args.input_fasta if args.input_fasta is not None else FASTA_IN
    output_dir = args.outdir if args.outdir is not None else os.path.join(DATA_PATH, "simulation_results")
    os.mkdir(output_dir) if not os.path.exists(output_dir) else None
    batch_size = args.batch_size
    total_simulations = args.n_simulations * batch_size
    if total_simulations == 0:
        raise Exception("Batch size or n_simulations cannot be 0")

    ## Run simulations
    m = simulator(
                n_generations=args.n_generations,
                n_individuals=args.n_individuals, ## Class randomly generates this
                max_mutations=args.max_mutations,
                batch_size=args.batch_size,
                n_repeats=args.n_simulations,
                mutation_rate=args.mutation_rate, ## Class randomly generates this
                input_fasta=input_fasta,
                work_dir=args.workdir,
                outdir=output_dir,
                filter_below=args.filter_below,
                save_data=args.save_all_data,
                add_parameters=True
            )

    ##Save simulation results
    save_results(m,output_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Wright-Fisher simulator for Effective population size and mutation rate parameters')
    parser.add_argument('--n_simulations', type=int, help='Number of simulations to run', required=True, default=5)
    parser.add_argument('--sim_name', type=str, help='Name of your simulation', required=False, default="simulation")
    parser.add_argument('--input_fasta', type=str, help='Input fasta file. If not provided, uses a predefined one', required=False, default=None)
    parser.add_argument('--n_generations', type=int, help='Number of generations to simulate', required=False, default=20)
    parser.add_argument('--n_individuals', type=int, help='Number of individuals to simulate', required=False, default=None)
    parser.add_argument('--mutation_rate', type=float, help='Mutation rate to simulate', required=False, default=None)
    parser.add_argument('--max_mutations', type=int, help='Maximum number of mutations to simulate', required=False, default=200)
    parser.add_argument('--batch_size', type=int, help='Batch size to simulate', required=False, default=1)
    parser.add_argument('--workdir', type=str, help='Work directory to simulate', required=False, default=None)
    parser.add_argument('--filter_below', type=float, help='Filter below', required=False, default=None)
    parser.add_argument('--outdir', type=str, help='Output directory for simulations', required=False, default=None)
    parser.add_argument('--compress', action='store_true', help='Compress the output files', required=False)
    parser.add_argument('--save_all_data', action='store_true', help='Save all data regarding the simulations', required=False)

    args = parser.parse_args()

    import timeit
    start = timeit.default_timer()
    run_simulator(args)
    stop = timeit.default_timer()
    eplased_time = stop - start
    print(f"Time taken: {eplased_time/60} minutes")

