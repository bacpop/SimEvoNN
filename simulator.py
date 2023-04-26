"""
This script contains simulator class for FW with mutations and phylogenetic tree models combined.
"""

import numpy as np
from phylogenetic_tree import PhyloTree
from FW_model import FWSim

import os
import time
from config import DATA_PATH, PROJECT_PATH
from utils import call_subprocess
from typing import List, Dict


class Simulator:
    """
    This simulator class calls FWSim class that takes fasta inputs, runs Fisher-Wright Simulations with infinite mutations
    (i.e. de novo mutations). From the resulting sequences, an external tool MAPLE is called to construct a phylogenetic tree
    and from the constructed tree, another class PhyloTree is called to extract tree statistics.

    :param input_fasta: Fasta DNA sequence file ---> will be used as initial sequences for the WF simulations
    :param n_repeats:  Number of observations to be made
    :param n_generations:
    :param n_individuals: population size
    :param mutation_rate: mutation rate of alleles to be mutated
    :param max_mutations: maximum allowed mutations during WF simulation
    :param out_dir: the path in which the results will be saved
    :return:
"""
    def __init__(self, input_fasta:str, n_repeats:int,
                 n_generations:int, n_individuals:int,
                 mutation_rate:float, max_mutations:int = None,
                 filter_allele_freq_below=0.0,
                 out_dir: str = None, save_data = True
                 ):
        self.out_dir = out_dir if out_dir is not None else os.path.join(DATA_PATH, "simulations", str(time.strftime("%Y%m%d-%H%M")))
        os.mkdir(self.out_dir) if not os.path.exists(self.out_dir) else None
        self.save_data = save_data

        self.in_fasta = input_fasta
        #self.tree_path = os.path.join(self.out_dir, "alleles.tree")
        self.out_fasta = None #### Will be set after running FWSim, required to construct the tree
        self.tree_path = None #### Will be set after running MAPLE

        self.n_repeats = n_repeats
        self.sim_number = None

        ### Fisher-Wright model parameters
        self.n_generations = n_generations
        self.n_individuals = n_individuals
        self.mutation_rate = mutation_rate
        self.max_mutations = max_mutations
        self.filter_allele_freq_below = filter_allele_freq_below

        ### Phylogenetic tree parameters
        self.tree_stats:List[Dict] = None
        #self.tree_stats_dict = None

        ### Output
        ##self.sumsts_matrix = None

    def run(self):
        self.tree_stats = []
        for i in range(self.n_repeats):
            out_sim_dir, sim_number = self._create_new_file(i)
            print(f"Running simulation {sim_number}...")
            self._run_FWSim(out_dir=out_sim_dir)
            try:
                self._run_MAPLE()
            except Exception as e:
                print(f"Maple failed for {sim_number}", e)
            self._run_PhyloTree(out_sim_dir)

    def _run_FWSim(self, out_dir=None):
        fwsim =FWSim(
            n_individuals=self.n_individuals, n_generations=self.n_generations,
            input_fasta=self.in_fasta, mutation_rates=self.mutation_rate, max_mutation_size=self.max_mutations,
            outdir=out_dir
        )
        fwsim.simulate_population()
        fwsim._filter_allele_freq(self.filter_allele_freq_below)
        fwsim.save_simulation()

        #### Sets out fasta that will be used by the MAPLE
        self.out_fasta = fwsim.out_fasta_path

    def _run_MAPLE(self):
        ### Run MAPLE through command line
        maple_script = os.path.join(PROJECT_PATH, "get_maple_tree.sh")
        working_dir = os.path.dirname(self.out_fasta)
        fasta_name = os.path.basename(self.out_fasta)
        call_subprocess("bash", [maple_script, working_dir, fasta_name])
        self.tree_path = os.path.join(working_dir, "_tree.tree")

    def _run_PhyloTree(self, outdir):
        ### Construct the tree from the MAPLE output

        phylotree = PhyloTree(tree_path=self.tree_path, tree_format=5)
        self.tree_stats.append(phylotree.get_tree_stats())
        phylotree.save_stats(os.path.join(outdir, 'tree_stats.json'))
        phylotree.save_tree(os.path.join(outdir, 'tree.png'))

    def _create_new_file(self, repeat_idx):
        sim_num = repeat_idx
        sim_dir = f"Sim_{repeat_idx}"
        outpath = os.path.join(self.out_dir, sim_dir)
        if os.path.exists(outpath):
            sim_num = max(list(map(lambda x: int(x.split("_")[-1]), os.listdir(os.path.join("/", *outpath.split("/")[:-1])))))
            sim_dir = f"Sim_{sim_num+1}"
            outpath = os.path.join(self.out_dir, sim_dir)
            self.sim_number = sim_num

        os.mkdir(outpath)
        return outpath, sim_num


def simulator(n_individuals, mutation_rate,  ### These are for Priors
              input_fasta, n_generations, max_mutations, work_dir=None,
              batch_size=1,
              random_state=None):
    """Wrapper for the simulator to make it compatible with the ELFI model

    This adds support for batch_size, transforms the data to a form that is efficient to store
    etc.

    Parameters
    ----------
    n_individuals
    mutation_rate
    input_fasta
    n_repeats
    n_generations
    max_mutations
    batch_size
    random_state
    dtype : np.dtype


    Returns
    -------
    np.ndarray

    """
    #y = np.zeros([batch_size, 4])
    y = np.zeros(batch_size, dtype=
    [
        ("max_H", np.float64),
        ("min_H", np.float64),
        ("a_BL_mean", np.float64),
        ("a_BL_median", np.float64),
    ])

    #for i in range(batch_size):
    s = Simulator(input_fasta=input_fasta,
                  n_generations=n_generations,
                  n_individuals=n_individuals, #if isinstance(n_individuals, int) else int(n_individuals[0]),
                  mutation_rate=mutation_rate, #if isinstance(mutation_rate, float) else float(mutation_rate[0]),
                  max_mutations=max_mutations,
                  n_repeats=batch_size,
                  out_dir=work_dir,
                  filter_allele_freq_below=0.001
                  )

    s.run()

    for repeat_idx, stats_dict in enumerate(s.tree_stats):
        #y[i,:] = np.array([values for keys, values in stats_dict.items()])
        for key, value in stats_dict.items():
            y[repeat_idx][key] = value

    return y

"""function_params = ["/Users/berk/Projects/jlees/data/WF_input.fasta", 3, 20, 500]
Ne = 32
mutation_rate = 0.05
print(simulator(
    Ne, mutation_rate,
    *function_params
))"""