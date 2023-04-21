"""
This script contains simulator class for FW with mutations and phylogenetic tree models combined.
"""

import numpy as np
import math
from phylogenetic_tree import PhyloTree
from FW_model import FWSim

import os
import time
from config import DATA_PATH, PROJECT_PATH
from utils import call_subprocess


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
                 out_dir: str = None,
                 ):
        self.out_dir = out_dir or os.path.join(DATA_PATH, "simulations", str(time.strftime("%Y%m%d-%H%M")))
        os.mkdir(self.out_dir) if not os.path.exists(self.out_dir) else None
        self.in_fasta = input_fasta
        #self.tree_path = os.path.join(self.out_dir, "alleles.tree")
        self.out_fasta = None #### Will be set after running FWSim, required to construct the tree
        self.tree_path = None #### Will be set after running MAPLE

        self.n_repeats = n_repeats
        ### Fisher-Wright model parameters
        self.n_generations = n_generations
        self.n_individuals = n_individuals
        self.mutation_rate = mutation_rate
        self.max_mutations = max_mutations

        ### Phylogenetic tree parameters
        #self.tree_stats = None
        self.tree_stats_dict = None

    def run(self):
        for i in range(self.n_repeats):
            print(f"Running simulation {i}...")
            out_sim_dir = os.path.join(self.out_dir, str(f"Sim_{i}"))
            os.mkdir(out_sim_dir) if not os.path.exists(out_sim_dir) else None
            self._run_FWSim(out_dir=out_sim_dir)
            self._run_MAPLE()
            self._run_PhyloTree(out_sim_dir)

    def _run_FWSim(self, out_dir=None):
        fwsim =FWSim(
            n_individuals=self.n_individuals, n_generations=self.n_generations,
            input_fasta=self.in_fasta, mutation_rates=self.mutation_rate, max_mutation_size=self.max_mutations,
            outdir=out_dir
        )
        fwsim.simulate_population()
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
        self.tree_stats_dict = phylotree.get_tree_stats()
        phylotree.save_stats(os.path.join(outdir, 'tree_stats.json'))
        phylotree.save_tree(os.path.join(outdir, 'tree.png'))

    def _construct_summary_statistics_matrix(self):
        ### Construct the summary statistics matrix from the tree stats
        ...


Simulator(
    input_fasta="/Users/berk/Projects/jlees/data/WF_input.fasta",
    n_repeats=3,
    n_generations=20,
    n_individuals=64,
    mutation_rate=0.1,
    max_mutations=500,
).run()
