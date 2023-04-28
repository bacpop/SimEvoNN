"""
This script contains simulator class for FW with mutations and phylogenetic tree models combined.
"""
import shutil
import tempfile

import numpy as np
from lib.phylogenetic_tree import PhyloTree
from lib.FW_model import FWSim

import os
import time
from config import DATA_PATH, PROJECT_PATH
from utils import call_subprocess

class Simulator(FWSim, PhyloTree):
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
                 filter_allele_freq_below=0.0, tree_path:str = None,out_fast:str = None,
                 out_dir: str = None, save_data = True,
                 ):
        self.out_dir = out_dir if out_dir is not None else os.path.join(DATA_PATH, "simulations", str(time.strftime("%Y%m%d-%H%M")))
        self.work_dir = tempfile.mkdtemp(prefix="Simulator")
        os.mkdir(self.out_dir) if not os.path.exists(self.out_dir) else None
        self.save_data = save_data

        self.in_fasta = input_fasta
        #self.tree_path = os.path.join(self.out_dir, "alleles.tree")
        self.out_fasta = out_fast if out_fast is not None else tempfile.mktemp(suffix=".fasta", dir=self.work_dir)#### Will be set after running FWSim, required to construct the tree
        self.tree_path = None

        self.n_repeats = n_repeats
        self.sim_number = 0

        ### Fisher-Wright model parameters
        self.filter_below = filter_allele_freq_below
        FWSim.__init__(self,
            n_individuals=n_individuals, n_generations=n_generations,
            input_fasta=input_fasta, mutation_rates=mutation_rate, max_mutation_size=max_mutations,
            outdir=out_dir
        )

        ### Phylogenetic tree parameters
        PhyloTree.__init__(self, tree_path=tree_path)

        ### Output
        #self.sumsts_matrix = np.zeros([n_repeats], dtype=[(keyname, data_type) for keyname in self.tree_stats_idx.keys()])
        self.sumsts_matrix = np.zeros([n_repeats, len(self.tree_stats_idx)])

    def run(self):
        for i in range(self.n_repeats):
            out_sim_dir, sim_number = self._create_new_dir(i)
            print(f"Running simulation {sim_number}...")
            self._run_FWSim()
            try: ##FIXME: sometimes Maple cannot construct tree from diverse seqeunces or seqeunce length does not add up
                self._run_MAPLE()
            except Exception as e:
                print(f"Maple failed for {sim_number}", e)
                shutil.rmtree(out_sim_dir)
                continue

            try:
                self._run_PhyloTree()
            except Exception as e: ##Sometimes when the tree is so small, it raises Exception
                print(f"PhyloTree failed for {sim_number}", e)
                shutil.rmtree(out_sim_dir)
                continue

            self.sumsts_matrix[i,:] = self.tree_stats

            if self.save_data:
                self.save_stats(os.path.join(out_sim_dir, 'tree_stats.json'))
                self.save_tree(os.path.join(out_sim_dir, 'tree.png'))
                self.save_simulation(os.path.join(out_sim_dir))
                self._mv_maple_outputs(out_sim_dir)

            else:
                shutil.rmtree(out_sim_dir)

            self.sim_number += 1

        np.save(os.path.join(self.out_dir, "simulation_results.npy"), self.allele_freq)

    def _run_FWSim(self):
        ### Run FWSim
        self.reset_frequencies() ## Re-initiate n_alleles, allele frequencies
        self.simulate_population()
        self._filter_allele_freq(self.filter_below)
        #### Sets out fasta that will be used by the MAPLE
        self.write_to_fasta(self.out_fasta)

    def _run_MAPLE(self):
        ### Run MAPLE through command line
        maple_script = os.path.join(PROJECT_PATH, "get_maple_tree.sh")
        fasta_name = os.path.basename(self.out_fasta)
        call_subprocess("bash", [maple_script, self.work_dir, fasta_name])
        self.tree_path = os.path.join(self.work_dir, "_tree.tree")

    def _run_PhyloTree(self):
        ### Construct the tree from the MAPLE output and get summary statistics
        self.reset_tree()
        self.get_summary_statistics()

    def _create_new_dir(self, repeat_idx):
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

    def _mv_maple_outputs(self,sim_out_dir):
        maple_files = os.listdir(self.work_dir)
        for filename in maple_files:
            shutil.move(os.path.join(self.work_dir, filename), sim_out_dir)


def simulator(n_individuals, mutation_rate,  ### These are for Priors
              input_fasta, n_generations, max_mutations, work_dir=None,
              save_data = False, filter_below = 0.0,
              batch_size=1, random_state=None
              ):
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

    s = Simulator(input_fasta=input_fasta,
                  n_generations=n_generations,
                  n_individuals=n_individuals,
                  mutation_rate=mutation_rate,
                  max_mutations=max_mutations,
                  n_repeats=batch_size,
                  out_dir=work_dir if work_dir is not None else os.path.join(DATA_PATH, "simulations", str(time.strftime("%Y%m%d-%H%M"))),
                  filter_allele_freq_below=filter_below,
                  save_data=save_data
                  )

    s.run()
    return s.sumsts_matrix

"""function_params = ["/Users/berk/Projects/jlees/data/WF_input.fasta", 50, 200]
Ne = 128
mutation_rate = 0.66

print(simulator(
    Ne, mutation_rate,
    *function_params,
    save_data=False,
    batch_size=100
))"""