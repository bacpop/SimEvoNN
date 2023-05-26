"""
This script contains simulator class for FW with mutations and phylogenetic tree models combined.
"""
import shutil
import tempfile

import numpy as np
from lib.phylogenetic_tree import PhyloTree
from lib.alleles import Alleles

import os
import time
from config import DATA_PATH
from utils import create_maple_tree


class Simulator(Alleles, PhyloTree):
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

    def __init__(self, input_fasta: str, n_repeats: int, n_batches: int = 1,
                 n_generations: int = None, n_individuals: int = None,
                 mutation_rate: float = None, max_mutations: int = None,
                 filter_allele_freq_below=None, tree_path: str = None,
                 outdir: str = None, workdir: str = None, save_data=True,
                 save_parameters_on_output_matrix=False,
                 sim_name: str = None, prior_parameters: dict = None
                 ):
        self.sim_name = sim_name if sim_name is not None else "sim"
        self.out_dir = outdir if outdir is not None else os.path.join(DATA_PATH, "simulations",
                                                                      str(time.strftime("%Y%m%d-%H%M")))
        os.mkdir(self.out_dir) if not os.path.exists(self.out_dir) else None
        self.work_dir = workdir if workdir is not None else tempfile.mkdtemp(prefix=f"Simulator_{self.sim_name}")
        os.mkdir(self.work_dir) if not os.path.exists(self.work_dir) else None
        self.save_data = save_data
        self.save_parameters_on_output_matrix = save_parameters_on_output_matrix

        self.in_fasta = input_fasta
        # self.tree_path = os.path.join(self.out_dir, "alleles.tree")
        self.maple_sequences_path = tempfile.mktemp(prefix=self.sim_name, suffix=".txt",
                                                                               dir=self.work_dir)  #### Will be set after running FWSim, required to construct the tree
        self.tree_path = None

        self.n_repeats = n_repeats
        self.n_batches = n_batches
        self.sim_number = 0

        ### Fisher-Wright model and ALleles parameters
        self.prior_parameters = prior_parameters
        self.n_generations = n_generations
        self.max_mutation_size = max_mutations
        self.mutation_rate = mutation_rate
        self.n_individuals = n_individuals
        self.filter_below = filter_allele_freq_below
        self.initial_sequence = self._read_fasta(self.in_fasta)
        self.initial_allele_seq = self.initial_sequence
        Alleles.__init__(self,
                         n_individuals=n_individuals, n_generations=self.n_generations,
                         initial_allele_seq=self.initial_sequence, mutation_rates=mutation_rate,
                         max_mutation_size=self.max_mutation_size, outdir=self.work_dir)

        ### Phylogenetic tree parameters
        PhyloTree.__init__(self, tree_path=tree_path)

        ### Output
        # self.sumsts_matrix = np.zeros([n_repeats], dtype=[(keyname, data_type) for keyname in self.tree_stats_idx.keys()])
        self.sumsts_matrix = np.zeros(
            [n_repeats * n_batches, len(self.tree_stats_idx) + len(self.allele_stats_indices)])
        self.resulting_matrix = None

    def run(self):
        for i in range(1, self.n_repeats + 1):
            ne, mu = np.random.randint(1, 1000), np.random.uniform(0, 1)
            for b in range(1, self.n_batches+1):
                self.sim_number += 1
                self._create_new_dir(self.sim_number)
                print(f"Running simulation {self.sim_number}...")

                #try:
                self._run_FWSim(ne=ne, mu=mu)
                #except Exception as e:
                #    print(f"Wright-Fisher simulator failed for simulation number {self.sim_number}", e)
                    #shutil.rmtree(out_sim_dir)
                #    continue

                ##Check point for the number of alleles
                if self.n_alleles < 3: ## Some calculations raise error after this point
                    print(f"Simulation number {self.sim_number} has less than 3 alleles")
                    # shutil.rmtree(out_sim_dir)
                    continue

                ##Calculate allele statistics and allele freqs of FWsim
                try:
                    self._run_Alleles()
                except Exception as e:
                    print(f"Run Alleles failed for {self.sim_number}", e)
                    # shutil.rmtree(out_sim_dir)
                    #continue
                    raise e

                try:  ##FIXME: sometimes Maple cannot construct tree from diverse seqeunces or seqeunce length does not add up
                    self._run_MAPLE()
                except Exception as e:
                    print(f"Maple failed for simulation number {self.sim_number}", e)
                    #shutil.rmtree(out_sim_dir)
                    #continue

                try:
                    self._run_PhyloTree()
                except Exception as e:  ##Sometimes when the tree is so small, it raises Exception
                    print(f"PhyloTree failed for {self.sim_number}", e)
                    #shutil.rmtree(out_sim_dir)

                self.sumsts_matrix[(i - 1) * self.n_batches:i * self.n_batches,:len(self.tree_stats_idx)] = self.tree_stats
                self.sumsts_matrix[(i - 1) * self.n_batches:i * self.n_batches, len(self.tree_stats_idx):] = self.allele_stats

                if self.save_data:
                    out_dir = os.path.join(self.out_dir, f"Sim_{self.sim_number}")
                    os.mkdir(out_dir) if not os.path.exists(out_dir) else None
                    self.save_stats(os.path.join(out_dir, 'tree_stats.json'))
                    self.save_tree(os.path.join(out_dir, 'tree.png'))
                    self.save_simulation(out_dir)
                    self._mv_maple_outputs(out_dir)

            if self.save_parameters_on_output_matrix:
                self.resulting_matrix = np.c_[
                    self.sumsts_matrix[(i - 1) * self.n_batches:i * self.n_batches], np.array([ne, mu, self.n_generations, self.max_mutation_size]) * np.ones(
                        [self.n_batches, 4])
                ]
            else:
                self.resulting_matrix = self.sumsts_matrix

        np.save(os.path.join(self.out_dir, f"{self.sim_name}_results.npy"), self.resulting_matrix)
        shutil.rmtree(self.work_dir)

    def _run_FWSim(self, ne, mu):
        ### Run Wright-Fisher simulation

        ## Re-initiate n_alleles, allele frequencies, mutation rates and n_individuals
        self.reinitialize_params_and_frequencies(mutation_rates=mu, n_individuals=ne)
        self.simulate_population()
        if self.filter_below is not None:
            self._filter_allele_freq(self.filter_below)
        #### writes a maple file that will be used by the MAPLE to construct tree
        self.write_MAPLE_file(self.maple_sequences_path)

    def _run_MAPLE(self):
        ### Run MAPLE
        self.tree_path = create_maple_tree(self.maple_sequences_path)

    def _run_PhyloTree(self):
        ### Construct the tree from the MAPLE output and get summary statistics
        self.reset_tree()
        self.get_summary_statistics()

    def _run_Alleles(self):
        self._init_haplotype_array()
        self.calculate_allele_stats()

    def _create_new_dir(self, repeat_idx):
        ### ELFI compatible directory creation
        ## ELFI does not run inside the class, but gives each value from outside
        ## In order to follow simulation number, this function follows the simulation number from directory name and number
        sim_dir = f"Sim_{repeat_idx}"
        outpath = os.path.abspath(os.path.join(self.work_dir, sim_dir))
        if os.path.exists(outpath):
            sim_num = max(
                list(map(lambda x: int(x.split("_")[-1]),
                         [folder for folder in os.listdir(os.path.join("/", *outpath.split("/")[:-1])) if
                          "Sim_" in folder]))
            )
            sim_dir = f"Sim_{sim_num + 1}"
            outpath = os.path.join(self.work_dir, sim_dir)
            self.sim_number = sim_num
            return outpath
        os.mkdir(outpath)
        return outpath

    def _mv_maple_outputs(self, target_dir):
        maple_files = os.listdir(self.work_dir)
        for filename in maple_files:
            shutil.move(os.path.join(self.work_dir, filename), target_dir)

    @staticmethod
    def _read_fasta(file_name):
        rv_list = []
        seq = ''
        old_seq = False
        with open(file_name, 'r') as f:
            for line in f:
                if line.startswith('>') and old_seq:
                    rv_list.append(seq)
                    seq = ''
                    old_seq = False

                elif line[0].lower() in {"a", "c", "g", "t", "n"}:
                    seq += line.strip()
                    old_seq = True

            rv_list.append(seq)

        return rv_list

    def _get_prior_distributions(self):
        ### Not used atm, but can be used to set random priors from cli
        if self.prior_parameters is None:
            params_dict = None
            pass  #### Automatically set the priors
        elif isinstance(self.prior_parameters, dict):
            pass  #### Set the priors from the dictionary
        elif isinstance(self.prior_parameters, str):
            pass  #### Read the priors from the file
        else:
            raise ValueError("prior_parameters must be either a dictionary, a string or None")


def simulator(n_individuals, mutation_rate,  ### These are for Priors
              input_fasta, n_generations, max_mutations, work_dir=None,
              save_data=False, filter_below=0.0,n_repeats=1,
              batch_size=1, random_state=None, add_parameters=False, outdir=None, dtype=np.float32
              ):
    """Wrapper for the simulator to make it compatible with the ELFI model

    This adds support for batch_size, random_state and dtype

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
                  n_repeats=n_repeats,
                  n_batches=batch_size,
                  workdir=work_dir,
                  outdir=outdir if outdir is not None else os.path.join(DATA_PATH, "simulations",
                                                                             str(time.strftime("%Y%m%d-%H%M"))),
                  filter_allele_freq_below=filter_below,
                  save_data=save_data,
                  save_parameters_on_output_matrix=add_parameters,
                  )

    s.run()
    return s.resulting_matrix
