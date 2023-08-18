
import numpy as np


def sim_haploid(n_repeats, n_generations, n_individuals, name):
    #Biallele (Haploid) model (Fisher-Wright)
    import matplotlib.pyplot as plt
    b_freq = np.zeros([n_generations, n_repeats])

    # Set the initial condition at the first step
    b_freq[0, :] = 0.5

    for repeat_idx in range(1, n_repeats):
        for generation in range(1, n_generations):
            number_mutants = np.random.binomial(n=n_individuals, p=b_freq[generation - 1, repeat_idx])
            b_freq[generation, repeat_idx] = number_mutants / n_individuals

    plt.plot(range(1, n_generations + 1), b_freq)
    plt.xlabel("Generation")
    plt.ylabel("B allele frequency")
    plt.title(name)
    plt.show()


def plot_multiallele(allele_freq, alleles=None):
    ##Plot the allele frequencies
    #rows=cols=n_alleles//2 + n_alleles%2
    alleles_length = len(allele_freq[0,0,:].nonzero()[0])
    columns=4
    rows=alleles_length//4 +1 if alleles_length%4 else alleles_length//4

    plt.figure(figsize=(18, 21))
    plt.subplots_adjust(hspace=0.5)

    for allele_idx in range(alleles_length):
        ax = plt.subplot(rows,columns , allele_idx+1, )
        ax.plot(range(1, len(allele_freq[:,0,0])+1), allele_freq[:, :, allele_idx])
        ax.set_xlabel("Generation")
        ax.set_ylabel(f"{alleles[allele_idx] if alleles else allele_idx} Allele frequency")
        #ax.set_title(allele)
    plt.show()

## ref: https://doi.org/10.1093/sysbio/syw056


def fisher_wright_simulator(
        n_repeats, n_generations, n_individuals,
        n_alleles=None,alleles=None, mutation_rates:np.array=None,
        plot=False,set_allele_freq_equal=True,
        batch_size=None, random_state=None ### Parameters for ELFI
):
    """
    Simulates the Fisher-Wright model with Jukes-Cantor mutation model
    :return: allele_freq: n_generations x n_repeats x n_alleles
    """
    ##(Fisher-Wright) model on haploid population
    assert n_alleles or alleles, "Either n_alleles or alleles must be provided"
    n_alleles = len(alleles) if alleles else n_alleles
    if n_alleles > 2: dist_func = np.random.multinomial
    else: dist_func = np.random.binomial

    allele_freq = np.zeros([n_generations, n_repeats, n_alleles], dtype=np.float64)

    # Set the initial condition at the first step
    if set_allele_freq_equal:
        allele_freq[0, :, :] = 1 / n_alleles
    else:
        allele_freq[0, :, :] = np.random.uniform(0, 1, size=(n_repeats, n_alleles))
        allele_freq[0, :, :] /= np.sum(allele_freq[0, :, :], axis=1)[:, None]


    ### Set mutation matrix
    if mutation_rates is None:
        mutation_rates = np.zeros(n_repeats)
    else:
        mutation_rates = np.array(mutation_rates)
    if len(mutation_rates) < n_repeats:
        mutation_rates = np.repeat(mutation_rates, n_repeats)

    mutation_matrix = jc_mutation_calc(n_alleles, mutation_rates=mutation_rates, n_repeats=n_repeats)

    for repeat_idx in range(n_repeats):
        for generation in range(1, n_generations):
            p = np.matmul(mutation_matrix[repeat_idx,:,:], allele_freq[generation - 1, repeat_idx, :].transpose())
            n_chosen_alleles = dist_func(n_individuals,p)
            allele_freq[generation, repeat_idx,:] = n_chosen_alleles / n_individuals

    ##Plot the allele frequencies
    if plot:
        plot_multiallele(allele_freq, alleles)

    return allele_freq


#sim_haploid(10, 100, 100, "Haploid")

## requires a mutation probability n_alleleXn_allele matrix (U)
## For mutation model, Jukes-Cantor model can be used.
## take into account the mutation rate (mu) and the mutation probability (U)

def jc_mutation_calc(n_alleles, mutation_rates:np.array=None, n_repeats=1):
    ##Jukes-Canto --> simple equal probability of mutation
    # mutation probability matrix
    U = np.zeros([n_repeats, n_alleles, n_alleles], dtype=np.float16)
    for repeat_idx in range(n_repeats):
        U[repeat_idx,:,:].fill(mutation_rates[repeat_idx] / (n_alleles - 1))
        np.fill_diagonal(U[repeat_idx,:,:], 1 - mutation_rates[repeat_idx])
    return U

##parameters : alleles:list, n_repeats, n_generations, n_individuals
#fisher_wright_simulator(["A", "B", "C", "D", "E", "F"], 5, 1000, 20, mutation_rate=0.0015)
"""fisher_wright_simulator(
    #alleles=["A", "B"],
    n_alleles=4,
    n_repeats=3,
    n_generations=100,
    n_individuals=50,
    mutation_rates=[0.05, 0.1, 0.5],
    plot=True,
    set_allele_freq_equal=True
)"""
## Note, adding mutation rate prevents allele extinction or fixation


#mutation_rates = {"A": [0.1, 0.2, 0.3, 0.4], "C": [0.1, 0.2, 0.3, 0.4], "G": [0.1, 0.2, 0.3, 0.4], "T": [0.1, 0.2, 0.3, 0.4]}
#print(mutation_simulator(construct_mutation_matrix(mutation_rates=0.01), ancestral_allele))

import os
import time
from config import DATA_PATH

class FWSim:
    """
    Fisher-Wright Simulator
    This class takes n_individuals, n_generations, initial_allele_seq or input_fasta, mutation_rates, max_mutation_size
    And
    produces a simulation result --> allele_freq: n_generations x (n_alleles + max_mutation_size)
    """

    def __init__(
            self, n_individuals, n_generations, initial_allele_seq:list=None,
            mutation_rates:dict or float=None, max_mutation_size:int=None,
            input_fasta:str=None, outdir:str=None

    ):
        self.input_fasta = input_fasta
        if input_fasta is not None and initial_allele_seq is None:
            initial_allele_seq = self.read_fasta(input_fasta)

        self.outdir = outdir if outdir is not None else os.path.join(DATA_PATH, f"FWSim_{time.strftime('%Y%m%d_%H%M')}")

        self.initial_allele_seq = initial_allele_seq
        self.n_individuals = n_individuals
        self.mutation_rates = mutation_rates
        self.n_generations = n_generations
        self.max_mutation_size = max_mutation_size if max_mutation_size is not None else 4**len(initial_allele_seq[0])

        self.n_alleles = len(initial_allele_seq) if isinstance(initial_allele_seq, list) else 1
        self.len_dna_sequence = len(initial_allele_seq[0]) if isinstance(initial_allele_seq, list) else len(initial_allele_seq)
        self.allele_freq = None
        self.allele_mutation_indices = None ## Holds seq_id: [(bp_location, nucleotide), ...]
        self.allele_mutation_indices_set = None
        self.mutation_matrix = None
        self.allele_indices = {idx: seq for idx, seq in enumerate(initial_allele_seq)}
        self.initialize_allele_freq_matrix()
        self.initialize_mutation_matrix()

    def reinitialize_params_and_frequencies(self, n_individuals, mutation_rates):
        self.n_individuals = n_individuals
        self.mutation_rates = mutation_rates
        self.reset_frequencies()
        self.initialize_mutation_matrix()

    def reset_frequencies(self):
        self.n_alleles = len(self.initial_allele_seq) if isinstance(self.initial_allele_seq, list) else 1
        self.allele_indices = {idx: seq for idx, seq in enumerate(self.initial_allele_seq)}
        self.initialize_allele_freq_matrix()

    def initialize_allele_freq_matrix(self):
        self.allele_freq = np.zeros([self.n_generations, self.n_alleles+self.max_mutation_size], dtype=np.float64)
        self.allele_freq[0, :self.n_alleles] = 1 / self.n_alleles
        self.allele_mutation_indices = {idx: [] for idx in range(self.n_alleles)}
        self.allele_mutation_indices_set = set()

    def initialize_mutation_matrix(self):
        self.mutation_matrix = self.construct_mutation_matrix(self.mutation_rates)

    def get_allele_summary_stats(self):
        pass

    @staticmethod
    def construct_mutation_matrix(mutation_rates=None):
        mutation_matrix = np.zeros([4, 4], dtype=np.float16)
        if mutation_rates is None:
            np.fill_diagonal(mutation_matrix, 1)
        elif isinstance(mutation_rates, float):  ## If one value is given, use it for all nucleotides
            mutation_matrix.fill(mutation_rates/3)
            np.fill_diagonal(mutation_matrix, 1 - mutation_rates)
        elif isinstance(mutation_rates, dict):
            for key, value in mutation_rates.items():
                if key == "A":
                    mutation_matrix[0, :] = value
                elif key == "C":
                    mutation_matrix[1, :] = value
                elif key == "G":
                    mutation_matrix[2, :] = value
                elif key == "T":
                    mutation_matrix[3, :] = value
        else:
            raise Exception("Unknown mutation rate. Could not calculate mutation matrix")
        return mutation_matrix

    def simulate_mutation_binary(self,sequence:str):
        """
        Selects a random mutation site and changes "1" to "0" or "0" to "1"
        :param sequence: str
        :return: str sequence
        """
        sequence = list(sequence)
        mutation_site = np.random.randint(0, len(sequence))
        sequence[mutation_site] = "0" if sequence[mutation_site] == "1" else "1"
        return "".join(sequence)

    def simulate_mutation(self, dna_sequence):
        """
        :param mutation_matrix: 0 - A, 1 - C, 2 - G, 3 - T
        :param dna_sequence: sequence of nucleotides
        :return:
        """

        b_loc = np.random.randint(0, self.len_dna_sequence)
        nucleotide = dna_sequence[b_loc]
        nucleotides = ["A", "C", "G", "T"]

        if nucleotide == "A":
            mutated_to = np.random.choice(nucleotides, p=self.mutation_matrix[0, :])
        elif nucleotide == "C":
            mutated_to = np.random.choice(nucleotides, p=self.mutation_matrix[1, :])
        elif nucleotide == "G":
            mutated_to = np.random.choice(nucleotides, p=self.mutation_matrix[2, :])
        elif nucleotide == "T":
            mutated_to = np.random.choice(nucleotides, p=self.mutation_matrix[3, :])
        else:
            print("Warning: Invalid nucleotide")
            mutated_to = nucleotide
            return dna_sequence, b_loc, mutated_to, nucleotide

        ### Update nucleotide mutation indices
        return f"{dna_sequence[:b_loc]}{mutated_to}{dna_sequence[b_loc + 1:]}", b_loc, mutated_to, nucleotide

    @staticmethod
    def _choose_mutated_alleles(alleles):
        alleles_len = len(alleles)
        p_mutate = np.repeat(np.array(1/alleles_len), alleles_len)
        #if alleles_len > 2:
        #    return np.random.multinomial(alleles, p_mutate)
        #else:
        return np.random.binomial(alleles, p_mutate)

    def _choose_next_alleles(self, allele_freq):
        return np.random.multinomial(self.n_individuals, allele_freq)

    def simulate_population(self):

        for generation in range(1, self.n_generations):

            #### Obtain how many alleles are selected for the next generation
            selected_allele_counts = self._choose_next_alleles(self.allele_freq[generation - 1,:])

            if self.n_alleles < self.max_mutation_size:

                ### How many of the alleles are mutated
                number_of_mutations = self._choose_mutated_alleles(selected_allele_counts[:self.n_alleles])

                ### How mutations occur
                counts = self.mutation_event(number_of_mutations)

                selected_allele_counts[:self.n_alleles] = selected_allele_counts[:self.n_alleles] - number_of_mutations
                for allele_idx, count in counts.items():
                    selected_allele_counts[allele_idx] += count

            ### Update allele frequency matrix
            self.allele_freq[generation, :] = selected_allele_counts / self.n_individuals

            self.n_alleles = len(self.allele_indices)

        return self.allele_freq

    def simulate_individual(self): ##TODO: Non-functional
        ## initialize alleles matrix
        population_matrix = np.zeros([self.n_generations, self.n_individuals], dtype=np.int16)
        for generation in range(1, self.n_generations):
            for individual in range(self.n_individuals):

                ## Choose alleles for the next generation
                population_matrix[generation, individual] = self._choose_next_alleles(self.allele_freq[generation - 1,:])
                ## Mutate alleles/individuals
                ###TODO: Implement mutation (Figure out how to do it)

            self.allele_freq[generation, :] = np.sum(population_matrix[generation:]) / self.n_individuals

    def mutation_event(self, number_of_mutations):
        ### Get new mutations, update allele indices and outputs mutation counts
        ###TODO: Think of optimising this part (using numpy arrays)
        #no_mutation_counts = {}
        #mutation_counts = {}
        counts = {}
        for allele_idx, allele_count in enumerate(number_of_mutations):
            for c in range(allele_count):
                ##Simulate mutation to get de-novo sequence
                mutated_seq, bp_loc, mutated_nuc, init_nuc = self.simulate_mutation(self.allele_indices[allele_idx])
                seq_representation = (allele_idx, init_nuc, mutated_nuc, bp_loc)
                len_indices = len(self.allele_indices)
                ##Check if the sequence is already in the dictionary
                exceeds_mutation_size = len_indices >= self.max_mutation_size
                new_mutation = all([
                    seq_representation not in self.allele_mutation_indices_set,
                    init_nuc != mutated_nuc,
                    not exceeds_mutation_size
                ])

                existing_mutation = all([
                    seq_representation in self.allele_mutation_indices_set,
                    init_nuc != mutated_nuc,
                ])

                no_mutation = all([
                    init_nuc == mutated_nuc,
                ])

                if new_mutation:
                    self.allele_mutation_indices_set.add(seq_representation)
                    ### Keep track of the mutation indices and allele indices
                    if allele_idx in self.allele_mutation_indices.keys():
                        ## Keep in minde that the mutation locations are relative to the original sequence
                        self.allele_mutation_indices[len_indices] = self.allele_mutation_indices[allele_idx].copy()
                        self.allele_mutation_indices[len_indices].append(seq_representation)
                    else:
                        self.allele_mutation_indices[len_indices] = [seq_representation]

                    self.allele_indices[len_indices] = mutated_seq
                    #mutation_counts[seq_representation] = 1
                    counts[len_indices] = 1

                elif existing_mutation:
                    ## find the index of the mutated sequence
                    existing_idx = self._get_index(mutated_seq)
                    counts.setdefault(existing_idx, 0)
                    counts[existing_idx] += 1

                elif no_mutation:
                    counts.setdefault(allele_idx, 0)
                    counts[allele_idx] += 1

                elif exceeds_mutation_size: continue  ## discard new mutations

                else:
                    raise ValueError("Invalid sequence")

        return counts

    def _get_initial_seq_index(self, seq_idx):
        ## Traces back to origin of a given idx
        while seq_idx > len(self.initial_allele_seq)-1:
            seq_idx = self.allele_mutation_indices[seq_idx][0][0]
        return seq_idx



    def _get_index(self, sequence):
        for key, value in self.allele_indices.items():
            if value == sequence:
                return key

    def _filter_allele_freq(self, filter_below=0.01):
        filtering_array = self.allele_freq[self.n_generations-1,:] > filter_below
        allele_freq = self.allele_freq[:,filtering_array]
        self._update_allele_indices(filtering_array)
        self._update_allele_freq(allele_freq)
        return allele_freq

    def _update_allele_indices(self, array:np.array):
        updated_allele_indices = {}
        updated_mutations_indices = {}
        return_idx_counter = 0
        for idx, boolean in enumerate(array):
            if len(self.allele_indices) <= idx: ###Number of new sequences cannot be more than the indexed sequences
                break

            if idx < len(self.initial_allele_seq) or boolean:  ### Do not change indices for initial sequences
                updated_allele_indices[return_idx_counter] = self.allele_indices[idx]
                updated_mutations_indices[return_idx_counter] = self.allele_mutation_indices[idx]
                return_idx_counter += 1

        self.allele_indices = updated_allele_indices
        self.allele_mutation_indices = updated_mutations_indices
        self.n_alleles = len(self.allele_indices)

    def _update_allele_freq(self, allele_freq):
        self.allele_freq = allele_freq
        return self.allele_freq

    def save_simulation(self, out_fasta_path=None, out_freq_path=None, out_parameters_path=None, outdir=None):
        outdir = outdir if outdir is not None else self.outdir
        out_fasta_path = os.path.join(outdir, "fw_sequences.fasta") if not out_fasta_path else out_fasta_path
        out_freq_path = os.path.join(outdir, "fw_freq.npy") if not out_freq_path else out_freq_path
        out_parameters_path = os.path.join(outdir, "fw_parameters.json") if not out_parameters_path else out_parameters_path
        self.save_parameters(out_parameters_path)
        self.save_allele_freq(out_freq_path)
        self.write_to_fasta(out_fasta_path)
        self.plot_allele_freq(save_fig=True, file_name=os.path.join(outdir,'allele_freq.png'), dont_plot=True)
        self.plot_allele_freq_line(save_fig=True, file_name=os.path.join(outdir,'allele_freq_line.png'), dont_plot=True)
        #os.replace('allele_freq.png', os.path.join(self.outdir, 'allele_freq.png'))

    def plot_allele_freq(self, filter_below=None, save_fig=True, file_name='allele_freq.png', dont_plot=False):
        import matplotlib.pyplot as plt
        if filter_below:
            self._filter_allele_freq(filter_below=filter_below)

        plt.figure(figsize=(20, 10))
        allele_to_plot = self.allele_freq[:, :len(self.allele_indices)]
        plt.imshow(allele_to_plot, cmap='viridis')
        plt.xticks(np.arange(len(self.allele_indices)), [f"Sample_{i}" for i in self.allele_indices.keys()], rotation=90)
        plt.xlabel('Alleles')
        plt.ylabel('Generations')
        plt.colorbar()
        if save_fig:
            plt.savefig(file_name)
        if dont_plot:
            plt.close()
            return None
        plt.show()

    def plot_allele_freq_line(self, filter_below=None, save_fig=True, file_name='allele_freq_line.png', dont_plot=False):
        import matplotlib.pyplot as plt
        if filter_below:
            self._filter_allele_freq(filter_below=filter_below)

        plt.figure(figsize=(20, 10))
        allele_to_plot = self.allele_freq[:, :len(self.allele_indices)]
        for i in range(allele_to_plot.shape[1]):
            plt.plot(allele_to_plot[:,i], label=f"Sample_{i}", alpha=0.9, linewidth=3)
        plt.xlabel('Generations')
        plt.ylabel('Allele Frequency')
        plt.legend()
        if save_fig:
            plt.savefig(file_name)
        if dont_plot:
            plt.close()
            return None
        plt.show()

    def write_to_fasta(self, file_name):
        with open(file_name, 'w') as f:
            for allele, sequence in self.allele_indices.items():
                f.write(f'>Sample{allele}\n{sequence}\n')

    def save_allele_freq(self, file_name):
        np.save(file_name, self.allele_freq)

    def save_parameters(self, file_name):
        """Save parameters to a json file"""
        import json
        with open(file_name, 'w') as f:
            json.dump({
                "initial_allele_seq": self.initial_allele_seq,
                "n_individuals": self.n_individuals,
                "n_generations": self.n_generations,
                "mutation_rates": self.mutation_rates,
                "max_mutation_size": self.max_mutation_size
            }, f)

    def read_fasta(self, file_name):
        rv_list = []
        with open(file_name, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    continue
                else:
                    rv_list.append(line.strip())

        return rv_list

    def write_MAPLE_file(self, maple_file_path):
        with open(maple_file_path, "w") as fh:
            for idx, seq_represents in self.allele_mutation_indices.items():
                if idx == 0 and seq_represents == []:  ## no mutation representation for reference
                    fh.write(f">reference\n{self.initial_allele_seq[0].lower()}\n>Sample{str(0)}\n")
                    continue
                fh.write(f">Sample{idx}\n")
                ### Maple requires sorted positions
                positions_lines = {}
                for seq_represent in seq_represents:
                    from_idx, ref, alt, bp_pos = seq_represent
                    positions_lines[bp_pos] = f"{alt.lower()}\t{str(bp_pos+1)}\n"

                for bp_pos in sorted(positions_lines.keys()):
                    fh.write(positions_lines[bp_pos])