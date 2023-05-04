## This script will include the classes for the alleles.
## Construct allele object for the sequences, include summary statistics calculation
import allel
import numpy as np
from lib.FW_model import FWSim
class Alleles(FWSim):

    allele_stats_indices = {
        'pi':0, ## Sequence diversity
        'theta_w':1,
        'tajimas_d':2,
        'f_st':3,
        'f_is':4,
        'entropy':5,
        'gc_content':6,
        'n_segregating_sites':7,
        'n_variants':8,
        'n_haplotypes':9,
        'h1':10,
        'h12':11,
        'h123':12,
        'h2_h1':13,
        'haplotype_diversity':14,
        'max_allele_freq':15,
        'min_allele_freq':16,
        'mean_allele_freq':17,
        'median_allele_freq':18,
    }
    def __init__(self,
                 *params, **kwargs):
        super().__init__(*params, **kwargs)

        self.positions = None
        self.sequences_mut_array = None
        self.variants_dict = None

        ### Haplotype/Allele statistics
        self.haplotype_array = None
        self.allele_counts_array = None
        self.n_variants = None
        self.n_haplotypes = None
        self.pi = None
        self.watterson_theta = None
        self.h1 = None
        self.h12 = None
        self.h123 = None
        self.h2_h1 = None
        self.ihs = None
        self.tajimas_d = None

        self.allele_stats = np.zeros(len(self.allele_stats_indices))

    def calculate_selection_coefficient(self):
        ##output depends on the "functional analysis"
        # https://www.nature.com/articles/s41559-017-0337-x
        ##for now, just return the selection coefficient
        ...
        #return self.selection_coefficient

    def reset_stats(self):
        self.allele_stats = np.zeros(len(self.allele_stats_indices))
        self.allele_counts_array, self.n_variants, self.n_haplotypes = None, None, None

    def _init_haplotype_array(self, maple_file=None):
        self.sequences_mut_array = np.zeros(
            [
                len(self.allele_indices),  ## Number of sequences, i.e. number of haplotypes
                len(self.allele_indices[0])  ## Reference sequence length, i.e. number of variants
            ], dtype='i1'
        )

        self.variants_dict = {pos: [ref.lower()] for pos, ref in enumerate(self.allele_indices[0])}
        self._construct_haplotypes_from_maple_variants_file(maple_file)
        self._generate_haplotype_array()
        self._get_counts()

    def get_allele_summary_stats(self):
        self._init_haplotype_array()
        self.calculate_allele_stats()
        return self.allele_stats

    def calculate_allele_stats(self):
        self.positions = self._get_positions()
        summaries = []


        ## Add pi as sequence diversity
        summaries.append(self.calculate_sequence_diversity())
        ## Add wattersons theta
        summaries.append(self.calculate_wattersons_theta())

        ## Add Tajima's D
        #summaries.append(self.calculate_tajimas_d())
        summaries.append(None)

        ## Calculate Fst
        summaries.append(None)
        ## Calculate Fis
        summaries.append(None)
        ## Calculate enthropy
        summaries.append(None)
        ## Calculate GC content
        summaries.append(None)
        ## Calculate number of segregating sites
        summaries.append(None)
        ## Add counts of alleles, variants and haplotypes
        summaries.extend([self.n_variants, self.n_haplotypes])
        ## Add haplotype statistics
        summaries.extend(self.calculate_garuds_h())
        ## Add haplotype diversity
        summaries.append(self.calculate_haplotype_diversity())
        ## Add max, min, mean, median allele frequencies
        summaries.extend([np.max(self.allele_freq), np.min(self.allele_freq), np.mean(self.allele_freq), np.median(self.allele_freq)])

        self.allele_stats[:] = np.array(summaries)
        return self.allele_stats

    def calculate_haplotype_diversity(self):
        return allel.haplotype_diversity(self.haplotype_array)

    def calculate_sequence_diversity(self):
        res = None
        try:
            res = allel.sequence_diversity(pos=self.positions, ac=self.allele_counts_array)
        except IndexError as e:
            print(e)
        finally:
            return res
        #return allel.mean_pairwise_difference(self.allele_counts)

    def calculate_wattersons_theta(self):
        return allel.watterson_theta(self.positions, self.allele_counts_array)

    def calculate_tajimas_d(self):
        self.tajimas_d = allel.tajima_d(self.allele_counts_array)
        return allel.tajima_d(self.allele_counts_array)

    def calculate_fst(self):
        ## Calculate Fst
        return None
    def calculate_ihs(self):
        ## Calculate integrated haplotype score
        #self.ihs = allel.ihs(self.haplotype_array, self._get_positions())
        return self.ihs

    def calculate_garuds_h(self):
        ## Calculate Garud's H
        self.h1, self.h12, self.h123, self.h2_h1 = allel.garud_h(self.haplotype_array)
        return self.h1, self.h12, self.h123, self.h2_h1
    def _generate_haplotype_array(self):
        self.haplotype_array = allel.HaplotypeArray(self.sequences_mut_array.T, dtype='i1')

    def _construct_haplotypes_from_maple_variants_file(self, maple_variants_file=None):
        ##Initialize the sequences_mut_array with the reference sequence

        self.sequences_mut_array[0, :] = 0
        import re
        ref_re = "^[acgt]{%d}" % len(self.allele_indices[0]) ## Regex to match the reference sequence
        seq_id_re = f"^>Sample[0-9]+"
        if maple_variants_file is not None:
            with open(maple_variants_file, 'r') as f:
                for line in f:
                    pline = line.strip()
                    if pline.startswith('>reference') or re.match(ref_re, pline):
                        continue
                    if re.match(seq_id_re, pline):
                        seq_id = int(pline.split('Sample')[-1])
                    elif pline[0] in {'a', 'c', 'g', 't'}:
                        nucleotide, position = pline.split('\t')
                        position = int(position) -1 ## 0-based indexing for mutation array
                        if nucleotide not in self.variants_dict[position]:
                            self.variants_dict.setdefault(position, []).append(nucleotide)
                        self.sequences_mut_array[seq_id, position] = self.variants_dict[position].index(nucleotide)

            return self.sequences_mut_array

    def _get_counts(self):
        if all([self.allele_counts_array, self.n_variants, self.n_haplotypes]):
            return self.allele_counts_array, self.n_variants, self.n_haplotypes
        self.allele_counts_array = self.haplotype_array.count_alleles()
        self.n_variants = self.haplotype_array.n_variants
        self.n_haplotypes = self.haplotype_array.n_haplotypes
        return self.allele_counts_array, self.n_variants, self.n_haplotypes

    def _get_positions(self):
        self.positions = []
        for key, value in self.variants_dict.items():
            if len(value) > 1:
                self.positions.append(key) ### allel uses 1-based indexing
        return self.positions



"""
allele_indices = {0: 'AAAATTTTGGGGCCCC', 1: 'AAAATTTTGGAGCCCC', 2: 'AAAATTTTGGGGCGCC', 3: 'AAAATTGTGGGGCCCC', 4: 'AAAATTTTGGTGCCCC', 5: 'AAAATTTTGGGGCCCG', 6: 'AAAATTATGGGGCCCC', 7: 'AAAATTTTAGGGCCCC', 8: 'AAAATTTTGGGGCCCA', 9: 'AATATTTTGGGGCGCC', 10: 'AAGATTTTGGTGCCCC', 11: 'AAAATTTTGGGGACCG', 12: 'AAAATATTGGGGCCCG', 13: 'ACAATTTTGGGGCCCG', 14: 'AAAATTTTGGTGCACC', 15: 'AAAATTTTGAGGACCG', 16: 'TAAATTTTGGGGACCG', 17: 'TAAATTTTGGGTACCG', 18: 'AAAGTTTTGGGGACCG', 19: 'AAAAGTTTGGGGACCG', 20: 'ATAATTTTGGGGACCG', 21: 'AAAATTTTAGGGACCG', 22: 'AAAATTGTCGGGCCCC', 23: 'AATATTGTGGGGCCCC', 24: 'AAAGTTTTGGGTACCG', 25: 'AAAATTTTGGGGTCCG', 26: 'AAAATTTTGGGGACTG', 27: 'AAAATTTTGGCGACCG'}
maple_file = '/Users/berk/Projects/jlees/data/TrueValues/temp_FWsim.txt'

al = Alleles(allele_indices, maple_file)


print(
    al.calculate_divergence(),
    al.calculate_haplotype_diversity(),
    al.calculate_garuds_h(),
)"""