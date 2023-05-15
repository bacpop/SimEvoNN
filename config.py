import os
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

FASTA_IN = os.path.join(DATA_PATH, "WF_input.fasta")

### Required for keeping track of columns in the simulation results
TREE_SS_INDICES = {
    "max_H": 0,
    "min_H": 1,
    "a_BL_mean": 2,
    "a_BL_median": 3,
    "a_BL_var": 4,
    "e_BL_mean": 5,
    "e_BL_median": 6,
    "e_BL_var": 7,
    "i_BL_mean_1": 8,
    "i_BL_median_1": 9,
    "i_BL_var_1": 10,
    "ie_BL_mean_1": 11,
    "ie_BL_median_1": 12,
    "ie_BL_var_1": 13,
    "i_BL_mean_2": 14,
    "i_BL_median_2": 15,
    "i_BL_var_2": 16,
    "ie_BL_mean_2": 17,
    "ie_BL_median_2": 18,
    "ie_BL_var_2": 19,
    "i_BL_mean_3": 20,
    "i_BL_median_3": 21,
    "i_BL_var_3": 22,
    "ie_BL_mean_3": 23,
    "ie_BL_median_3": 24,
    "ie_BL_var_3": 25,
    "colless": 26,
    "sackin": 27,
    "WD_ratio": 28,
    "delta_w": 29,
    "max_ladder": 30,
    "IL_nodes": 31,
    "staircaseness_1": 32,
    "staircaseness_2": 33,
    "tree_size": 34,
}
ALLELE_SS_INDICES = {
    'pi':0, ## Sequence diversity
    'theta_w':1,
    'tajimas_d':2,
    'f_st':3,
    'f_is':4,
    'entropy':5,
    'delta_gc_content':6,
    'n_segregating_sites':7,
    'n_variants':8,
    'n_haplotypes':9,
    'h1':10,
    'h12':11,
    'h123':12,
    'h2_h1':13,
    'haplotype_diversity':14,
    'allele_freq_max':15,
    'allele_freq_min':16,
    'allele_freq_mean':17,
    'allele_freq_median':18,
    'allele_freq_var':19,
    'ihs':20,
}
PARAMETER_INDICES = {
    "n_individuals" : 0,#56 ,
    "mutation_rate" : 1,#57,
    "n_generations" : 2,#58,
    "max_mutations" : 3,#59
}
SS_INDICES = TREE_SS_INDICES.copy()
SS_INDICES.update({key: len(TREE_SS_INDICES) + idx for key, idx in ALLELE_SS_INDICES.items()})
ALL_INDICES = SS_INDICES.copy()
ALL_INDICES.update({key : len(SS_INDICES) + idx for key, idx in PARAMETER_INDICES.items()})