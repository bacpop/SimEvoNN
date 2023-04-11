
import numpy as np
import matplotlib.pyplot as plt


def sim_haploid(n_repeats, n_generations, n_individuals, name):
    #Biallele (Haploid) model (Fisher-Wright)
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

    allele_freq = np.zeros([n_generations, n_repeats, n_alleles], dtype=np.float16)

    # Set the initial condition at the first step
    if set_allele_freq_equal:
        allele_freq[0, :, :] = 1 / n_alleles
    else:
        allele_freq[0, :, :] = np.random.uniform(0, 1, size=(n_repeats, n_alleles))
        allele_freq[0, :, :] /= np.sum(allele_freq[0, :, :], axis=1)[:, None]


    ### Set mutation marrix
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