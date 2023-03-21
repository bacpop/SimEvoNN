
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

def plot_multiallele(allele_freq, n_generations, alleles):
    ##Plot the allele frequencies
    #rows=cols=n_alleles//2 + n_alleles%2
    columns=4
    rows=len(alleles)//4 +1 if len(alleles)%4 else len(alleles)//4

    plt.figure(figsize=(18, 21))
    plt.subplots_adjust(hspace=0.5)

    for allele_idx, allele in enumerate(alleles):
        ax = plt.subplot(rows,columns , allele_idx+1, )
        ax.plot(range(1, n_generations + 1), allele_freq[:, :, allele_idx])
        ax.set_xlabel("Generation")
        ax.set_ylabel(f"{allele} Allele frequency")
        ax.set_title(allele)
    plt.show()

## ref: https://doi.org/10.1093/sysbio/syw056


def fisher_wright_simulator(alleles:list, n_repeats, n_generations, n_individuals, mutation_rate=0.0):
    ##(Fisher-Wright) model
    n_alleles = len(alleles)
    if n_alleles > 2: dist_func = np.random.multinomial
    else: dist_func = np.random.binomial

    allele_freq = np.zeros([n_generations, n_repeats, n_alleles])
    # Set the initial condition at the first step
    allele_freq[0, :, :] = 1 / n_alleles

    mutation_matrix = jc_mutation_calc(n_alleles, mutation_rate=mutation_rate)

    for repeat_idx in range(n_repeats):
        for generation in range(1, n_generations):
            p = np.matmul(mutation_matrix, allele_freq[generation - 1, repeat_idx, :].transpose())
            n_chosen_alleles = dist_func(n_individuals,p)
            allele_freq[generation, repeat_idx,:] = n_chosen_alleles / n_individuals

    ##Plot the allele frequencies
    plot_multiallele(allele_freq, n_generations, alleles)


#sim_haploid(10, 100, 100, "Haploid")

##alleles:list, n_repeats, n_generations, n_individuals, name
#multiallele_simulator(["A", "B", "C", "D", "E", "F","G", "H"], 5, 100, 1000000, "Multiallele")
#multiallele_simulator(["A", "B", "C", "D", "E", "F","G", "H", "I", "J"], 10, 100, 100, "Multiallele")
#multiallele_simulator(["A", "B", "C"], 5, 1000000, 10000, "Multiallele")

##Multiallele (Fisher-Wright) model with selection

##Multiallele (Fisher-Wright) model with migration


##Multiallele (Fisher-Wright) model with mutation
## requires a mutation probability n_alleleXn_allele matrix (U)
## For mutation model, Jukes-Cantor model can be used.
## take into account the mutation rate (mu) and the mutation probability (U)

def jc_mutation_calc(n_alleles, mutation_rate=0.0):
    ##Jukes-Canto --> simple equal probability of mutation
    # mutation probability matrix
    U = np.zeros([n_alleles, n_alleles])
    U.fill(mutation_rate / (n_alleles - 1))
    np.fill_diagonal(U, 1 - mutation_rate)
    return U

fisher_wright_simulator(["A", "B", "C", "D"], 3, 10, 10, mutation_rate=1)