
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

def multiallele_simulator(alleles:list, n_repeats, n_generations, n_individuals, name):
    ##Multiallele (Fisher-Wright) model
    n_alleles = len(alleles)
    allele_freq = np.zeros([n_generations, n_repeats, n_alleles])

    # Set the initial condition at the first step
    allele_freq[0, :, :] = 1 / n_alleles

    for repeat_idx in range(n_repeats):
        for generation in range(1, n_generations):
            n_chosen_alleles = np.random.multinomial(n=n_individuals, pvals=allele_freq[generation - 1, repeat_idx,:])
            allele_freq[generation, repeat_idx,:] = n_chosen_alleles / n_individuals

    ##Plot the allele frequencies
    #rows=cols=n_alleles//2 + n_alleles%2
    columns=4
    rows=n_alleles//4 +1 if n_alleles%4 else n_alleles//4

    plt.figure(figsize=(18, 21))
    plt.subtitle = name
    plt.subplots_adjust(hspace=0.5)

    for allele, allele_idx in enumerate(alleles):
        ax = plt.subplot(rows,columns , allele_idx+1, )
        ax.plot(range(1, n_generations + 1), allele_freq[:, :, allele_idx])
        ax.set_xlabel("Generation")
        ax.set_ylabel(f"{allele} Allele frequency")
        ax.set_title(allele)
    plt.show()



#sim_haploid(10, 100, 100, "Haploid")

##alleles:list, n_repeats, n_generations, n_individuals, name
#multiallele_simulator(["A", "B", "C", "D", "E", "F","G", "H"], 5, 100, 1000000, "Multiallele")
#multiallele_simulator(["A", "B", "C", "D", "E", "F","G", "H", "I", "J"], 10, 100, 100, "Multiallele")
multiallele_simulator(["A", "B", "C"], 5, 1000000, 10000, "Multiallele")
