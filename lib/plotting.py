### This script will plot how summary statistics change over input parameters for a given model.

## Gather inputs

## update plot woth every loop

## save plot

### Output: plots of summary statistics over input parameters


from lib.simulator import simulator
from lib.phylogenetic_tree import PhyloTree
from lib.alleles import Alleles
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


in_fasta = "/Users/berk/Projects/jlees/data/WF_input.fasta"
n_generations = 30
n_individuals = [8, 16, 20, 32, 64, 80, 128, 156, 200, 256]
#n_individuals = [32, 64]
mutation_rate = [0.01, 0.001, 0.005, 0.05, 0.16, 0.33, 0.45, 0.5, 0.66, 0.82, 0.99]
#mutation_rate = [0.33]
max_mutations = 300
batch_size = 50

indices = PhyloTree.tree_stats_idx.copy()
##Add allele related stats
indices.update({key: len(PhyloTree.tree_stats_idx) + idx for key, idx in Alleles.allele_stats_indices.items()})
mlists = []
#ax = plt.subplot(len(n_individuals), len(mutation_rate), idx + 1)
m = None
for mutrate in mutation_rate:
    for ne in n_individuals:
        if m is None:
            m = simulator(
                ne, mutrate,
                in_fasta, n_generations, max_mutations,
                batch_size=1, save_data=False,
                add_parameters=True,
                filter_below=0.00005
            )
        else:
            m = np.r_[m, simulator(
                ne, mutrate,
                in_fasta, n_generations, max_mutations,
                batch_size=batch_size, save_data=False,
                add_parameters=True,
                filter_below=0.00005
            )]


### n_individuals, mutation_rate, n_generations, max_mutations
## m[:,-3] --> mutation rate
## m[:,-4] --> n_individuals
## m[:,-2] --> n_generations
## m[:,-1] --> max_mutations
from config import DATA_PATH
import os
import time
outdir = os.path.join(DATA_PATH, "plots", time.strftime("%Y%m%d-%H%M"))
os.makedirs(outdir, exist_ok=True)

for idx_name, idx in indices.items():
    fig = plt.figure()
    axs = plt.axes(projection='3d')
    axs.set_title(f"{idx_name}")
    axs.set_xlabel("Ne")
    axs.set_ylabel("mu")
    axs.set_zlabel(idx_name)
    axs.scatter3D(m[:,-4], m[:,-3], m[:,idx], cmap="Greens") #c=m[:,idx], cmap='Greens')
    #axs.plot_surface(m[:,-4], m[:,-3], m[:,idx], cmap='viridis', edgecolor='none')
    #axs.plot3D(m[:,-4], m[:,-3], m[:,idx], 'gray')
    plt.savefig(os.path.join(outdir, f"{idx_name}.png"))
    plt.close(fig)
    #plt.show()

np.save(os.path.join(outdir, "simulation_results.npy"), m)

## TODO: 1-add more stats, 2-analise which stats more informative,(PCA, correlation, etc.) 3- Deep Learning project
