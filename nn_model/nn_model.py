import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


def plot_correlation_matrix(corr):
    """
    Plot the correlation matrix of pandas dataset.
    """
    import matplotlib.pyplot as plt
    plt.imshow(corr, interpolation='nearest')
    plt.colorbar()
    plt.show()


indices = tree_stats_idx = {
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

allele_stats_indices = {
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
##Add allele related stats
indices.update({key: len(tree_stats_idx) + idx for key, idx in allele_stats_indices.items()})

indices.update({
    "n_individuals" : 56 ,
    "mutation_rate" : 57,
    "n_generations" : 58,
    "max_mutations" : 59
})

path_ss = "/Users/berk/Projects/jlees/data/plots/20230509-0906/simulation_results.npy"
data_ss = np.load(path_ss)
pd_ss = pd.DataFrame(data_ss, columns=indices.keys())

pd_ss = pd_ss.drop_duplicates()
pd_ss.drop(
    columns=[
        "max_mutations",
        "n_generations"
    ], inplace=True)

rows_dropped = pd_ss.iloc[:,:-2].any(axis=1)
pd_ss = pd_ss.loc[rows_dropped, :]

pd_ss.replace([np.inf, -np.inf], np.nan, inplace=True)
pd_ss = pd_ss.dropna(axis=1, how="all")
### Drop columns are mostly nans
column_na_freq_fltr = pd_ss.isnull().sum()/max(pd_ss.count()) > 0.4
pd_ss.drop(columns=pd_ss.loc[:, column_na_freq_fltr].columns, inplace=True)

"""##Plots
pd_ss.cov().plot()
pd_ss.kurtosis().plot()
plot_correlation_matrix(pd_ss.corr())
"""

### Deal with high correlations
pd_features = pd_ss.iloc[:,:-2]
# Compute the correlation matrix
corr_matrix = pd_features.corr().abs()

# Set the threshold for high correlation
corr_threshold = 0.95

# Get the upper triangular matrix
upper_triangular = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with high correlation
high_corr_features = {column for column in upper_triangular.columns if any(upper_triangular[column] > corr_threshold)}

pd_ss = pd_ss.drop(high_corr_features, axis=1)

### Fill or drop na
### Drop all rows containing na for now

pd_naless = pd_ss.dropna(axis=0, how="any")
### Normalise the data
from sklearn.preprocessing import MinMaxScaler
pd_norm = pd.DataFrame(MinMaxScaler().fit_transform(pd_naless), columns=pd_naless.columns)

### NN Model related tasks
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

batch_size =64
df_t = torch.tensor(pd_norm.values, dtype=torch.float32, device="cpu")
training_data_fw, test_data_fw = random_split(df_t, [0.7, 0.3])
train_dataloader = DataLoader(training_data_fw, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data_fw, batch_size=batch_size)

# Get cpu, gpu or mps device for training.
"""device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)"""
device = "cpu"
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_inputs, n_inputs*2),
            nn.ReLU(),
            nn.Linear(n_inputs*2, n_inputs//2),
            nn.ReLU(),
            nn.Linear(n_inputs//2, n_outputs)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork(pd_norm.iloc[:,:-2].shape[1], 1).to(device)

#### Optimizing the Model Parameters
#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, data in enumerate(train_dataloader):
        X, y_ne, y_mu = data[:,:-2], data[:,-2], data[:,-1]
        #Calculate for effective population first
        X, y = X.to(device), y_ne.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data in dataloader:
            X, y_ne, y_mu = data[:,:-2], data[:,-2], data[:,-1]
            #Calculate for effective population first
            X, y = X.to(device), y_ne.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 25
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

#### Save Model
torch.save(model.state_dict(), "data/model.pth")
print("Saved PyTorch Model State to data/model.pth")
#%%
