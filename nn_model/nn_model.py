### NN Model related tasks
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_torch_data_loaders(df:pd.DataFrame, batch_size:int=64, shuffle:bool=True, test_split:float=0.3):
    df_t = torch.tensor(df.values, dtype=torch.float32, device="cpu")
    training_data_fw, test_data_fw = random_split(df_t, [1.0-test_split, test_split])
    train_dataloader = DataLoader(training_data_fw, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_data_fw, batch_size=batch_size)
    return train_dataloader, test_dataloader


def get_nn_model(n_inputs:int, n_outputs:int, device:str="cpu"):
    class NeuralNetwork(nn.Module):
        def __init__(self, n_inputs, n_outputs):
            super().__init__()
            # self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(n_inputs, n_inputs * 4),
                nn.GELU(),
                nn.Linear(n_inputs * 4, n_inputs * 2),
                nn.GELU(),
                nn.Linear(n_inputs * 2, n_inputs // 2),
                nn.GELU(),
                nn.Linear(n_inputs // 2, n_outputs * 4),
                nn.GELU(),
                nn.Linear(n_outputs * 4, n_outputs),
                nn.GELU()
            )

        def forward(self, x):
            # x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork(n_inputs, n_outputs).to(device)
    return model


def get_loss_fn(loss:str="mse"):
    if loss == "mse":
        return nn.MSELoss()
    elif loss == "nll":
        return nn.NLLLoss()
    elif loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss == "l1":
        return nn.L1Loss()
    elif loss == "huber":
        return nn.HuberLoss()
    elif loss == "smooth_l1":
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Invalid loss function: {loss}")


def get_optimizer(model, optimizer:str="sgd", lr:float=1e-3, momentum:float=0.9, weight_decay:float=0.0):
    if optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adamax":
        return torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adadelta":
        return torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer}")


def train(dataloader, model, loss_fn, optimizer, device:str="cpu"):
    size = len(dataloader.dataset)
    model.train()
    for batch, data in enumerate(dataloader):
        ## y includes y_ne, y_mu
        X, y = data[:, :-2], data[:, -2:]
        # Calculate for effective population first
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device:str="cpu"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data in dataloader:
            X, y = data[:, :-2], data[:, -2:]
            # Calculate for effective population first
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (np.isclose(pred, y)).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


def train_model(n_epochs, train_dataloader, test_dataloader, model, loss_fn, optimizer, device:str="cpu"):
    test_loss_lst = []
    train_loss_lst = []
    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device=device)
        train_loss_lst.append(test(train_dataloader, model, loss_fn, device=device))
        test_loss_lst.append(test(test_dataloader, model, loss_fn, device=device))
    print("Done!")
    return train_loss_lst, test_loss_lst


def plot_loss(train_loss_lst, test_loss_lst):
    fig, ax = plt.subplots()
    ax.plot(test_loss_lst, label="test_loss")
    ax.plot(train_loss_lst, label="train_loss")
    ax.set(title='Test and Train Loss', xlabel='Number of epochs', ylabel='Loss')
    fig.legend(loc="center")
    plt.show()


def save_model(model, path:str="data/model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")


def get_r2_array(dataloader,model, scaler, n_batches, plot=False, save_path="data/figures/r2_scatter.png"):
    from sklearn.metrics import r2_score
    model.eval()
    batch_size = len(dataloader.dataset)
    num_batches = len(dataloader)
    if plot:
        plt.ion()
        plt.figure(figsize=(15, 18))
        figure, (ax_ne, ax_mu) = plt.subplots(nrows=2)

    with torch.no_grad():
        values_array = np.zeros([n_batches, 4+2, batch_size])
        ne_r2_avg = 0
        mu_r2_avg = 0
        for counter, data in enumerate(dataloader):
            if counter+1 > n_batches:
                break
            X, y = data[:,:-2], data[:,-2:]
            pred = model(X)

            ##Rescale the inputs to original values
            y_true_trans = scaler.inverse_transform(data)[:,-2:]
            y_hat_trans =  scaler.inverse_transform(np.c_[X,pred])[:,-2:]
            ne_t, ne_p =  y_true_trans[:,0], y_hat_trans[:,0]
            mu_t, mu_p =  y_true_trans[:,1], y_hat_trans[:,1]

            ne_msd = (1/batch_size)*((ne_t/np.linalg.norm(ne_t))-(ne_p/np.linalg.norm(ne_p)))**2
            ne_r2 = r2_score(ne_t, ne_p)
            ne_r2_avg += ne_r2
            mu_msd = (1/batch_size)*((mu_t/np.linalg.norm(mu_t))-(mu_p/np.linalg.norm(mu_p)))**2
            mu_r2 = r2_score(mu_t, mu_p)
            mu_r2_avg += mu_r2


            #values_array[counter,:,:] = ne_t, ne_p, mu_t, mu_p, ne_r2, mu_r2

            if plot:
                ax_ne.scatter(ne_t, ne_p, alpha=0.7)#, label=f"batch_number {counter}")
                ax_mu.scatter(mu_t, mu_p, alpha=0.7)
                # drawing updated values
                figure.canvas.draw()

                # This will run the GUI event
                # loop until all UI events
                # currently waiting have been processed
                figure.canvas.flush_events()


        print(f"number of last batch {counter}")
        ne_r2_avg = ne_r2_avg / counter
        ax_ne.annotate(f"r-squared = {ne_r2_avg.__round__(3)}", (0,999))
        ax_ne.plot([0, 1000], [0, 1000], 'k--', lw=3, label="Identity Line")
        ax_ne.plot(np.linspace(0,1000, 1000)*ne_r2_avg, 'r--', lw=3, label="R^2 Line")
        ax_ne.set(title='Effective Population Size',xlabel='Ne True Values',ylabel='Ne Predicted Values')

        mu_r2_avg = mu_r2_avg/counter
        ax_mu.annotate(f"r-squared = {mu_r2_avg.__round__(3)}", (0,0.9))
        ax_mu.plot([0,1], [0,1], 'k--', lw=3)
        ax_mu.plot(np.linspace(0,1), np.linspace(0,1)*mu_r2_avg, 'r--', lw=3)
        ax_mu.set(title='Mutation Rate',xlabel='Mu True Values',ylabel='Mu Predicted Values')

        figure.legend(loc="center left", framealpha=0.4, borderpad=0.4)
        figure.tight_layout()
        figure.savefig(save_path)
        figure.show()


def process_nn_model(
        df:pd.DataFrame, loss:str, optimiser:str, epochs:int, scaler_type="max_abs",
        batch_size=64, test_size=0.3, device:str="cpu", save_path:str="data/model.pth",
        lr:float=1e-3, momentum:float=0.9, weight_decay:float=0.0
):
    from nn_model.preprocess_data import normalise
    df, scaler = normalise(df, scaler_type=scaler_type)
    train_df, test_df = get_torch_data_loaders(df, batch_size, shuffle=True, test_split=test_size)
    model = get_nn_model(df.iloc[:,:-2].shape[1], 2)
    loss_fn = get_loss_fn(loss)
    optimiser= get_optimizer(model, optimiser, lr, momentum, weight_decay)
    train_loss, test_loss = train_model(epochs,
                train_df, test_df, model,
                loss_fn=loss_fn, optimizer=optimiser,  device=device)

    save_model(model, save_path)
    plot_loss(train_loss, test_loss)
    get_r2_array(test_df, model, scaler, 50, plot=True, save_path=save_path.replace(".pth", ".png"))

