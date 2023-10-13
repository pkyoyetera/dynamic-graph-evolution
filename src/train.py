import os

import torch
import wandb

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.preprocessing import StandardScaler

from src.models import sts_conv_model


# Set some parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sequence of channel sizes
channels = np.array([1, 16, 64], [64, 16, 64])

kernel_size = 3  # Size of temporal kernel
K = 3  # Chebyshev filter size

# Training parameters
learning_rate = 1e-3
batch_size = 50
num_epochs = 2
num_layers = 2  # Number of STConv blocks
n_his = 20  # Number of historical time steps to consider (144 used)
n_pred = 5  # Steps into the future to predict

train_prop = 0.03  # 0.7 actual
val_prop = 0.02  # 0.2 actual
test_prop = 0.01  # 0.1 actual

# save path
model_save_path = os.path.join('models', 'sts_conv_model.pt')


# Load data
def data_transform(data, n_his, n_pred, device):
    """
    Transform data into input and target.
    :param data: slice of V matrix
    :param n_his: int, number of historical speed observations to consider
    :param n_pred: int, number of future speed observations to predict
    """
    num_nodes = data.shape[1]
    num_obs = len(data) - n_his - n_pred

    x = np.zeros([num_obs, n_his, num_nodes, 1])
    y = np.zeros([num_obs, num_nodes])

    obs_idx = 0
    for i in range(num_obs):
        head = i
        tail = i + n_his
        x[obs_idx, :, :, :] = data[head:tail].reshape(n_his, num_nodes, 1)
        y[obs_idx] = data[tail + n_pred - 1]

        obs_idx += 1

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)


def evaluate_model(model, loss_fn, data_iter, edge_index, edge_weight, device):
    model.eval()

    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x.to(device), edge_index, edge_weight).view(len(x), -1)
            loss = loss_fn(y_pred, y)
            l_sum = loss.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def evaluate_metric(model, data_iter, scaler, edge_index, edge_weight, device):
    model.eval()

    with torch.no_grad():
        mae, mape, mse = [], [], []

        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(
                model(x.to(device), edge_index, edge_weight).view(len(x), -1).cpu().numpy()
            ).reshape(-1)

            d = np.abs(y - y_pred)
            mae += d.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()

        return np.mean(mae), np.mean(mape), np.mean(mse)


weighted_adj_matrix_path = os.path.join('data', 'Processed_data', 'Graph_Inputs', 'W_50.csv')
W = pd.read_csv(weighted_adj_matrix_path)

feature_vector_path = os.path.join('data', 'Processed_data', 'Graph_Inputs', 'V_50.csv')
V = pd.read_csv(feature_vector_path)


# Adapted from Hao Wei
num_samples, num_nodes = V.shape

# Get splits for V matrix
len_train = round(num_samples * train_prop)
len_val = round(num_samples * val_prop)

train = V[:len_train]
val = V[len_train: len_train+len_val]
test = V[len_train + len_val: len_train + len_val + round(num_samples * test_prop)]

# Normalize values
scaler = StandardScaler()
train = np.nan_to_num(scaler.fit_transform(train))
val = np.nan_to_num(scaler.fit_transform(val))
test = np.nan_to_num(scaler.fit_transform(test))

# Create training examples with helper function
x_train, y_train = data_transform(train, n_his, n_pred, device)
x_val, y_val = data_transform(val, n_his, n_pred, device)
x_test, y_test = data_transform(test, n_his, n_pred, device)

#  Create torch datasets and dataloaders
train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)


# Format graph for PyG inputs
G = sp.coo_matrix(W.values)
edge_index = torch.tensor(np.array([G.row, G.col]), dtype=torch.int64).to(device)
edge_weight = torch.tensor(G.data).float().to(device)




