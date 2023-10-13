import argparse
import os

import torch
import wandb

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.models.sts_conv_model import TrafficModel


# Sequence of channel sizes
channels = np.array([[1, 16, 64], [64, 16, 64]])

# kernel_size = 3  # Size of temporal kernel
# K = 3  # Chebyshev filter size
#
# # Training parameters
# train_prop = 0.03  # 0.7 actual
# val_prop = 0.02  # 0.2 actual
# test_prop = 0.01  # 0.1 actual


# Load data
def data_transform(data, args, device):
    """
    Transform data into input and target.
    :param data: slice of V matrix
    :param args: argparse.Namespace, arguments passed to script
    """
    num_nodes = data.shape[1]
    num_obs = len(data) - args.n_his - args.n_pred

    x = np.zeros([num_obs, args.n_his, num_nodes, 1])
    y = np.zeros([num_obs, num_nodes])

    obs_idx = 0
    for i in range(num_obs):
        head = i
        tail = i + args.n_his
        x[obs_idx, :, :, :] = data[head:tail].reshape(args.n_his, num_nodes, 1)
        y[obs_idx] = data[tail + args.n_pred - 1]

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

#
# def train_model(model, train_iter, loss_fn, optimizer, )


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


def load_data(args: argparse.Namespace, device: torch.device):
    if not args.weighted_adj_matrix_path:
        W = pd.read_csv(os.path.join('../data', 'Processed_data', 'Graph_Inputs', 'W_50.csv'))
    else:
        W = pd.read_csv(args.weighted_adj_matrix_path)

    if not args.feature_vector_path:
        V = pd.read_csv(os.path.join('../data', 'Processed_data', 'Graph_Inputs', 'V_50.csv'))
    else:
        V = pd.read_csv(args.feature_vector_path)

    # Adapted from Hao Wei
    num_samples, num_nodes = V.shape

    # Get splits for V matrix
    len_train = round(num_samples * args.train_prop)
    len_val = round(num_samples * args.val_prop)

    train = V[:len_train]
    val = V[len_train: len_train+len_val]
    test = V[len_train + len_val: len_train + len_val + round(num_samples * args.test_prop)]

    # Normalize values
    scaler = StandardScaler()
    train = np.nan_to_num(scaler.fit_transform(train))
    val = np.nan_to_num(scaler.fit_transform(val))
    test = np.nan_to_num(scaler.fit_transform(test))

    # Create training examples with helper function
    x_train, y_train = data_transform(train, args, device)
    x_val, y_val = data_transform(val, args, device)
    x_test, y_test = data_transform(test, args, device)

    #  Create torch datasets and dataloaders
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    test_data = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    val_data = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size)

    # Format graph for PyG inputs
    G = sp.coo_matrix(W)  # depends on pandas==1.4.3
    edge_index = torch.tensor(np.array([G.row, G.col]), dtype=torch.int64).to(device)
    edge_weight = torch.tensor(G.data).float().to(device)

    return train_loader, val_loader, test_loader, edge_index, edge_weight, num_nodes  #, scaler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weighted_adj_matrix_path', type=str, default=None)
    parser.add_argument('--feature_vector_path', type=str, default=None)
    parser.add_argument('--model_save_path', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2, help='Number of STConv blocks')
    parser.add_argument('--n_his', type=int, default=20, help='Number of historical time steps to consider')
    parser.add_argument('--n_pred', type=int, default=5, help='Steps into the future to predict')
    parser.add_argument('--train_prop', type=float, default=0.03)
    parser.add_argument('--val_prop', type=float, default=0.02)
    parser.add_argument('--test_prop', type=float, default=0.01)
    parser.add_argument('--K', type=int, default=3, help='Chebyshev filter size')
    parser.add_argument('--kernel_size', type=int, default=3, help='Size of temporal kernel')
    parser.add_argument('--normalization', type=str, default='sym', help='Normalization method for adjacency matrix')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_loader, val_loader, test_loader, edge_index, edge_weight, num_nodes = load_data(args, device)

    # Create model
    model = TrafficModel(
        device,
        num_nodes,
        channels,
        args,
    ).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    min_val_loss = np.inf

    for epoch in tqdm(range(args.num_epochs)):
        l_sum, n = 0.0, 0

        model.train()

        for x, y in tqdm(train_loader, desc='Batch', position=0):
            optimizer.zero_grad()

            y_pred = model(x.to(device), edge_index, edge_weight).view(len(x), -1)
            loss = loss_fn(y_pred, y)

            loss.backward()
            optimizer.step()

            l_sum += loss.item() * y.shape[0]
            n += y.shape[0]

        # Compute validation loss
        val_loss = evaluate_model(model, loss_fn, val_loader, edge_index, edge_weight, device)
        # Save model if validation loss is lower than previous minimum
        if val_loss < min_val_loss:
            torch.save(model.state_dict(), args.model_save_path)
            min_val_loss = val_loss
        print(f"Epoch: {epoch}, Training Loss: {l_sum / n}, Validation Loss: {val_loss}")
