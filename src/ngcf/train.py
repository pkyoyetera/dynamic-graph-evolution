"""


"""

import os

from time import time
from datetime import datetime

import torch
import wandb

import numpy as np
import pandas as pd

from utils.load_data import Data
from utils.parser import parse_args
from utils.helper_functions import (early_stopping, train, split_matrix, compute_ndcg_k, eval_model)

from model import NGCF

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(0)

if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Set up wandb
    ...

    # Set up data
    data_dir = args.data_dir
    dataset = args.dataset  # ml-100k
    batch_size = args.batch_size
    layers = eval(args.layers)  # [64, 64, 64]
    emb_dim = args.emb_dim
    lr = args.lr
    reg = args.reg
    mess_dropout = args.mess_dropout
    node_dropout = args.node_dropout
    k = args.k

    # Generate the NGCF-adjacency matrix
    data_generator = Data(path=args.data_dir + args.dataset, batch_size=args.batch_size)
    adj_mtx = data_generator.get_adj_mat()

    # Create model name and save
    model_name = f"NGCF_bs{args.batch_size}_nemb_{args.emb_dim}_nodedr_{args.node_dropout}" # + \
    # "_bs_" + str(batch_size) + \
    # "_nemb_" + str(emb_dim) + \
    # "_layers_" + str(layers) + \
    # "_nodedr_" + str(node_dropout) + \
    # "_messdr_" + str(mess_dropout) + \
    # "_reg_" + str(reg) + \
    # "_lr_" + str(lr)

    # Create NGCF model
    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 emb_dim,
                 layers,
                 reg,
                 node_dropout,
                 mess_dropout,
                 adj_mtx,
                 use_cuda=use_cuda)

    if use_cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Current best
    curr_best_metric = 0
    curr_best_loss, stopping_step, should_stop = 1e3, 0, False

    today = datetime.now()

    print(f"Start at {today}")
    print(f"Training on {device}")

    results = {'epoch': [], 'train_loss': [], 'recall': [], 'val_loss': [], 'test_loss': [], 'ndcg': []}

    for epoch in range(args.n_epochs):
        t1 = time()
        loss = train(model, data_generator, optimizer)

        training_time = time() - t1
        print(f"Epoch: {epoch}, Training Loss: {loss:.4f}, Training Time: {training_time}")

        if epoch % args.eval_every == 0:
            with torch.no_grad():
                t2 = time()
                recall, ndcg = eval_model(model.u_g_embeddings,
                                          model.i_g_embeddings,
                                          data_generator.R_train,
                                          data_generator.R_test,
                                          k)
            print(f"Epoch: {epoch}, Recall@{k}: {recall:.4f}, NDCG@{k}: {ndcg:.4f}")

            curr_best_metric, stopping_step, should_stop = early_stopping(recall, curr_best_metric, stopping_step, flag_step=5)

            # Save
            results['epoch'].append(epoch)
            results['train_loss'].append(loss)
            results['recall'].append(recall.item())
            results['ndcg'].append(ndcg)
            # results['training_time'].append(training_time)
        else:
            results['epoch'].append(epoch)
            results['train_loss'].append(loss)
            results['recall'].append(None)
            results['ndcg'].append(None)

            if should_stop:
                break

    if args.save_results:
        date = today.strftime("%Y-%m-%d:%H:%M:%S")

        # Save model as pt file
        if os.path.isdir(args.save_dir):
            torch.save(model.state_dict(), args.save_dir + model_name + "_" + date + ".pt")



