"""
Pytorch Implementation of Neural Graph Collaborative Filtering (NGCF) (https://doi.org/10.1145/3331184.3331267)

This file is an adaptation of the  code written by the following authors:
Mohammed Yusuf Noor, Muhammed Imran Özyar, Calin Vasile Simon
"""

import torch

import numpy as np


def early_stopping(log_value, best_value, stopping_step, flag_step, expected_order="asc"):
    """
    Check if early_stopping is needed
    Function copied from original code
    """
    assert expected_order in ["asc", "des"]
    if (expected_order == "asc" and log_value >= best_value) or (
        expected_order == "des" and log_value <= best_value
    ):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print(f"Early stopping at step: {flag_step} log: {log_value}")
        should_stop = True
    else:
        should_stop = False

    return best_value, stopping_step, should_stop


def train(model, data_generator, optimizer):
    """
    Train the model PyTorch style
    @param: model: PyTorch model
    @param: data_generator: Data object
    @param: optimizer: PyTorch optimizer
    """
    model.train()
    n_batch = data_generator.n_train // data_generator.batch_size + 1
    running_loss = 0
    for _ in range(n_batch):
        u, i, j = data_generator.sample()
        optimizer.zero_grad()
        loss = model(u, i, j)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss


def split_matrix(X, n_splits=100):
    """
    Split a matrix/Tensor into n_folds (for the user embeddings and the R matrices)
    @param: X: matrix to be split
    @param: n_folds: number of folds
    @return: splits: split matrices
    """
    splits = []
    chunk_size = X.shape[0] // n_splits
    for i in range(n_splits):
        start = i * chunk_size
        end = X.shape[0] if i == n_splits - 1 else (i + 1) * chunk_size
        splits.append(X[start:end])
    return splits


def compute_ndcg_k(pred_items, test_items, test_indices, k):
    """
    Compute NDCG@k
    @param: pred_items: binary tensor with 1s in those locations corresponding to the predicted item interactions
    @param: test_items: binary tensor with 1s in locations corresponding to the real test interactions
    @param: test_indices: tensor with the location of the top-k predicted items
    @param: k: k'th-order
    @return: NDCG@k
    """
    r = (test_items * pred_items).gather(1, test_indices)
    f = torch.from_numpy(np.log2(np.arange(2, k + 2))).float().cuda()
    dcg = (r[:, :k] / f).sum(1)
    dcg_max = (torch.sort(r, dim=1, descending=True)[0][:, :k] / f).sum(1)
    ndcg = dcg / dcg_max
    ndcg[torch.isnan(ndcg)] = 0
    return ndcg


def eval_model(u_emb, i_emb, Rtr, Rte, k):
    """
    Evaluate the model
    @param: u_emb: User embeddings
    @param: i_emb: Item embeddings
    @param: Rtr: Sparse matrix with the training interactions
    @param: Rte: Sparse matrix with the testing interactions
    @param: k : kth-order for metrics
    @return: result: Dictionary with lists corresponding to the metrics at order k for k in Ks
    """
    # split matrices
    ue_splits = split_matrix(u_emb)
    tr_splits = split_matrix(Rtr)
    te_splits = split_matrix(Rte)

    recall_k, ndcg_k = [], []
    # compute results for split matrices
    for ue_f, tr_f, te_f in zip(ue_splits, tr_splits, te_splits):
        scores = torch.mm(ue_f, i_emb.t())

        test_items = torch.from_numpy(te_f.todense()).float().cuda()
        non_train_items = torch.from_numpy(1 - (tr_f.todense())).float().cuda()
        scores = scores * non_train_items

        _, test_indices = torch.topk(scores, dim=1, k=k)
        pred_items = torch.zeros_like(scores).float()
        pred_items.scatter_(dim=1, index=test_indices, src=torch.tensor(1.0).cuda())  # issue here

        topk_preds = torch.zeros_like(scores).float()
        topk_preds.scatter_(
            dim=1, index=test_indices[:, :k], src=torch.tensor(1.0)
        )  # issue here

        TP = (test_items * topk_preds).sum(1)
        rec = TP / test_items.sum(1)
        ndcg = compute_ndcg_k(pred_items, test_items, test_indices, k)

        recall_k.append(rec)
        ndcg_k.append(ndcg)

    return torch.cat(recall_k).mean(), torch.cat(ndcg_k).mean()
