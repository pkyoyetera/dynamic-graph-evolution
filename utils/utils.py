#
# Patrick Kyoyetera, Vijay Gochinghar
# 2023
#

import random

import torch

import pandas as pd

import torch.nn.functional as F


def data_loader(data, _batch_size, n_usr, n_itm, device=torch.device("cpu")):
    def negative_sampler(x):
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id

    intersection_df = (
        data.groupby("user_id_index")["item_id_index"].apply(list).reset_index()
    )
    indices = [x for x in range(n_usr)]

    if n_usr < _batch_size:
        _users = [random.choice(indices) for _ in range(_batch_size)]
    else:
        _users = random.sample(indices, _batch_size)

    _users.sort()
    users_df = pd.DataFrame(_users, columns=["users"])

    intersection_df = pd.merge(
        intersection_df,
        users_df,
        how="right",
        left_on="user_id_index",
        right_on="users",
    )
    positive_items = (
        intersection_df["item_id_index"].apply(lambda x: random.choice(x)).values
    )
    negative_items = (
        intersection_df["item_id_index"].apply(lambda x: negative_sampler(x)).values
    )

    return (
        torch.LongTensor(list(_users)).to(device),
        torch.LongTensor(list(positive_items)).to(device) + n_usr,
        torch.LongTensor(list(negative_items)).to(device) + n_usr,
    )


# BPR loss
def compute_bpr_loss(
    _users, user_emb, pos_emb_, neg_emb_, init_user_emb, init_pos_emb, init_neg_emb
):
    # Compute loss from initial embeddings, for regularization
    reg_loss_ = (
        (1 / 2)
        * (
            init_user_emb.norm().pow(2)
            + init_pos_emb.norm().pow(2)
            + init_neg_emb.norm().pow(2)
        )
        / float(len(_users))
    )

    # Compute BPR loss from user, and positive item, and negative item embeddings
    pos_scores = torch.mul(user_emb, pos_emb_).sum(dim=1)
    neg_scores = torch.mul(user_emb, neg_emb_).sum(dim=1)

    bpr_loss_ = torch.mean(F.softplus(neg_scores - pos_scores))

    return bpr_loss_, reg_loss_


def get_metrics(
    user_embeddings, item_embeddings, n_users_, n_items_, train_data_, test_data_, K, device=torch.device("cpu")
):
    # compute the score of all user-item pairs
    relevance_score = torch.matmul(
        user_embeddings, torch.transpose(item_embeddings, 0, 1)
    )

    # Dense tensor for all user-item interactions in the training data
    i = torch.stack(
        (
            torch.LongTensor(train_data_["user_id_index"].values),
            torch.LongTensor(train_data_["item_id_index"].values),
        )
    )
    v = torch.ones((len(train_data_)), dtype=torch.float64)

    interactions_t = (
        torch.sparse.FloatTensor(i, v, (n_users_, n_items_)).to_dense().to(device)
    )

    # mask out training user-item interactions from metric computation
    relevance_score = torch.mul(relevance_score, (1 - interactions_t))

    # compute top scoring items for each user
    topk_relevance_indices = torch.topk(relevance_score, K).indices
    topk_relevance_indices_df = pd.DataFrame(
        topk_relevance_indices.cpu().numpy(),
        columns=["top_index_" + str(x + 1) for x in range(K)],
    )
    topk_relevance_indices_df["user_ID"] = topk_relevance_indices_df.index
    topk_relevance_indices_df["top_relevant_item"] = topk_relevance_indices_df[
        ["top_index_" + str(x + 1) for x in range(K)]
    ].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[
        ["user_ID", "top_relevant_item"]
    ]

    # measure overlap between recommended (top-scoring) and held-out user-item interactions
    test_interacted_items = (
        test_data_.groupby("user_id_index")["item_id_index"].apply(list).reset_index()
    )

    metrics_df = pd.merge(
        test_interacted_items,
        topk_relevance_indices_df,
        how="left",
        left_on="user_id_index",
        right_on=["user_ID"],
    )
    metrics_df["intersecting_item"] = [
        list(set(a).intersection(b))
        for a, b in zip(metrics_df.item_id_index, metrics_df.top_relevant_item)
    ]

    metrics_df["recall"] = metrics_df.apply(
        lambda x: len(x["intersecting_item"]) / len(x["item_id_index"]), axis=1
    )
    metrics_df["precision"] = metrics_df.apply(
        lambda x: len(x["intersecting_item"]) / K, axis=1
    )

    return metrics_df["recall"].mean(), metrics_df["precision"].mean()
