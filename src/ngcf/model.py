import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp


class NGCF(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        emb_dim,
        layers,
        reg,
        node_dropout,
        message_dropout,
        adjacency_matrix,
        device="cpu",
    ):
        super().__init__()

        # Class attributes
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.adjacency_matrix = adjacency_matrix
        self.laplacian = adjacency_matrix - sp.eye(adjacency_matrix.shape[0])
        self.reg = reg
        self.layers = layers
        self.n_layers = len(self.layers)
        self.node_dropout = node_dropout
        self.message_dropout = message_dropout

        # Initialize weights
        self.weight_dict = self._init_weights()
        print("Initialized model weights.")

        # Create matrix 'A', PyTorch sparse tensor of SP adjacency matrix
        self.A = self._convert_sp_mat_to_sp_tensor(self.adjacency_matrix)
        self.L = self._convert_sp_mat_to_sp_tensor(self.laplacian)

    def _init_weights(self):
        """
        Initializes model weights.
        """
        print(f"Initializing model weights...")

        weight_dict = nn.ParameterDict()
        weight_dict["user_embedding"] = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.n_users, self.emb_dim))
        )
        weight_dict["item_embedding"] = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(self.n_items, self.emb_dim))
        )
        # todo move to device

        weight_size_list = [self.emb_dim] + self.layers

        for k in range(self.n_layers):
            weight_dict[f"W_gc_{k}"] = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.empty(weight_size_list[k], weight_size_list[k + 1])
                )
            )
            weight_dict[f"b_gc_{k}"] = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(1, weight_size_list[k + 1]))
            )

            weight_dict[f"W_bi_{k}"] = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.empty(weight_size_list[k], weight_size_list[k + 1])
                )
            )
            weight_dict[f"b_bi_{k}"] = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(1, weight_size_list[k + 1]))
            )

        return weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X) -> torch.Tensor:
        """
        Converts SciPy sparse matrix to sparse PyTorch tensor.
        @param X: SciPy sparse matrix
        @return: PyTorch sparse tensor
        """
        coo = X.tocoo().astype(np.float32)
        # indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
        # values = torch.from_numpy(coo.data)
        # shape = torch.Size(coo.shape)
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)

        # return torch.sparse.FloatTensor(indices, values, shape)
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _dropout_sparse(self, X):
        """
        Drop individual locations in X
        """
        node_dropout_mask = (
            ((self.node_dropout) + torch.rand(X._nnz())).floor().bool()
        )  # .to(device)
        i = X.coalesce().indices()
        v = X.coalesce()._values()

        i[:, node_dropout_mask] = 0
        v[node_dropout_mask] = 0

        x_dropout = torch.sparse.FloatTensor(i, v, X.shape)  # .to(X.device)

        return x_dropout.mul(1 / (1 - self.node_dropout))

    def forward(self, u, i, j):
        """
        Computes the forward pass
        @param u: user
        @param i: positive item (User has interacted with this item)
        @param j: negative item (User has not interacted with this item)
        """
        # Apply dropout mask
        a_hat = self._dropout_sparse(self.A) if self.node_dropout > 0 else self.A
        l_hat = self._dropout_sparse(self.L) if self.node_dropout > 0 else self.L

        ego_embeddings = torch.cat(
            [self.weight_dict["user_embedding"], self.weight_dict["item_embedding"]],
            dim=0,
        )

        all_embeddings = [ego_embeddings]

        #
        for k in range(self.n_layers):
            # Weighted sum message of neighbors
            side_embeddings = torch.sparse.mm(a_hat, ego_embeddings)
            side_l_embeddings = torch.sparse.mm(l_hat, ego_embeddings)

            # transformed sum messages of neighbors
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict[f"W_gc_{k}"]) \
                             + self.weight_dict[f"b_gc_{k}"]

            # bi messages of neighbors
            bi_embeddings = torch.mul(ego_embeddings, side_l_embeddings)
            # transformed bi messages of neighbors
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict[f"W_bi_{k}"]) \
                            + self.weight_dict[f"b_bi_{k}"]

            # non-linear activation
            ego_embeddings = F.leaky_relu(sum_embeddings + bi_embeddings)
            ego_embeddings = nn.Dropout(self.message_dropout)(ego_embeddings)

            # Norm
            norm_embeddings = F.normalize(ego_embeddings, p=1, dim=1)

            all_embeddings.append(norm_embeddings)

        all_embeddings = torch.cat(all_embeddings, 1)

        # Back to user/item dimension
        u_g_embeddings, i_g_embeddings = all_embeddings.split(
            [self.n_users, self.n_items], 0
        )

        self.u_g_embeddings = nn.Parameter(u_g_embeddings)
        self.i_g_embeddings = nn.Parameter(i_g_embeddings)

        u_emb = u_g_embeddings[u]  # User embeddings
        p_emb = i_g_embeddings[i]  # Positive item embeddings
        n_emb = i_g_embeddings[j]  # Negative item embeddings

        y_ui = torch.mul(u_emb, p_emb).sum(dim=1)
        y_uj = torch.mul(u_emb, n_emb).sum(dim=1)
        log_prob = (torch.log(torch.sigmoid(y_ui - y_uj))).mean()

        # Compute BPR loss
        bpr_loss = -log_prob
        if self.reg > 0:
            l2_norm = (torch.sum(u_emb**2) / 2.
                       + torch.sum(p_emb**2) / 2.
                       + torch.sum(n_emb**2) / 2.) / u_emb.shape[0]
            l2_reg = self.reg * l2_norm
            bpr_loss = -log_prob + l2_reg

        return bpr_loss
