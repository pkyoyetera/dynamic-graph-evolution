#
# Patrick Kyoyetera, Vijay Gochinghar
# 2023
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class LightGCNConv(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr="add")

    def forward(self, x, edge_index):
        # Compute normalization
        from_, to_ = edge_index

        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class NGCFConv(MessagePassing):
    def __init__(self, _latent_dim, dropout, bias=True, **kwargs):
        super(NGCFConv, self).__init__(aggr="add", **kwargs)

        self.dropout = dropout

        self.lin_1 = nn.Linear(_latent_dim, _latent_dim, bias=bias)
        self.lin_2 = nn.Linear(_latent_dim, _latent_dim, bias=bias)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.lin_1.weight)
        nn.init.xavier_uniform_(self.lin_2.weight)

    def forward(self, x, edge_index):
        # Compute normalization
        from_, to_ = edge_index

        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages
        _out = self.propagate(edge_index, x=(x, x), norm=norm)

        # Update after aggregation
        _out += self.lin_1(x)
        _out = F.dropout(_out, self.dropout, self.training)

        return F.leaky_relu(_out)

    def message(self, x_j, x_i, norm):
        return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i))


class RecommenderSystem(nn.Module):
    def __init__(
        self,
        _latent_dim,
        _num_layers,
        num_users,
        num_items,
        conv_name="ngcf",
        dropout=0.5,
    ):
        super(RecommenderSystem, self).__init__()

        self.embedding = nn.Embedding(num_users + num_items, _latent_dim)
        self.dropout = nn.Dropout(dropout)

        self.conv_arch = conv_name

        if self.conv_arch == "lightgcn":
            self.convs = nn.ModuleList(LightGCNConv() for _ in range(_num_layers))
        else:
            # NGCF convs
            self.convs = nn.ModuleList(
                NGCFConv(_latent_dim, dropout) for _ in range(_num_layers)
            )

        self.init_parameters()

    def init_parameters(self):
        if self.conv_arch == "ngcf":
            nn.init.xavier_uniform_(self.embedding.weight)
        elif self.conv_arch == "lightgcn":
            nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index):
        emb_init = self.embedding.weight
        embs = [emb_init]

        emb = emb_init
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)

        # out = (torch.cat(embs, dim=-1))
        out = (
            torch.cat(embs, dim=-1)
            if self.conv_arch == "ngcf"
            else torch.mean(torch.stack(embs, dim=0), dim=0)
        )

        return emb_init, out

    def encode_minibatch(self, _users, positive_items, negative_items, edge_index):
        emb0, _out = self(edge_index)  # .to(torch.int64))

        return (
            _out[_users],
            _out[positive_items],
            _out[negative_items],
            emb0[_users],
            emb0[positive_items],
            emb0[negative_items],
        )
