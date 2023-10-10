import torch
import torch.nn as nn

from torch_geometric_temporal.nn import STConv


class FullyConnLayer(nn.Module):
    def __init__(self, c):
        super(FullyConnLayer, self).__init__()
        self.conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        return self.conv(x)


# Adapted from Hao Wei
class OutputLayer(nn.Module):
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()

        self.tconv = nn.Conv2d(c, c, (T, 1), 1, dilation=1, padding=(0, 0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation=1, padding=(0, 0))
        self.fc = FullyConnLayer(c)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)

        return self.fc(x_t2)


class TrafficModel(nn.Module):
    def __init__(
            self,
            device,
            num_nodes,
            channel_size_list,
            num_layers,
            kernel_size,
            K,
            window_size,
            normalization='sym',
            bias=True
    ):
        """
        :param device: torch.device
        :param num_nodes: int, number of nodes in input graph
        :param channel_size_list: List[int], 2D array representing feature dims throughout model
        :param num_layers: int, number of STConv blocks
        :param kernel_size: int, length of temporal kernel
        :param K: int, size of Chebyshev filter for spatial convolution
        :param window_size: int, number of historical time steps to consider
        """
        super(TrafficModel, self).__init__()

        self.layers = nn.ModuleList()

        # STConv blocks
        for l in range(num_layers):
            self.layers.append(
                STConv(
                    num_nodes,
                    channel_size_list[l][0],
                    channel_size_list[l][1],
                    channel_size_list[l][2],
                    kernel_size,
                    K,
                    normalization,
                    bias
                )
            )

        # output layer
        self.layers.append(
            OutputLayer(
                channel_size_list[-1][-1],
                window_size-2*num_layers*(kernel_size-1),
                num_nodes
            )
        )

        for layer in self.layers:
            layer.to(device)

    def forward(self, x, edge_index, edge_weight):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_weight)
        x = self.layers[-1](x.permute(0, 3, 1, 2))

        return x
