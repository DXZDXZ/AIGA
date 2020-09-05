import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax


class GatConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels,
                 negative_slope=0.2, dropout=0):
        super(GatConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.fc = nn.Linear(in_channels, out_channels)
        self.attn = nn.Linear(2 * out_channels + edge_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = self.fc(x)
        try:
            return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)
        except AssertionError as e:
            print(e)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        x = torch.cat([x_i, x_j, edge_attr], dim=-1)
        alpha = self.attn(x)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha

    def update(self, aggr_out):
        return aggr_out


class Encoder(nn.Module):
    def __init__(self, input_node_dim, hidden_node_dim, input_edge_dim, hidden_edge_dim, conv_layers=3):
        super(Encoder, self).__init__()
        self.hidden_node_dim = hidden_node_dim
        self.fc_node = nn.Linear(input_node_dim, hidden_node_dim)
        self.fc_edge = nn.Linear(input_edge_dim, hidden_edge_dim)
        self.bn = nn.BatchNorm1d(hidden_node_dim)
        self.convs = nn.ModuleList(
            [GatConv(hidden_node_dim, hidden_node_dim, hidden_edge_dim) for i in range(conv_layers)])

    def forward(self, data):
        batch_size = data.num_graphs
        edge_attr = data.edge_attr
        x = self.fc_node(data.x)
        edge_attr = self.fc_edge(data.edge_attr)
        x = self.bn(x)

        for conv in self.convs:
            x = conv(x, data.edge_index, edge_attr)

        x = x.reshape((batch_size, -1, self.hidden_node_dim))

        return x
