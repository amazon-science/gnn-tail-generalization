import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv, GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from torch import optim
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))
from dgl import backend as dgl_F

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, g, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        """model parameters setting
        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)
        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        # only need node embedding
        return h


class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, dropout):
        super(Encoder, self).__init__()
        self.conv = GIN(g, n_layers, 1, in_feats, n_hidden, n_hidden, dropout, True, 'sum', 'sum')

    def forward(self, g, features):
        features = self.conv(g, features)
        return features



class contextpred_GIN(nn.Module):
    def __init__(self, args, g, in_feats, n_hidden, n_layers, dropout):
        super(contextpred_GIN, self).__init__()
        self.args = args
        self.g = g
        self.num_layer = n_layers
        assert n_layers > args.l1
        assert args.l2 > args.l1
        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, dropout)
        self.model_context =  Encoder(g, in_feats, n_hidden, int(args.l2 - args.l1), dropout)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer_substruct = optim.Adam(self.encoder.parameters(), lr=args.central_encoder_lr)
        self.optimizer_context = optim.Adam(self.model_context.parameters(), lr=args.context_encoder_lr)


    def forward(self, features):
        substruct_rep = self.encoder(self.g, features)
        return substruct_rep


    def cycle_index(self, num, shift):
        arr = torch.arange(num) + shift
        arr[-shift:] = torch.arange(shift)
        return arr


    def train_model(self, features):
        self.optimizer_substruct.zero_grad()
        self.optimizer_context.zero_grad()

        # self.context_Gs[idx], self.overlap_nodes[idx], self.central_id[idx]
        # TODO: check cids and the format of bg_overlap_nodes
        for bg, bg_features, bg_overlap_nodes, batch_num_overlaped_nodes, cids in self.contextgraph_loader:
            substruct_rep = self.forward(features)
            overlapped_node_rep = self.model_context(bg, bg_features)

        # TODO: agg by bg_overlap_nodes
        context_rep = overlapped_node_rep[bg_overlap_nodes]
        n_graphs = bg.batch_size
        batch_num_objs = batch_num_overlaped_nodes
        seg_id = dgl_F.zerocopy_from_numpy(np.arange(n_graphs, dtype='int64').repeat(batch_num_objs))
        seg_id = dgl_F.copy_to(seg_id, dgl_F.context(context_rep))
        context_rep = dgl_F.unsorted_1d_segment_mean(context_rep, seg_id, n_graphs, 0)

        assert context_rep.shape == substruct_rep.shape

        neg_context_rep = torch.cat(
            [context_rep[self.cycle_index(len(context_rep), i + 1)] for i in range(self.args.neg_samples)], dim=0)

        pred_pos = torch.sum(substruct_rep * context_rep, dim=1)
        pred_neg = torch.sum(substruct_rep.repeat((self.args.neg_samples, 1)) * neg_context_rep, dim=1)

        loss_pos = self.criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
        loss_neg = self.criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())

        loss = loss_pos + self.args.neg_samples * loss_neg
        loss.backward()
        self.optimizer_substruct.step()
        self.optimizer_context.step()

        return loss