import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv, GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

from src.models.MLP import MLP as feat_MLP


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
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, linear_or_not=True):
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
        self.linear_or_not = linear_or_not  # default is linear model
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
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
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

        # score_over_layer = 0
        #
        # # perform pooling over all nodes in each graph in every layer
        # for i, h in enumerate(hidden_rep):
        #     pooled_h = self.pool(g, h)
        #     score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
        #
        # return score_over_layer


class GCN(nn.Module):
    def __init__(self, n_layers, in_feats, n_hidden, output_dim, activation, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type='gcn', activation=activation))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type='gcn', activation=activation))
        self.layers.append(SAGEConv(n_hidden, output_dim, aggregator_type='gcn', activation=None))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, dropout, type):
        super(Encoder, self).__init__()
        if type == 'gcn':
            self.conv = GCN(n_layers+1, in_feats, n_hidden, n_hidden, F.relu, dropout)
        elif type == 'gin':
            self.conv = GIN(n_layers+1, 1, in_feats, n_hidden, n_hidden, dropout, True, 'sum', 'sum')

    def forward(self, g, features):
        features = self.conv(g, features)
        return features


class masking_GIN(nn.Module):
    def __init__(self, args, in_feats, n_hidden, n_layers, n_degree, dropout):
        super(masking_GIN, self).__init__()
        self.args = args
        self.num_layer = n_layers
        self.in_feats = in_feats
        self.hidden = n_hidden
        self.encoder = Encoder(in_feats, n_hidden, n_layers, dropout, args.encoder_type)
        if args.pretrain is not None:
            self.degree_classifier = torch.nn.Linear(n_hidden, 103)
        else:
            self.degree_classifier = torch.nn.Linear(n_hidden, n_degree)
        self.feat_encoder = None
        # self.prepare()

    def prepare(self):
        self.feat_encoder = feat_MLP(self.in_feats, self.hidden, self.in_feats) # MLP(1, self.in_feats, self.hidden, self.hidden)


    def forward(self, g, features, test_mask=None):
        if self.feat_encoder is not None:
            embedding = F.relu(self.feat_encoder(features))
        else:
            embedding = features
        embedding = self.encoder(g, embedding)
        if test_mask is not None:
            pred = torch.log_softmax(self.degree_classifier(embedding[test_mask]), dim=1)
            embedding = embedding[test_mask]
        else:
            pred = torch.log_softmax(self.degree_classifier(embedding), dim=1)
        return pred, embedding


    def train_model(self, g, features, test_mask=None, train_label=None):
        self.train()
        self.optimizer.zero_grad()
        pred, embedding = self.forward(g, features, test_mask)
        if train_label is not None:
            loss = F.nll_loss(pred, train_label)
        else:
            loss = F.nll_loss(pred, self.degree)
        loss.backward()
        self.optimizer.step()
        return loss.item()
