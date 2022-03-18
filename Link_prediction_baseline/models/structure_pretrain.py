import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv, GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from torch.nn.parameter import Parameter

from models.structure_sageconv import StructSAGEConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h, h_per_layer=False):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        # only need node embedding
        if h_per_layer:
            return hidden_rep
        else:
            return h


class NeuralTensorLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralTensorLayer, self).__init__()
        self.W = nn.Bilinear(input_dim, input_dim, output_dim, bias=False)
        self.V_b = nn.Linear(2*input_dim, output_dim, bias=True)

    def forward(self, input1, input2):
        tmp1 = self.W(input1, input2)
        tmp2 = self.V_b(torch.cat((input1, input2), dim=1))
        return torch.tanh(tmp1+tmp2)


class StructGCN(nn.Module):
    def __init__(self, in_feats, n_hidden, dropout):
        super(StructGCN, self).__init__()
        self.feat_encoder_layers = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.feat_encoder_layers.append(nn.Linear(in_feats, n_hidden, bias=False))
        self.layers.append(StructSAGEConv(n_hidden, n_hidden, aggregator_type='gcn', activation=None, norm=nn.BatchNorm1d(n_hidden)))
        self.feat_encoder_layers.append(nn.Linear(in_feats, n_hidden, bias=False))
        self.layers.append(StructSAGEConv(n_hidden, n_hidden, aggregator_type='gcn', activation=None, norm=nn.BatchNorm1d(n_hidden)))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features, h_per_layer=False):
        h = features
        hidden_rep = [h]
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = F.relu(self.feat_encoder_layers[i](h))
            h = F.relu(layer(g, h))
            hidden_rep.append(h)

        return hidden_rep if h_per_layer else h


class GCN(nn.Module):
    def __init__(self, n_layers, in_feats, n_hidden, output_dim, activation, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type='gcn', activation=activation))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type='gcn', activation=activation))
        self.layers.append(SAGEConv(n_hidden, output_dim, aggregator_type='gcn', activation=None))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features, h_per_layer=False):
        h = features
        hidden_rep = [h]
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            hidden_rep.append(h)
        if h_per_layer:
            return h
        else:
            return hidden_rep


class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, dropout, type):
        super(Encoder, self).__init__()
        if type == 'gcn':
            self.conv = GCN(n_layers+1, in_feats, n_hidden, n_hidden, F.relu, dropout)
        elif type == 'gin':
            # TODO: check GIN layer number, use which GIN, forward args
            self.conv = GIN(n_layers+1, 1, in_feats, n_hidden, n_hidden, dropout, True, 'sum', 'sum')

        # self.conv = StructGCN(g, in_feats, n_hidden, dropout)
    def forward(self, g, features, h_per_layer=False):
        embedding = self.conv(g, features, h_per_layer=h_per_layer)
        return embedding


class NTN_Decoder(nn.Module):
    def __init__(self, input_dim, class_num, output_activation=None):
        super(NTN_Decoder, self).__init__()
        self.link_decoder = NeuralTensorLayer(input_dim, output_dim=4)
        self.MLP = nn.Linear(4, class_num)
        self.output_activation = output_activation

    def forward(self, embedding1, embedding2):
        output = self.link_decoder(embedding1, embedding2)
        if self.output_activation == None:
            return self.MLP(output)
        else:
            return F.softmax(self.MLP(output)) # for classification


class MLP_Decoder(nn.Module):
    def __init__(self, input_dim=512):
        super(MLP_Decoder, self).__init__()
        hidden = input_dim//2
        self.MLP1 = nn.Linear(input_dim, hidden)
        self.MLP2 = nn.Linear(hidden, 1)

    def forward(self, embedding):
        output = F.relu(self.MLP1(embedding))
        return self.MLP2(output)


class Struct_Feat_Pretrain(nn.Module):
    def __init__(self, args, g, masked_g, in_feats, n_hidden, n_layers, n_centralities, dropout):
        super(Struct_Feat_Pretrain, self).__init__()
        self.args = args
        self.g = g
        self.masked_g = masked_g
        self.num_layer = n_layers
        self.feature_mapping = nn.Linear(in_feats, n_hidden) # feature mapping
        self.encoder = Encoder(n_hidden, n_hidden, n_layers, dropout, args.encoder_type)

        self.link_psi = Parameter(torch.Tensor(n_layers+2))
        self.link_alpha = Parameter(torch.Tensor(1))
        self.link_deocder = NTN_Decoder(n_hidden, 1)

        self.centrality_psi = Parameter(torch.Tensor(n_layers+2))
        self.centrality_alpha = Parameter(torch.Tensor(1))
        self.centrality_decoder_list = nn.ModuleList()
        for i in range(n_centralities):
            self.centrality_decoder_list.append(MLP_Decoder(input_dim=n_hidden))

        # set parameter
        torch.nn.init.uniform(self.link_psi)
        self.link_alpha.data.fill_(1)
        torch.nn.init.uniform(self.centrality_psi)
        self.centrality_alpha.data.fill_(1)

        # self.cluster_psi = Parameter(torch.Tensor(n_layers+1))
        # self.cluster_alpha = Parameter(torch.Tensor(1))
        # self.cluster_deocder = NTN_Decoder(input_dim=n_hidden, class_num=cluster_num)
        # self.cluster_loss = nn.CrossEntropyLoss(cluster_weight)


    def forward(self, g, features):
        # embedding = torch.tanh(self.feature_mapping(features))
        embedding = features
        embedding = torch.stack(self.encoder(g, embedding, h_per_layer=True))
        return embedding


    def train_model(self, features):
        loss_list = []
        for link_batch, centrality_batch in zip(self.link_reconstruct_loader, self.centrality_score_loader):
            self.optimizer.zero_grad()
            link_edges = link_batch[0]
            link_labels = link_batch[1].unsqueeze(-1)
            link_embedding = self.forward(self.masked_g, features)
            link_embedding = (F.softmax(self.link_psi).unsqueeze(-1).unsqueeze(-1) * link_embedding).sum(dim=0) * self.link_alpha
            recovered = self.link_deocder(link_embedding[link_edges[:, 0]], link_embedding[link_edges[:, 1]])
            # reconstruct_pos_weight = torch.FloatTensor([float(link_labels.shape[0] - link_labels.sum()) / link_labels.sum()]).to(device)
            # reconstruct_norm = link_labels.shape[0] / float((link_labels.shape[0] - link_labels.sum()))
            # reconstruct_loss = reconstruct_norm * F.binary_cross_entropy_with_logits(recovered, link_labels, pos_weight=reconstruct_pos_weight)
            reconstruct_loss = F.binary_cross_entropy_with_logits(recovered, link_labels)

            centrality_edges = centrality_batch[0].to(device)
            centrality_labels = centrality_batch[1].to(device)

            centrality_embedding = self.forward(self.g, features)
            centrality_embedding = (F.softmax(self.centrality_psi).unsqueeze(-1).unsqueeze(-1) * centrality_embedding).sum(dim=0) * self.centrality_alpha

            total_centrality_loss = 0
            for i in range(centrality_labels.shape[1]):
                centrality_label = centrality_labels[:, i].unsqueeze(-1)
                centrality_score = self.centrality_decoder_list[i](centrality_embedding)
                pred = centrality_score[centrality_edges[:, 0]] - centrality_score[centrality_edges[:, 1]]
                pos_weight = torch.FloatTensor([float(centrality_label.shape[0] - centrality_label.sum()) / centrality_label.sum()]).to(device)
                norm = centrality_label.shape[0] / float((centrality_label.shape[0] - centrality_label.sum()))
                centrality_loss = norm * F.binary_cross_entropy_with_logits(pred, centrality_label, pos_weight=pos_weight)

                if (centrality_loss != centrality_loss).any():
                    print("1")
                total_centrality_loss += centrality_loss
            loss = total_centrality_loss + reconstruct_loss
            loss.backward()
            if (loss != loss).any():
                print("2")

            self.optimizer.step()
            loss_list.append(loss.item())
        return sum(loss_list) / len(loss_list)

    # def train_model(self, features):
    #     link_loss = self.link_reconstruction(features)
    #     centrality_loss = self.centrality_score_ranking(features)
    #     # cluster_loss = self.cluster_preserving(embedding)
    #     loss = link_loss + centrality_loss # + cluster_loss
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss


    # task 1, return loss
    # def link_reconstruction(self, features):
    #     embedding = self.forward(self.masked_g, features)
    #     embedding = (F.softmax(self.link_psi).unsqueeze(-1).unsqueeze(-1) * embedding).sum(dim=0) * self.centrality_alpha
    #     for batch in self.link_reconstruct_loader:
    #         edges = batch[0]
    #         labels = batch[1].unsqueeze(-1)
    #         recovered = self.link_deocder(embedding[edges[:, 0]], embedding[edges[:, 1]])
    #         pos_weight = torch.FloatTensor([float(labels.shape[0] - labels.sum()) / labels.sum()]).to(device)
    #         norm = labels.shape[0] / float((labels.shape[0] - labels.sum()) * 2)
    #         loss = norm * F.binary_cross_entropy_with_logits(recovered, labels, pos_weight=pos_weight)
    #         return loss
    #
    #
    # # task 2, return loss
    # def centrality_score_ranking(self, features):
    #     embedding = self.forward(self.g, features)
    #     embedding = (F.softmax(self.centrality_psi).unsqueeze(-1).unsqueeze(-1) * embedding).sum(dim=0) * self.centrality_alpha
    #     for batch in self.centrality_score_loader:
    #         edges = batch[0].to(device)
    #         labels = batch[1].to(device)[:, 0]
    #         if len(labels.shape) == 1:
    #             labels = labels.unsqueeze(-1)
    #         score0 = self.centrality_decoder(embedding[edges[:, 0]])
    #         score1 = self.centrality_decoder(embedding[edges[:, 1]])
    #         pos_weight = torch.FloatTensor([float(labels.shape[0] - labels.sum()) / labels.sum()]).to(device)
    #         norm = labels.shape[0] / float((labels.shape[0] - labels.sum()) * 2)
    #         pred = score0 - score1
    #         loss = norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_weight)
    #         return loss


    # # task 3, return loss
    # def cluster_preserving(self, embedding):
    #     embedding = self.cluster_alpha * (F.softmax(self.cluster_psi) * embedding).sum(dim=0)
    #     for batch in self.cluster_loader:
    #         nodes = batch[0].to(device)
    #         labels = batch[1].to(device)
    #         pred = self.cluster_deocder(embedding[nodes])
    #         loss = self.cluster_loss(pred, labels)
    #         return loss



if __name__ == "__main__":
    m = nn.Bilinear(30, 30, 4, bias=False)
    input1 = torch.randn(3, 30)
    input2 = torch.randn(3, 30)
    output = m(input1, input2)

    print(output.size())

    print(m.weight.shape)
    cache = []
    for i in range(4):
        tmp = torch.mm(input1, m.weight[i])
        cache.append((tmp * input2).sum(dim=1, keepdims=True))

    print(torch.cat(cache, dim=1).shape)
    assert (torch.cat(cache, dim=1) == output).all()