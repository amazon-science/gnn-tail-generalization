import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv, GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from IPython import embed
import numpy as np

try:
    from models.utils import get_positive_expectation, get_negative_expectation
except ModuleNotFoundError:
    from baselines.models.utils import get_positive_expectation, get_negative_expectation


class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
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
        self.g = g

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

    def forward(self, h):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers):
            h = self.ginlayers[i](self.g, h)
            # print('batch norm')
            # 
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        # only need node embedding
        return h

class NodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(NodeUpdate, self).__init__()
        self.linear =nn.Linear(in_feats, out_feats, bias=True)
        self.activation = activation

    def forward(self, node):
        h = node.data['h']
        h = self.linear(h)
        if self.activation:
            h = self.activation(h)
        return {'activation': h}

class EdgeUpdate(nn.Module):
    def __init__(self, shuffle = False):
        super(EdgeUpdate, self).__init__()
        self.shuffle = shuffle
    def forward(self, edges):
        if self.shuffle:
            return {'m': edges.src['h'][torch.randperm(edges.src['h'].shape[0])]}
        else:
            
            return {'m': edges.src['h']}




class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        
        if in_feats != n_hidden:
            self.feat_encoder = MLP(in_feats, n_hidden, n_hidden)
        else:
            self.feat_encoder = None
        
        self.feat_encoder = None
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type='gcn', activation=activation))
        #self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type='gcn', activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            #self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
            self.layers.append(SAGEConv(n_hidden, n_hidden,  aggregator_type='gcn',activation=activation))
        # output layer
        #self.layers.append(GraphConv(n_hidden, n_classes))
        self.layers.append(SAGEConv(n_hidden, n_classes,  aggregator_type='gcn',activation=None))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        #if self.feat_encoder is not None:
        #    features = self.feat_encoder(features)
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)

        return h 

class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        
        self.g = g
        
        #self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)
        self.conv = GIN(g, n_layers + 1, 1, in_feats, n_hidden, n_hidden, dropout, True, 'sum', 'sum')
        # embed()

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features


class GNNDiscLayer(nn.Module):
    def __init__(self, in_feats, n_hidden):
        super(GNNDiscLayer, self).__init__()
        self.fc = nn.Linear(in_feats, n_hidden)
        self.layer_1 = True
        # self.self_fc = nn.Linear(in_feats, n_hidden)

    
    #def edge_update(self, edges):
    #    pass
    def reduce(self, nodes):
        return {'m': F.relu(self.fc(nodes.data['x']) + nodes.mailbox['m'].mean(dim=1) ), 'root': nodes.mailbox['root'].mean(dim=1)}

    def msg(self, edges):
        if self.layer_1:
            return {'m': self.fc(edges.src['x']), 'root': edges.src['root']}
        else:
            # embed()
            return {'m': self.fc(edges.src['m']), 'root': edges.src['root']}
    
    def edges(self, edges):
        # embed()
        return {'output':torch.cat([edges.src['root'], edges.src['m'], edges.dst['x']], dim=1)}

    def forward(self, g, v, edges, depth=1):
        #self.layer_nodes = v
        #self.g = g
        if depth == 1:
            self.layer_1 = True
        else:
            self.layer_1 = False
        g.apply_edges(self.edges, edges)
        # g = g.local_var()
        #embed()
        g.push(v, self.msg, self.reduce)
        
        return g.edata.pop('output')[edges]

class SubGDiscriminator(nn.Module):
    def __init__(self, g, in_feats, n_hidden, model_id, n_layers = 2):
        super(SubGDiscriminator, self).__init__()
        self.g = g
        # in_feats = in_feats
        self.dc_layers = nn.ModuleList()
        for i in range(n_layers):
            if model_id > 0:
                self.dc_layers.append(GNNDiscLayer(in_feats, n_hidden))
        
        self.linear = nn.Linear(in_feats + 2 * n_hidden, n_hidden, bias = True)
        self.in_feats = in_feats
        self.model_id = model_id
        self.U_s = nn.Linear(n_hidden, 1)
        #self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        #self.reset_parameters()

    def edge_output(self, edges):
        
        # return {'h': torch.cat([edges.src['root'], edges.dst['emb']], dim=1)}
        if self.model_id == 1:
            return {'h': torch.cat([edges.src['root'], edges.dst['x']], dim=1)}
        elif self.model_id in [2,3]:
            return {'h': torch.cat([edges.src['root'], edges.src['m'], edges.dst['x']], dim=1)}

    def find_common(self, layer_nid, nf):
        reverse_nodes = set()
        for i in range(nf.num_blocks):
            u, v = self.g.find_edges(nf.block_parent_eid(i))
            reverse_nodes.update(u.tolist())
            reverse_nodes.update(v.tolist())
        layer_nid = set(layer_nid.tolist())
        return torch.tensor(list(layer_nid.intersection(reverse_nodes)))

    def forward(self, nf, emb, features):
        
        if self.model_id == 0:
            return self.U_s(F.relu(emb))
        else:
            reverse_edges = []
            for i in range(nf.num_blocks):
                #print(i)
                #embed()
                u,v = self.g.find_edges(nf.block_parent_eid(i))
                reverse_edges += self.g.edge_ids(v,u).numpy().tolist()
            small_g = self.g.edge_subgraph( reverse_edges)
            small_g.ndata['root'] = emb[small_g.ndata['_ID']]
            small_g.ndata['x'] = features[small_g.ndata['_ID']]
            small_g.ndata['m']= torch.zeros_like(emb[small_g.ndata['_ID']])
            #small_g.ndata['m']= torch.zeros_like(emb[small_g.ndata['_ID']])
            #self.g.ndata['x'] = features.detach()
        
            # embed()
            edge_embs = []
            for i in range(nf.num_blocks)[::-1]:
                #nf.layers[i].data['z'] = z[nf.layer_parent_nid(i)]
                #nf.layers[i].data['nz'] = nz[nf.layer_parent_nid(i)]
                #u,v = self.g.find_edges(nf.block_parent_eid(i))
                #reverse_edges = self.g.edge_ids(v,u)
                v = small_g.map_to_subgraph_nid(nf.layer_parent_nid(i+1))
                
                #
                    # print('YES')
                #    small_g.apply_nodes(self.dc_layers[1], v) 
                    #self.g.apply_edges(self.dc_layers[1], v) 
            #  embed()
                
                #embed()
                uid = small_g.out_edges(v, 'eid')
                #small_g.apply_edges(self.edge_output, uid)
                # embed()
                if i+1 == nf.num_blocks:
                    h = self.dc_layers[0](small_g, v, uid, 1)
                #embed()
                else:
                    h = self.dc_layers[0](small_g, v, uid, 2)

                edge_embs.append(self.U_s(F.relu(self.linear(h))))
            return edge_embs

class SubGI(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, model_id=0, pretrain=None):
        super(SubGI, self).__init__()
        if model_id > 0:
            self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout)
        else:
            self.encoder = GCNSampling(in_feats, n_hidden, n_hidden, n_layers, activation, dropout)
        self.g = g

        self.subg_disc = SubGDiscriminator(g, in_feats, n_hidden, model_id)
        self.loss = nn.BCEWithLogitsLoss()
        self.model_id = model_id
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        if pretrain is not None:
            print("Loaded pre-train model: {}".format(pretrain) )
            #self.load_state_dict(torch.load(project_path + '/' + pretrain))
            self.load_state_dict(torch.load(pretrain))
    
    def reset_parameters(self):
        self.encoder = Encoder(self.g, self.in_feats, self.n_hidden, self.n_layers, self.activation, self.dropout)
        self.encoder.conv.g = self.g
        self.subg_disc = SubGDiscriminator(self.g, self.in_feats, self.n_hidden, self.model_id)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features, nf):
        
        if self.model_id > 0:
            positive = self.encoder(features, corrupt=False)
            perm = torch.randperm(self.g.number_of_nodes())
            # embed()
            negative = positive[perm]
            #negative = torch.zeros_like(positive)
            # summary = torch.sigmoid(positive.mean(dim=0))
            #self.g.ndata['emb'] = positive
            #self.g.ndata['m'] = torch.zeros_like(positive)
            #perm = torch.randperm(self.g.number_of_nodes())
            #features = features[perm]
            #self.g.ndata['x'] = features.detach()
        else:
            nf.copy_from_parent()
            #print("here")
            positive = self.encoder(nf, False)
            #embed()
            #perm = torch.randperm(self.g.number_of_nodes())
            #negative = positive[perm]
            negative = self.encoder(nf, True)
        
        positive_batch = self.subg_disc(nf, positive, features)

        
        #negative = positive
        negative_batch = self.subg_disc(nf, negative, features)
        #embed()
        E_pos, E_neg, l = 0.0, 0.0, 0.0
        pos_num, neg_num = 0, 0
        for positive_edge, negative_edge in zip(positive_batch, negative_batch):
            # embed()
            E_pos += get_positive_expectation(positive_edge, 'JSD', average=False).sum()
            pos_num += positive_edge.shape[0]
            #E_pos = E_pos / positive_edge.shape[0]
            E_neg += get_negative_expectation(negative_edge, 'JSD', average=False).sum()
            neg_num += negative_edge.shape[0]
            #E_neg = E_neg / negative_edge.shape[0]
            #l1 = self.loss(positive_edge, torch.ones_like(positive_edge))
            #l2 = self.loss(negative_edge, torch.zeros_like(negative_edge))
            l += E_neg - E_pos
            #l += (l1 + l2)
        # print("1 loop finished")
        return E_neg / neg_num - E_pos / pos_num
        #return l
    
    def train_model(self):
        self.train()
        cur_loss = []
        
        for nf in self.train_sampler:
            # neighbor_ids, target_ids = extract_nodeflow(nf)
            #print(idx)
            self.optimizer.zero_grad()
            l = self.forward(self.features, nf)
            l.backward()
            cur_loss.append(l.item())
            # continue
            self.optimizer.step()
        #print("Train NLL:{}".format(np.sum(cur_nll)))
        # embed()
        return np.mean(cur_loss)