import numpy
from pypapi import events, papi_high as high

high.start_counters([events.PAPI_FP_OPS, ])

for n in [10, 30, 100, 300, 1000, 10000, 20000]:
    aa = numpy.mgrid[0:n:1, 0:n:1][0]
    high.start_counters([events.PAPI_FP_OPS, ])
    a = numpy.fft.fft(aa)
    x = high.stop_counters()
    print(n, x)
    
    
    
raise

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
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        # only need node embedding
        return h

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
        
        #if in_feats != n_hidden:
        #    self.feat_encoder = MLP(in_feats, n_hidden, n_hidden)
        #else:
        #    self.feat_encoder = None
        
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

class GRUReduce(nn.Module):
    def __init__(self, in_feats, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size

        self.W_z = nn.Linear(in_feats, hidden_size)
        self.W_h = nn.Linear(in_feats + hidden_size, hidden_size)
        #self.W_1 = nn.Linear(in_feats, hidden_size)


    def forward(self, node):
        if node.mailbox:
            '''
            s = node.mailbox['m'].sum(dim=1)
            rm = node.mailbox['rm'].sum(dim=1)
            z = torch.sigmoid(self.W_z(torch.cat([node.data['x'], s], 1)))
            m = torch.tanh(self.W_h(torch.cat([node.data['x'], rm], 1)))
            m = (1 - z) * s + z * m
            '''
            m = F.relu(self.W_z(node.data['x']) + node.mailbox['m'].mean(dim=1))
            root = node.mailbox['root'].mean(dim=1)
            #embed()
            #nroot = node.mailbox['nroot'].mean(dim=1)
            #h = torch.cat([node.mailbox['root'], node.data['x'].repeat(1, node.mailbox['root'].shape[1], 1),
            # s.repeat(1, node.mailbox['root'].shape[1], 1)], dim=1)

            #pred = self.U_s(F.relu(self.linear(h)))

            return {'m': m, 'root': root} #, 'out': pred}
        else:
            '''
            z = torch.sigmoid(self.W_z(torch.cat([node.data['x'], torch.zeros_like(node.data['m'])], 1)))
            m = torch.tanh(self.W_h(torch.cat([node.data['x'], torch.zeros_like(node.data['m'])], 1)))
            m = z * m
            '''
            m = self.W_z(node.data['x'])
            #m =torch.cat([node.data['x'], torch.zeros_like(node.data['m'])], 1)
            return {'m': m, 'root':node.data['emb']}

class MsgLayer(nn.Module):
    def __init__(self, in_feats, hidden_size):
        super(MsgLayer, self).__init__()

        self.W_r = nn.Linear(in_feats, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, edges):

        # z = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        '''
        r_1 = self.W_r(edges.dst['x'])
        r_2 = self.U_r(edges.src['m'])
        r = torch.sigmoid(r_1 + r_2)
        return {'m': edges.src['m'], 'rm': r * edges.src['m'], 'root': edges.src['root']}
        '''
        return {'m': edges.src['m'], 'root': edges.src['root']}

class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.g = g
        #self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)
        self.conv = GIN(g, n_layers + 1, 1, in_feats, n_hidden, n_hidden, dropout, True, 'sum', 'sum')
    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features

class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class DGI(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(DGI, self).__init__()
        self.g = g
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout)
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()
    
    def reset_parameters(self):
        self.encoder = Encoder(self.g, self.in_feats, self.n_hidden, self.n_layers, self.activation, self.dropout)
        self.discriminator = Discriminator(self.n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features):
        # features = self.g.ndata['features']
        positive = self.encoder(features, corrupt=False)
        negative = self.encoder(features, corrupt=True)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2


class SubGDiscriminator(nn.Module):
    def __init__(self, g, in_feats, n_hidden):
        super(SubGDiscriminator, self).__init__()
        self.g = g
        # in_feats = in_feats
        self.dc_layers = nn.ModuleList([
            MsgLayer(in_feats, n_hidden),
            GRUReduce(in_feats, n_hidden),
            # DecodeLayer(hidden_dim2 * 2, hidden_dim2)
        ])
        self.linear = nn.Linear(n_hidden + n_hidden, n_hidden, bias = True)
        self.U_s = nn.Linear(n_hidden, 1)
        self.in_feats = in_feats
        #self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        #self.reset_parameters()

    def edge_output(self, edges):
        
        #return {'h': torch.cat([edges.src['root'], edges.dst['emb']], dim=1)}

        return {'h': torch.cat([edges.src['root'], edges.dst['x']], dim=1)}

    def forward(self, nf, emb, features):

        #features = torch.matmul(features, torch.matmul(self.weight, summary))
        

        #self.g.edata['score'] = torch.zeros((self.g.number_of_edges())).cuda()
        #sub_graph
        reverse_edges = []
        for i in range(nf.num_blocks):
            # print(i)
            u,v = self.g.find_edges(nf.block_parent_eid(i))
            reverse_edges += self.g.edge_ids(v,u).numpy().tolist()
        small_g = self.g.edge_subgraph( reverse_edges)
        small_g.ndata['emb'] = emb[small_g.ndata['_ID']]
        small_g.ndata['x'] = features[small_g.ndata['_ID']]
        small_g.ndata['m']= torch.zeros_like(emb[small_g.ndata['_ID']])
        #small_g.ndata['m']= torch.zeros_like(emb[small_g.ndata['_ID']])
        #self.g.ndata['x'] = features.detach()
        
        
        edge_embs = []
        for i in range(nf.num_blocks)[::-1]:
            #nf.layers[i].data['z'] = z[nf.layer_parent_nid(i)]
            #nf.layers[i].data['nz'] = nz[nf.layer_parent_nid(i)]
            #u,v = self.g.find_edges(nf.block_parent_eid(i))
            #reverse_edges = self.g.edge_ids(v,u)
            v = small_g.map_to_subgraph_nid(nf.layer_parent_nid(i+1))
            
            if i+1 == nf.num_blocks:
                # print('YES')
                small_g.apply_nodes(self.dc_layers[1], v) 
                #self.g.apply_edges(self.dc_layers[1], v) 
           #  embed()
            
            #embed()
            uid = small_g.out_edges(v, 'eid')
            small_g.apply_edges(self.edge_output, uid)


            small_g.push(v, self.dc_layers[0], self.dc_layers[1])

            #small_g.copy_to_parent()
            #self.g.apply_edges(self.edge_output, reverse_edges)
            #self.g.send_and_recv(reverse_edges, self.dc_layers[0], 
            #    self.dc_layers[1])
            #h_tmp = torch.zeros( (self.g.number_of_edges(), emb.shape[1])).cuda()
            #embed()
           #
            #small_g.copy_to_parent()
            #h = self.g.edata.pop('h')
            
            h = small_g.edata.pop('h')[uid]
           
            edge_embs.append(self.U_s(F.relu(self.linear(h))))
            #edge_embs += h.sum(dim=0)
            #edge_embs.append(self.U_s(F.relu(self.linear(h[reverse_edges]))))
        # embed()
        return edge_embs
        #return features

class SubGI(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, pretrain=None):
        super(SubGI, self).__init__()
        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout)
        self.g = g
        self.subg_disc = SubGDiscriminator(g, in_feats, n_hidden)
        self.loss = nn.BCEWithLogitsLoss()
        if pretrain is not None:
            print("Loaded pre-train model: {}".format(pretrain) )
            #self.load_state_dict(torch.load(project_path + '/' + pretrain))
            self.load_state_dict(torch.load(pretrain))
    def forward(self, features, nf):
        positive, x = self.encoder(features, corrupt=False)
        # negative, x = self.encoder(features, corrupt=True)
        perm = torch.randperm(self.g.number_of_nodes())
        #
        negative = positive[perm]
        #negative = torch.zeros_like(positive)
        # summary = torch.sigmoid(positive.mean(dim=0))
        #self.g.ndata['emb'] = positive
        #self.g.ndata['m'] = torch.zeros_like(positive)
        #perm = torch.randperm(self.g.number_of_nodes())
        #features = features[perm]
        #self.g.ndata['x'] = features.detach()
        positive_batch = self.subg_disc(nf, positive, features)
        #self.g.ndata['emb'] = negative
        # perm = torch.randperm(self.g.number_of_nodes())
        #embed()
        
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

class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, features):
        features = self.fc(features)
        return features
        #return features.squeeze()
        #return torch.log_softmax(features, dim=-1)

class MultiClassifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(MultiClassifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, features):
        features = self.fc(features)
        #return features.squeeze()
        return torch.log_softmax(features, dim=-1)
