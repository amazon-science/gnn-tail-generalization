import time

import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, SAGEConv
import dgl.function as fn
import torch.nn.functional as F
import networkx as nx

from src.models.MLP import MLP
from src.models.inner_product_decoder import InnerProductDecoder
from src.utils import loss_function
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, label_ranking_average_precision_score
import scipy.sparse as sp
from IPython import embed
from torch.nn.utils import clip_grad_norm_
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def extract_nodeflow(nf):
    node_set = set()
    for i in range(nf.num_layers):
        node_set.update(nf.layer_parent_nid(i).tolist())
    node_idx = list(node_set)
    # embed()
    return node_idx 
    #return nf.layer_parent_nid(0), nf.layer_parent_nid(1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class VGAE(nn.Module):
    def __init__(self, g, in_feats, hidden_dim1, hidden_dim2, dropout, pretrain = None):
        super(VGAE, self).__init__()
        self.g = g
        self.in_feats = in_feats
        self.pretrain = pretrain
        self.hidden_dim1 = hidden_dim1

        self.gc1 = SAGEConv(in_feats, hidden_dim1, aggregator_type='gcn', activation=F.relu)
        self.gc2 = SAGEConv(hidden_dim1, hidden_dim2, aggregator_type='gcn', activation=None)
        self.gc3 = SAGEConv(hidden_dim1, hidden_dim2, aggregator_type='gcn', activation=None)
        #self.gc1 = GraphConv(in_feats, hidden_dim1, bias=False, activation=F.relu)
        #self.gc2 = GraphConv(hidden_dim1, hidden_dim2, bias=False, activation=lambda x: x)
        #self.gc3 = GraphConv(hidden_dim1, hidden_dim2, bias=False, activation=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

        if pretrain is not None:
            #pass
            self.load_state_dict(torch.load(pretrain))
            print("Loaded pre-train model")

    def prepare(self):
        if self.pretrain is not None:
            self.feat_encoder = MLP(self.in_feats, self.hidden_dim1, self.hidden_dim1)
            #self.feat_encoder = None
        else:
            #self.feat_encoder = MLP(self.in_feats, self.hidden_dim1, self.hidden_dim1)
            self.feat_encoder = None

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def encode(self, feature):
        if self.feat_encoder is not None:
            feature = self.feat_encoder(feature)
        hidden1 = self.gc1(self.g, feature)
        return self.gc2(self.g, hidden1), self.gc3(self.g, hidden1)

    

    def forward(self, features, relative_node_idx=None):
        mu, logvar = self.encode(features)
        # embed()
        if relative_node_idx is not None:
            z = self.reparameterize(mu[relative_node_idx], logvar[relative_node_idx])
            return self.dc(z), mu, logvar
        else:
            return None, mu, None


    def train_model(self):
        self.train()
        cur_loss = []
        for idx, nf in enumerate(self.train_sampler):
            t = time.time()
            decoder_node_id = extract_nodeflow(nf)
            self.optimizer.zero_grad()
            recovered, mu, logvar = self.forward(self.features, decoder_node_id)
            sub_sampled_adj = self.adj_train[decoder_node_id, :][:, decoder_node_id]
            adj_label = sub_sampled_adj + sp.eye(sub_sampled_adj.shape[0])
            adj_label = torch.FloatTensor(adj_label.toarray()).reshape(-1).to(device)
            
            pos_weight = torch.FloatTensor( [float(sub_sampled_adj.shape[0] * sub_sampled_adj.shape[0] - sub_sampled_adj.sum()) / sub_sampled_adj.sum()]).to(device)
            norm = sub_sampled_adj.shape[0] * sub_sampled_adj.shape[0] / float( (sub_sampled_adj.shape[0] * sub_sampled_adj.shape[0] - sub_sampled_adj.sum()) * 2)
            
            loss = loss_function(recovered, adj_label, mu=mu, logvar=logvar, n_nodes=self.features.shape[0], norm=norm,
                                 pos_weight=pos_weight)
            loss.backward()
            # clip_grad_norm_(self.parameters(), 1.0)
            cur_loss.append(loss.item())
            self.optimizer.step()
            # embed()

        return np.mean(cur_loss)


    def test_model(self, test_edges, test_edges_false, feature_only = False):
        with torch.no_grad():
            self.eval()
            # feature verify
            if feature_only:
                output_emb = self.features.cpu()
            else:
                _, mu, __ = self.forward(self.features)
                output_emb = mu.cpu().detach()
            # embed()
            pos_pred = torch.sigmoid((output_emb[test_edges[:,0]] * output_emb[test_edges[:,1]]).sum(dim=1)).numpy()
            neg_pred = torch.sigmoid((output_emb[test_edges_false[:,0]] * output_emb[test_edges_false[:,1]]).sum(dim=1)).numpy()
            #offset = 0
            pred = np.concatenate( (pos_pred, neg_pred))

            roc_score = roc_auc_score(
                np.concatenate((np.ones(test_edges.shape[0]), np.zeros(test_edges_false.shape[0]))), pred)
            ap_score = average_precision_score(
                np.concatenate((np.ones(test_edges.shape[0]), np.zeros(test_edges_false.shape[0]))), pred)
            mrr_score = label_ranking_average_precision_score(
                np.concatenate((np.ones( (test_edges.shape[0],1) ), np.zeros( (test_edges_false.shape[0],1) )), axis = 0), pred[..., np.newaxis])
        return roc_score, ap_score, mrr_score
    
    def generate_subgraph(self, nf):
        nll = 0.0
        message_func = fn.v_dot_u('z', 'z', 'm')
        for i in range(nf.num_blocks):
            nf.layers[i].data['z'] = self.z[nf.layer_parent_nid(i)]
            nf.layers[i+1].data['z'] = self.z[nf.layer_parent_nid(i+1)]
            #nf.layers[i].data['nz'] = nz[nf.layer_parent_nid(i)]
            #if x is not None:
            #    nf.layers[i].data['x'] = x[nf.layer_parent_nid(i)]
            # embed()
            nf.block_compute(i, lambda edges: {'m': ((edges.src['z'] - edges.dst['z'])**2).sum(dim=1) }, 
            lambda node :{'nll': - 1 / (1 + (node.mailbox['m']-1).exp() ).log().sum(dim=1)})
            nll += nf.layers[i+1].data['nll'].sum().item()
            # nf.block_compute(i, fn.v_dot_u('z', 'z', 'm'), reduce)
        return nll

    
    def output_nll(self):
        with torch.no_grad():
            self.eval()
            nll = 0.0
            _, mu, __ = self.forward(self.features)
            self.z = mu.detach()
            # output_emb = mu.cpu().detach()
            for idx, nf in enumerate(self.test_sampler):
                nll += self.generate_subgraph(nf)

            return nll

        