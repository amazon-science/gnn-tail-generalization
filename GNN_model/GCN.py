# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import math
import torch as th
import torch.nn.functional as F

from .drop_tricks import DropoutTrick
from .norm_tricks import *
from .res_tricks import InitialConnection, DenseConnection, ResidualConnection
from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from torch import nn
from torch.nn import init

class TricksComb(nn.Module):
    def __init__(self, args):
        super(TricksComb, self).__init__()
        self.args = args
        self.dglgraph = None
        self.alpha = args.res_alpha
        self.embedding_dropout = args.dropout

        for k, v in vars(args).items():
            setattr(self, k, v)
        # cannot cache graph structure when use graph dropout tricks
        self.cached = self.transductive = args.transductive
        if AcontainsB(self.type_trick, ['DropEdge', 'DropNode', 'FastGCN', 'LADIES']):
            self.cached = False
        # set self.has_residual_MLP as True when has residual connection
        # to keep same hidden dimension
        self.has_residual_MLP = False
        if AcontainsB(self.type_trick, ['Jumping', 'Initial', 'Residual', 'Dense']):
            self.has_residual_MLP = True
        # graph network initialize
        self.layers_GCN = nn.ModuleList([])
        self.layers_res = nn.ModuleList([])
        self.layers_norm = nn.ModuleList([])
        self.layers_MLP = nn.ModuleList([])
        # set MLP layer
        self.layers_MLP.append(nn.Linear(self.num_feats, self.dim_hidden))
        if not self.has_residual_MLP:
            self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached, args = self.args, whetherHasSE=self.args.TeacherGNN.whetherHasSE[0]))

        for i in range(self.num_layers):
            if (not self.has_residual_MLP) and (
                    0 < i < self.num_layers - 1):  # if don't want 0_th MLP, then 0-th layer is assigned outside the for loop
                self.layers_GCN.append(GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached,args = self.args, whetherHasSE=self.args.TeacherGNN.whetherHasSE[1]))
            elif self.has_residual_MLP:
                self.layers_GCN.append(GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached,args = self.args, whetherHasSE=self.args.TeacherGNN.whetherHasSE[1]))

            appendNormLayer(self, args, self.dim_hidden if i < self.num_layers - 1 else self.num_classes)

            # set residual connection type
            if AcontainsB(self.type_trick, ['Residual']):
                self.layers_res.append(ResidualConnection(alpha=self.alpha))
            elif AcontainsB(self.type_trick, ['Initial']):
                self.layers_res.append(InitialConnection(alpha=self.alpha))
            elif AcontainsB(self.type_trick, ['Dense']):
                if self.layer_agg in ['concat', 'maxpool']:
                    self.layers_res.append(
                        DenseConnection((i + 2) * self.dim_hidden, self.dim_hidden, self.layer_agg))
                elif self.layer_agg == 'attention':
                    self.layers_res.append(
                        DenseConnection(self.dim_hidden, self.dim_hidden, self.layer_agg))

        self.graph_dropout = DropoutTrick(args)
        if not self.has_residual_MLP:
            self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=self.cached, args = self.args, whetherHasSE=self.args.TeacherGNN.whetherHasSE[2]))

        if AcontainsB(self.type_trick, ['Jumping']):
            if self.layer_agg in ['concat', 'maxpool']:
                self.layers_res.append(
                    DenseConnection((self.num_layers + 1) * self.dim_hidden, self.num_classes, self.layer_agg))
            elif self.layer_agg == 'attention':
                self.layers_res.append(
                    DenseConnection(self.dim_hidden, self.num_classes, self.layer_agg))
        else:
            self.layers_MLP.append(nn.Linear(self.dim_hidden, self.num_classes))

        # set lambda
        if AcontainsB(self.type_trick, ['IdentityMapping']):
            self.lamda = args.lamda
        elif self.type_model == 'SGC':
            self.lamda = 0.
        elif self.type_model == 'GCN':
            self.lamda = 1.

    def forward(self, x, edge_index, want_les=False):
        if self.dglgraph is None:
            l12 = tonp(edge_index).tolist()
            self.dglgraph = dgl.graph((l12[0],l12[1])).to(self.args.device)
        graph = self.dglgraph

        x_list = []
        le_collection = []
        se_reg_all = None

        new_adjs = self.graph_dropout(edge_index)
        # new_adjs = [edge_index, None]
        if self.has_residual_MLP:
            x = F.dropout(x, p=self.embedding_dropout, training=self.training)
            x = self.layers_MLP[0](x)
            x = F.relu(x)
            x_list.append(x)

        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            edge_index, _ = new_adjs[i]
            beta = math.log(self.lamda / (i + 1) + 1) if AcontainsB(self.type_trick,
                                                                    ['IdentityMapping']) else self.lamda
            # x = self.layers_GCN[i](x, edge_index, beta)
            x, se_reg = self.layers_GCN[i](graph, x)
            if se_reg is not None:
                if se_reg_all is None:
                    se_reg_all = se_reg
                else:
                    se_reg_all += se_reg

            x = run_norm_if_any(self, x, i)

            if want_les:
                le_collection.append(x.clone().detach())

            if self.has_residual_MLP or i < self.num_layers - 1:
                x = F.relu(x)
            x_list.append(x)
            if AcontainsB(self.type_trick, ['Initial', 'Dense', 'Residual']):
                x = self.layers_res[i](x_list)

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        if self.has_residual_MLP:
            if AcontainsB(self.type_trick, ['Jumping']):
                x = self.layers_res[0](x_list)
            else:
                x = self.layers_MLP[-1](x)
        if want_les:
            return x, se_reg_all, th.cat(le_collection, dim=-1)
        else:
            return x, se_reg_all

    def get_se_dim(self, x, edge_index):
        _,_,les = self.forward(x, edge_index, want_les=1)
        return les.shape[-1]

    def collect_SE(self, x, edge_index):
        _,_,les = self.forward(x, edge_index, want_les=1)
        return les

class GCNConv(nn.Module):
    # Cold Brew's GCN; modified from DGL official GCN implementation
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False, cached=None, args=None,whetherHasSE=False):
        super(GCNConv, self).__init__()
        self.args = args
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = True
        self._allow_zero_in_degree = allow_zero_in_degree
        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self._activation = activation

        self.whetherHasSE = whetherHasSE
        if whetherHasSE:
            self.le = nn.Parameter(th.randn(args.N_nodes, self._out_feats), requires_grad=True)
 
    def forward(self, graph, feat, weight=None, edge_weight=None):

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')
            aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            # mult W first to reduce the feature size for aggregation.
            if weight is not None:
                feat_src = th.matmul(feat_src, weight)

            # ______________ add Structural Embeddings ______________
            # Math: X^{(l+1)}=\sigma\left(\tilde{\bm{A}}\left(X^{(l)} W^{(l)}+E^{(l)}\right)\right), X^{(l)} \in R^{N\times d_{1}}, W^{(l)} \in R^{d_1\times d_{2}}, E^{(l)} \in R^{N\times d_{2}}
            # X^{(L)} and X^{(L+1)} is the input and output of the current convolution layer; E is the structural embedding; \tilde{\bm{A}} is the normalized adjacency matrix. W and E are learnable.
            if self.whetherHasSE:
                graph.srcdata['h'] = feat_src + self.le
                se_reg = th.norm(self.le)

            else:
                graph.srcdata['h'] = feat_src
                se_reg = None

            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))

            rst = graph.dstdata['h']

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst, se_reg

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

def tonp(arr):
    if type(arr) is th.Tensor:
        return arr.detach().cpu().data.numpy()
    else:
        return np.asarray(arr)
