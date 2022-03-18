# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# --- torch_geometric below are used to load public datasets; models are implemented in dgl ---
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid, Coauthor, WebKB, Actor, Amazon, WikipediaNetwork
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, to_networkx, negative_sampling
import torch_geometric
import torch_geometric.transforms as T

import argparse
import configparser
import gc
import itertools
import json
import torch as th
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict
import os
import pickle
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from sklearn import metrics
import cv2
import dgl
import pandas as pd
import numpy as np
import os
from time import time as timer
import time
import copy
import numpy.linalg as la
from tqdm import tqdm
import matplotlib.pyplot as plt

def random_mask(L, true_prob=0.9):
    mask = np.random.rand(L)
    mask = mask<true_prob
    return mask

def plot_dist(arr, ttl = '', bins = 100, saveFig_fname = ''):
    arr = tonp(arr).reshape(-1)
    plt.figure()
    plt.title(ttl+f'\nmin = {min(arr):.3f}, max = {max(arr):.3f}')
    value, loc, handle = plt.hist(x=arr, bins=bins, color='#0504aa', alpha=0.5, rwidth=1)
    non0iloc = np.where(value!=0)[0]
    non0loc = (loc[non0iloc] + loc[non0iloc+1])/2
    plt.plot(non0loc,non0loc*0+0.08*max(arr),'k|')
    if saveFig_fname:
        os.makedirs('wIns',exist_ok=1)
        lt2 = time.strftime("%Y-%m-%d--%H_%M_%S", time.localtime())
        plt.savefig(join('wIns',saveFig_fname)+f'__{lt2}.jpg')
    return

def init_split_edge_unified_impl(data, is_bipt=False):
    # require that input 'data' has attr 'is_unique_in_targetG_mask', which is a node mask; or 'is_unique_in_targetG_edge_mask', which is edge mask.

    # ---- generate edges ----
    # regime: 
        # positive edge:
            # edges within large market, all use as training set
            # so long as one end outside large mk, do split into train/valid/test with prob.
        # negative edge: regardless of on large mk or not, sample with the same prob.
        # do this for both edge and ESCI
    has_edge_attr = data.edge_attr != None
    As = graphUtils.edge_index_to_A(data.edge_index, data.num_nodes, edge_weight=data.edge_attr)

    prob_train_p, prob_valid_p = 0.2, 0.4
    # prob_train_n, prob_valid_n = 0.8, 0.1
    prob_train_n, prob_valid_n = 0.2, 0.4
    p_edge_train, p_edge_valid, p_edge_test = [], [], []
    n_edge_train, n_edge_valid, n_edge_test = [], [], []
    p_edge_train_feat, p_edge_valid_feat, p_edge_test_feat = [], [], []

    from tqdm import tqdm
    for ie, e in enumerate(tqdm(tonp(data.edge_index.T).tolist())):
        e = tuple(e)
        _rand = np.random.rand()
        if hasattr(data, 'is_unique_in_targetG_edge_mask'):
            cond0 = not data.is_unique_in_targetG_edge_mask[ie]
        else:
            cond0 = (not data.is_unique_in_targetG_mask[e[0]]) and (not data.is_unique_in_targetG_mask[e[1]]) # both nodes in source graph

        if cond0:
            p_edge_train.append(e)
            p_edge_train_feat.append([A[e] for A in As]) if is_bipt else None
        elif _rand < prob_train_p:
            p_edge_train.append(e)
            p_edge_train_feat.append([A[e] for A in As]) if is_bipt else None
        elif prob_train_p <= _rand < prob_train_p+prob_valid_p:
            p_edge_valid.append(e)
            p_edge_valid_feat.append([A[e] for A in As]) if is_bipt else None
        elif _rand > prob_train_p+prob_valid_p:
            p_edge_test.append(e)
            p_edge_test_feat.append([A[e] for A in As]) if is_bipt else None

    num_edges = data.edge_index.shape[1]
    from torch_geometric.utils import remove_self_loops, to_undirected, to_networkx, negative_sampling, remove_isolated_nodes
    neg_edge_samp = negative_sampling(data.edge_index, num_nodes=(data.N_asin, data.N_kw) if is_bipt else data.num_nodes, num_neg_samples = num_edges)
    neg_edge_samp, _ = remove_self_loops(neg_edge_samp)
    print(f'num of neg sample edge = {neg_edge_samp.shape[1]}')

    for ie, e in enumerate(tqdm(tonp(data.edge_index.T).tolist())):
        _rand = np.random.rand()

        if hasattr(data, 'is_unique_in_targetG_edge_mask'):
            cond0 = not data.is_unique_in_targetG_edge_mask[ie]
        else:
            cond0 = (not data.is_unique_in_targetG_mask[e[0]]) and (not data.is_unique_in_targetG_mask[e[1]]) # both nodes in source graph

        if cond0:
            n_edge_train.append(e)
        elif _rand < prob_train_n:
            n_edge_train.append(e)
        elif prob_train_n <= _rand < prob_train_n+prob_valid_n:
            n_edge_valid.append(e)
        elif _rand > prob_train_n+prob_valid_n:
            n_edge_test.append(e)

    # ---- save processed edges ----
    # elem = {'edge': torch.zeros([10,2]), 'edge_neg':torch.zeros([10,2])}
    if has_edge_attr:
        split_edge = {
            'train':{'edge': torch.tensor(p_edge_train), 'edge_neg':torch.tensor(n_edge_train), 'edge_feat': torch.tensor(p_edge_train_feat)},
            'valid':{'edge': torch.tensor(p_edge_valid), 'edge_neg':torch.tensor(n_edge_valid), 'edge_feat': torch.tensor(p_edge_valid_feat)},
            'test': {'edge': torch.tensor(p_edge_test),  'edge_neg':torch.tensor(n_edge_test),  'edge_feat': torch.tensor(p_edge_test_feat)},
            }
    else:
        split_edge = {
            'train':{'edge': torch.tensor(p_edge_train), 'edge_neg':torch.tensor(n_edge_train)},
            'valid':{'edge': torch.tensor(p_edge_valid), 'edge_neg':torch.tensor(n_edge_valid)},
            'test': {'edge': torch.tensor(p_edge_test),  'edge_neg':torch.tensor(n_edge_test)},
            }
    print(f'\n----------\n checking edge splits: \n pos: train/val/test = {len(p_edge_train),len(p_edge_valid),len(p_edge_test)}\n neg: train/val/test = {len(n_edge_train),len(n_edge_valid),len(n_edge_test)} \n--------------')
    # split_edge['valid']['edge'].shape:        [N_e,2]
    # split_edge['train']['edge_neg'].shape:    [N_e,2]

    return split_edge

def _subgraph(subset, edge_index,
             edge_attr, relabel_nodes,
             num_nodes, return_edge_mask):
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    def maybe_num_nodes(edge_index, num_nodes=None):
        if num_nodes is not None:
            return num_nodes
        elif isinstance(edge_index, Tensor):
            return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
        else:
            return max(edge_index.size(0), edge_index.size(1))

    device = edge_index.device

    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    if subset.dtype == bool: # or subset.dtype == torch.uint8:
        node_mask = subset
        num_nodes = node_mask.shape[0]

        if relabel_nodes:
            node_idx = torch.zeros(node_mask.shape[0], dtype=torch.long,
                                   device=device)
            node_idx[subset] = torch.arange(subset.sum().item(), device=device)
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        node_mask[subset] = 1

        if relabel_nodes:
            node_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
            node_idx[subset] = torch.arange(subset.shape[0], device=device)

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index = node_idx[edge_index]

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr

def down_sample_A(edge_index, perm, edge_weight=None, num_nodes=None, return_edge_mask=False):
    def v2(edge_index, perm, edge_weight=None, num_nodes=None):
        # from torch_geometric.utils import subgraph
        # edge_index, edge_attr
        return _subgraph(perm, edge_index, edge_attr=edge_weight, num_nodes=num_nodes, relabel_nodes=True, return_edge_mask=return_edge_mask)

    def v1(edge_index, perm, edge_weight=None, num_nodes=None):
        perm = torch.tensor(tonp(perm))

        from w import graphUtils, index_sparse_tensor
        As = graphUtils.edge_index_to_A(edge_index, num_nodes=num_nodes, edge_weight=edge_weight, want_tensor=1)

        if type(As) is list:
            As = [index_sparse_tensor(A, perm) for A in As]
        else:
            As = index_sparse_tensor(As, perm)

        edge_index, edge_weight = graphUtils.A_to_edge_index(As)
        return edge_index, edge_weight

    return v2(edge_index, perm, edge_weight, num_nodes)

def down_sample_graph_with_node_perm(data, perm=None, drop_rate=0.9, do_remove_isolated_nodes=True):
    data.num_nodes = len(data.x)
    if perm is None:
        perm = np.array(sorted(np.random.choice(len(data.x), int(len(data.x)*(1-drop_rate)), replace=False)))
    else:
        perm = tonp(perm)

    # ---- down sample A ---
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        data.edge_index, data.edge_attr, e_mask = down_sample_A(data.edge_index, perm, edge_weight=data.edge_attr, num_nodes=data.num_nodes, return_edge_mask=True)
        # data.edge_weight = data.edge_attr
    elif hasattr(data, 'edge_weight') and data.edge_weight is not None:
        data.edge_index, data.edge_weight, e_mask = down_sample_A(data.edge_index, perm, edge_weight=data.edge_weight, num_nodes=data.num_nodes, return_edge_mask=True)
        # data.edge_attr = data.edge_weight
    else:
        data.edge_index, _ = down_sample_A(data.edge_index, perm, num_nodes=data.num_nodes)

    # ---- remove iso nodes shall follows downsample and before building adj_t ----
    if do_remove_isolated_nodes:
        from torch_geometric.utils import remove_isolated_nodes
        data.edge_index, data.edge_attr, _mask = remove_isolated_nodes(data.edge_index, edge_attr=data.edge_attr, num_nodes=len(perm))
        perm = perm[_mask]

    # ---- re-compute N_nodes ----
    if hasattr(data, 'N_asin'):
        data.N_asin = (perm<data.N_asin).sum()
        data.N_kw = len(perm) - data.N_asin
    data.N_nodes = data.num_nodes = data.edge_index.max() + 1
    # ---- make sure the isolated nodes does not exist at last, otherwise some GNN methods will bug ----
    perm = perm[:data.num_nodes]
    data.down_sample_perm = torch.tensor(perm)

    # ---- down sample x ---
    data.x = data.x[perm]
    if hasattr(data, 'texts'):
        data.texts = [data.texts[p] for p in perm]
    if hasattr(data, 'node_year'):
        data.node_year = data.node_year[perm]
    if hasattr(data, 'edge_year'):
        data.edge_year = data.edge_year[e_mask]

    return data

def get_A_from_relabeled_edge_index(data, subset_idx, subset_newID, num_nodes):
    # given a graph, a subset of nodes, and the new target idx to be relabeld, return the A matrix of the relabeld subgraph.
    if hasattr(data, 'edge_attr'):
        _edge_index, _edge_attr = subgraph_relabel(data.edge_index, subset_idx, subset_newID=subset_newID, edge_attr=data.edge_attr, return_edge_mask=False)
        A = graphUtils.edge_index_to_A(_edge_index, num_nodes=num_nodes, edge_weight=_edge_attr, want_tensor=1)
    elif hasattr(data, 'edge_weight'):
        _edge_index, _edge_weight = subgraph_relabel(data.edge_index, subset_idx, subset_newID=subset_newID, edge_attr=data.edge_weight, return_edge_mask=False)
        A = graphUtils.edge_index_to_A(_edge_index, num_nodes=num_nodes, edge_weight=_edge_weight, want_tensor=1)
    else:
        _edge_index, _ = subgraph_relabel(data.edge_index, subset_idx, subset_newID=subset_newID, edge_attr=None, return_edge_mask=False)
        A = graphUtils.edge_index_to_A(_edge_index, num_nodes=num_nodes, edge_weight=None, want_tensor=1)

    return A

def retrieve_edge_index_from_A(data, A):
    if hasattr(data, 'edge_attr'):
        data.edge_index, data.edge_attr = graphUtils.A_to_edge_index(A)
    elif hasattr(data, 'edge_weight'):
        data.edge_index, data.edge_weight = graphUtils.A_to_edge_index(A)
    else:
        data.edge_index, _ = graphUtils.A_to_edge_index(A)
    return data

def graph_analyze(N_nodes, edge_index):

    edge_index = edge_index.cpu().data.numpy()
    degreedic = {} # inode ›› [indeg, outdeg]
    for ie in range(edge_index.shape[1]):
        ori, dst = edge_index[:,ie]

        if not degreedic.get(ori):
            degreedic[ori] = [0,0]
        if not degreedic.get(dst):
            degreedic[dst] = [0,0]
        if ori==dst:
            # print('this edge is node to itself: node #',ori)
            pass
        if ori==234:
            # print('node 234')
            pass
        
        degreedic[ori][0] += 1
        degreedic[dst][1] += 1
    degs_ori, degs_dst = [], []    
    for ino in range(N_nodes):
        ori_dst = degreedic.get(ino)
        if not ori_dst:
            degs_ori.append(0)
            degs_dst.append(0)
        else:
            degs_ori.append(ori_dst[0])
            degs_dst.append(ori_dst[1])
            if ori_dst[0]!=ori_dst[1]:
                pass
                # print('inequal!  ',ino)


    return tonp(degs_ori), tonp(degs_dst)

def get_item2id(items):
    item2id = {}
    for i,x in enumerate(items):
        item2id[x] = i
    return item2id

def cal_union(data1, data2, is_bipt=False):
    # input data is homo graph, it has attribute:
        # x
        # text (optional)
        # edge_index
        # edge_weight
        # y (optional)
    # returns:
        # x; edge_index; edge_attr;
        # sharing_status: shape = [N_nodes_union], value can be: 0/1/2: 0 means graph_1 unshared, 1 means shared, 2 means graph_2 unshared.
        # is_shared_1_idx/is_shared_1_idx_map2/is_shared_1_mask: np.array, shape = [N_nodes_shared] / [N_nodes_shared] / [N_nodes_1]

    # how to perform union (calculate new edge_index):
        # node features is the concatenation of [x2, x1_unshared]
        # edge has two types: those linked to graph2_unshared, and not. 
        # edge_index is computed from A, and A is the addition of A_1 (expanded) and A_2 (expanded).
        # how to expand A_1: simply add zeros to the end.
        # how to expand A_2: permute twice:
            # first permute: shared nodes permute to first;
            # then expand with zeros (N_nodes_unshared_1) to the front;
            # then second permute: shared nodes move to the correct index locations.

    # ---- get necessary mapping ----
    def get_union_graphs_idx_mapU(t1, t2, N_asin_12=()):
        # returns two nodeID -> union graph ID mappings.
        # t1 & t2 are globally uniquely identifiable node labels, such as raw texts.
        # node features is the concatenation of [x2, x1_unshared]
        # N_asin_12 != () means bipartite graph.

        if not type(t1[0]) is str:
            t1 = [tuple(x) for x in tonp(t1).tolist()]
            t2 = [tuple(x) for x in tonp(t2).tolist()]

        idx_2_map_U = np.arange(len(t2))

        is_unique_in_targetG_mask = [True]*len(t2)
        import copy
        texts = copy.deepcopy(list(t2))

        if N_asin_12==():
            # ---- get nodeID to union graph ID mapping: homo graph: [x2, x1_unshared] ----
            idx_1_map_U = []
            t_map_i_2 = get_item2id(t2)
            id_starting_graph2 = len(t2)

            for i,t in enumerate(t1):
                if t in t_map_i_2:
                    idx_1_map_U.append(t_map_i_2[t])
                    is_unique_in_targetG_mask[t_map_i_2[t]] = False
                else:
                    idx_1_map_U.append(id_starting_graph2)
                    is_unique_in_targetG_mask.append(False)
                    texts.append(t)
                    id_starting_graph2 += 1
            idx_1_map_U = np.array(idx_1_map_U)
            is_unique_in_targetG_mask = torch.tensor(is_unique_in_targetG_mask)
            return idx_1_map_U, idx_2_map_U, is_unique_in_targetG_mask, texts, None, None

        else:
            # ---- get nodeID to union graph ID mapping: bipt graph: [asin2, asin1_unshared, kw2, kw1_unshared] ----
            asin1, kw1 = t1[:N_asin_12[0]], t1[N_asin_12[0]:]
            asin2, kw2 = t2[:N_asin_12[1]], t2[N_asin_12[1]:]

            idx_1_map_U = []

            # ---- process asin1 ----
            t_map_i_2 = get_item2id(asin2)
            id_starting_graph2 = len(asin2)
            for i,t in enumerate(asin1):
                if t in t_map_i_2:
                    idx_1_map_U.append(t_map_i_2[t])
                    is_unique_in_targetG_mask[t_map_i_2[t]] = False
                else:
                    idx_1_map_U.append(id_starting_graph2)
                    is_unique_in_targetG_mask.insert(id_starting_graph2, False)
                    texts.insert(id_starting_graph2, t)
                    id_starting_graph2 += 1

            num_asin_all = id_starting_graph2
            num_graph1_unshared_asin = num_asin_all - len(asin2)
            
            # ---- process kw1 ----
            t_map_i_2 = get_item2id(kw2)
            id_starting_graph2 = len(kw2) + num_asin_all
            for i,t in enumerate(kw1):
                if t in t_map_i_2:
                    idx_1_map_U.append(t_map_i_2[t] + num_asin_all)
                    is_unique_in_targetG_mask[t_map_i_2[t] + num_asin_all] = False
                else:
                    idx_1_map_U.append(id_starting_graph2)
                    is_unique_in_targetG_mask.insert(id_starting_graph2, False)
                    texts.insert(id_starting_graph2, t)
                    id_starting_graph2 += 1
            num_nodes_all = id_starting_graph2
            idx_1_map_U = np.array(idx_1_map_U)

            # ---- process asin2 & kw2 ----
            idx_2_map_U[N_asin_12[1]:] += num_graph1_unshared_asin
            N_asin = num_asin_all
            N_kw = num_nodes_all-num_asin_all

            is_unique_in_targetG_mask = torch.tensor(is_unique_in_targetG_mask)
            assert len(is_unique_in_targetG_mask)==num_nodes_all
            return idx_1_map_U, idx_2_map_U, is_unique_in_targetG_mask, texts, N_asin, N_kw

    # ---- init dataU ----
    dataU = torch_geometric.data.data.Data()
    dataU.edge_attr = None

    if is_bipt:
        idx_1_map_U, idx_2_map_U, is_unique_in_targetG_mask, texts, N_asin, N_kw = get_union_graphs_idx_mapU(data1.texts, data2.texts, (data1.N_asin, data2.N_asin))
        dataU.texts = texts
    else:
        idx_1_map_U, idx_2_map_U, is_unique_in_targetG_mask, texts, N_asin, N_kw = get_union_graphs_idx_mapU(data1.texts, data2.texts, ())

    dataU.num_nodes = dataU.N_nodes = max(idx_1_map_U.max(), idx_2_map_U.max()) +1
    dataU.N_asin, dataU.N_kw, dataU.is_unique_in_targetG_mask = N_asin, N_kw, is_unique_in_targetG_mask

    # ---- check ----
    assert int(is_unique_in_targetG_mask.sum() + data1.N_nodes) == int(dataU.N_nodes)
    print(f'check N_asin, N_kw: previously = {(data1.N_asin, data1.N_kw)}, {(data2.N_asin, data2.N_kw)} \n\t  now = {(N_asin, N_kw)}')

    # ---- compute x ----
    dataU.x = torch.zeros([dataU.num_nodes, data2.x.shape[1]], dtype=data2.x.dtype, device=data2.edge_index.device)
    dataU.x[idx_1_map_U] = data1.x
    dataU.x[idx_2_map_U] = data2.x

    # ---- compute edge_index ----
    A1 = get_A_from_relabeled_edge_index(data1, subset_idx=torch.arange(data1.num_nodes, dtype=torch.long, device=data1.edge_index.device), subset_newID=idx_1_map_U, num_nodes=dataU.num_nodes)
    A2 = get_A_from_relabeled_edge_index(data2, subset_idx=torch.arange(data2.num_nodes, dtype=torch.long, device=data2.edge_index.device), subset_newID=idx_2_map_U, num_nodes=dataU.num_nodes)
    A_U = graphUtils.add_As(A1,A2)
    dataU = retrieve_edge_index_from_A(dataU, A_U)

    return dataU

def target_seeded_by_source(data1, data2, actually_do_addition=True):
    # return the modified target graph: nodes in g2 remain unchanged; edges seeded by g1.
    # also need: is_unique_in_targetG_mask

    # ---- get necessary mapping ----
    def get_shared_node_idx_and_map(t1, t2):
        # returns two nodeID -> union graph ID mappings.
        # t1 & t2 are globally uniquely identifiable node labels, such as raw texts.
        t_map_i_2 = get_item2id(t2)
        is_shared_1_idx, is_shared_1_idx_map2 = [], []
        for i,t in enumerate(t1):
            if t in t_map_i_2:
                is_shared_1_idx.append(i)
                is_shared_1_idx_map2.append(t_map_i_2[t])
        return is_shared_1_idx, is_shared_1_idx_map2
    is_shared_1_idx, is_shared_1_idx_map2 = get_shared_node_idx_and_map(data1.texts, data2.texts)

    is_unique_in_targetG_mask = torch.ones(data2.num_nodes).bool()
    is_unique_in_targetG_mask[is_shared_1_idx_map2] = False
    data2.is_unique_in_targetG_mask = is_unique_in_targetG_mask

    if actually_do_addition:
        # ---- prepare A1/A2 matrix for addition ----
        A2 = graphUtils.edge_index_to_A(data2.edge_index, num_nodes=data2.num_nodes, edge_weight=data2.edge_attr, want_tensor=1)
        A1 = get_A_from_relabeled_edge_index(data1, is_shared_1_idx, is_shared_1_idx_map2, data2.num_nodes)

        # ---- perform addtion ----
        A2 = graphUtils.add_As(A1,A2)

        # ---- retrieve orig data format ----
        data2 = retrieve_edge_index_from_A(data2, A2)

    return data2

def down_sample_edge_split(edge_split, perm):
    def check_maxv(edge_split):
        maxv = -1
        for key_level1, v1 in edge_split.items():
            for key_level2, v2 in v1.items():
                if 'feat' not in key_level2:
                    maxv = max(maxv, v2.max())
        print(f'in down_sample_edge_split, maxv is {maxv}')
        return

    check_maxv(edge_split)
    for key_level1, v1 in edge_split.items():
        for key_level2, v2 in v1.items():
            if 'feat' in key_level2: # only 'edge_feat' has shape [N_e, 5]; others are all edge_index.T, which have shape [N_e,2]
                k_hold, v_hold = key_level2, v2
            else:
                v2, _, edge_mask_tmp = down_sample_edge_index(v2.T, perm)
                edge_split[key_level1][key_level2] = v2.T
                if key_level2=='edge':
                    edge_mask = edge_mask_tmp
        edge_split[key_level1][k_hold] = v_hold[edge_mask]
    check_maxv(edge_split)
    return edge_split

def down_sample_edge_index(edge_index, perm, edge_attr=None, return_edge_mask=True):
    # this is core function of downsampling the edge_index of original labeled graph, given the newly labeled subgraph (perm is the selected nodes).
    return subgraph_relabel(edge_index, perm, edge_attr=edge_attr, return_edge_mask=return_edge_mask, subset_newID=None)

def subgraph_relabel(edge_index, subset_idx, subset_newID=None, edge_attr=None, return_edge_mask=True):
    # given a graph, a subset of nodes, and their new label, return the new subgraph.

    device = edge_index.device
    subset_idx = torch.tensor(tonp(subset_idx), dtype=torch.long, device=device)

    num_nodes = max([edge_index.max(), subset_idx.max()]) + 1
    node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    node_mask[subset_idx] = 1

    if subset_newID is None:
        subset_newID = torch.arange(len(subset_idx), device=device)
    else:
        subset_newID = torch.tensor(tonp(subset_newID), dtype=torch.long, device=device)
        assert subset_idx.shape == subset_newID.shape
    node_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
    node_idx[subset_idx] = subset_newID

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    edge_index = node_idx[edge_index]

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr

def cal_recall(pos_score_1D, neg_score_1D, topk=None):
    if topk is None or float(topk)==0: # default threshold by 0.
        return toitem((pos_score_1D>0).sum())/len(pos_score_1D)
    elif float(topk)>5: # absolute value
        topk = int(topk)
    else: # relative value
        topk = int(float(topk)*len(pos_score_1D))

    N_pos_total = len(pos_score_1D)
    force_greater_0 = 1
    if force_greater_0:
        pos_score_1D = pos_score_1D[pos_score_1D>0]

    scores = np.concatenate([tonp(pos_score_1D), tonp(neg_score_1D)])
    labels = np.concatenate([np.ones(len(pos_score_1D)), np.zeros(len(neg_score_1D))])
    arr = np.asarray([scores, labels]).T   # shape=[N,2]
    sortarr = np.asarray(sorted(list(arr), key=lambda x: x[0], reverse=True)) # [N,2]
    recall = sortarr[:topk,1].sum()/N_pos_total
    return recall

def set_arch_configs(args):

    target_dim_CV = 108 
    args.SEMLP__downgrade_to_MLP = args.SEMLP_topK_2_replace == -99
    args.activation = ['relu' , 'gelu'][1]
    args.is_bipartite = False

    args.TeacherGNN = C()
    args.TeacherGNN.lossa_semantic = 1
    args.TeacherGNN.lossa_structure = 1
    args.TeacherGNN.change_to_featureless = args.change_to_featureless
    args.TeacherGNN.num_layers = args.num_layers

    if args.whetherHasSE=='111':  # SE refers to structural embeddging;  The first ‘1’ means structural embedding exist in first layer; second ‘1’ means structural embedding exist in every middle layers; third ‘1’ means last layer.
        args.TeacherGNN.whetherHasSE = [1,1,1]
    elif args.whetherHasSE=='000':
        args.TeacherGNN.whetherHasSE = [0,0,0]
    elif args.whetherHasSE=='001':
        args.TeacherGNN.whetherHasSE = [0,0,1]
    elif args.whetherHasSE=='100':
        args.TeacherGNN.whetherHasSE = [1,0,0]
    else:
        raise NotImplementedError

    # below is the output of GraphConv. It is the common "embedding" for link prediction and node classification
    if args.has_proj2class:
        args.dim_commonEmb = 128
    else:
        args.dim_commonEmb = args.num_classes
    args.num_feats_bkup = args.num_feats
    args.embDim_linkp = 10
    args.num_classes_bkup = args.num_classes

    _neurons_proj2class_ = [20]
    _neurons_proj2linkp = [32]  # last dim of this is the projected emb for linkp. can also directly use dim_commonEmb
    args.TeacherGNN.neurons_proj2class = [args.dim_commonEmb] + _neurons_proj2class_ + [args.num_classes_bkup]
    args.TeacherGNN.neurons_proj2linkp = [args.dim_commonEmb] + _neurons_proj2linkp

    dim2mimic = args.dim_commonEmb
    args.StudentBaseMLP = C()

    if args.studentMLP__skip_conn_T_and_res_blks != '':
        skip_period, num_blocks = args.studentMLP__skip_conn_T_and_res_blks.split('&')
        args.StudentBaseMLP.skip_conn_period = int(skip_period)
        args.StudentBaseMLP.num_blocks = int(num_blocks)
    else:
        args.StudentBaseMLP.skip_conn_period = int(2)
        args.StudentBaseMLP.num_blocks = int(3)
    args.StudentBaseMLP.dims_in_out = [args.num_feats_bkup, args.num_classes_bkup]
    args.StudentBaseMLP.dim_model = args.StudentMLP__dim_model
    args.StudentBaseMLP.lrn_from = ['teacherGNN', 'label'][1]

    if args.studentMLP__opt_lr != '':
        _opt, _lr = args.studentMLP__opt_lr.split('&')
        args.optfun = _opt
        args.lr = float(_lr)

    return

def unfold_NCS_2_NLKC(unfolder, arr_NCS, kernal_size):
    # this function is applicable to kernal size of 2D, 3D, ... any shape.
    # S is for num of spacial tokens
    # in returned shape, K is expanded
    arr__N_CK_L = unfolder(arr_NCS)
    N, CK, L = arr__N_CK_L.shape
    C, K = CK//np.prod(kernal_size), np.prod(kernal_size)
    arr__N_L_C_K = arr__N_CK_L.permute(0,2,1).view(N,L,C,K)
    arr__NLKC = arr__N_L_C_K.permute(0,1,3,2)
    Ks = list(kernal_size)
    shape_ = [N,L] + Ks + [C]
    arr__NLKC_KExpand = arr__NLKC.view(*shape_)
    return arr__NLKC_KExpand

def AcontainsB(A,listB):
    # A: string; listB: list of strings
    for s in listB:
        if s in A: return True
    return False

def ensure_symmetric(edge_index):
    # edge_index: a tensor of shape [2, N_edge]
    N = edge_index.max()+1
    v = torch.ones(edge_index.shape[1], dtype=torch.long)
    tens = torch.sparse_coo_tensor(edge_index, v, [N,N])
    tens = tens + tens.t() 
    new_index = tens.coalesce().indices()
    return new_index

def gen_rec_for_table1_stats(N_nodes, degs):
    recs = [N_nodes, np.sum(degs), np.max(degs), np.mean(degs), np.median(degs), sum(np.asarray(degs)==0)/N_nodes*100 ]
    return recs

def save_graph_analyze(N_nodes,data,use_special_split):
    data.N_nodes = N_nodes
    degs_ori, degs_dst = graph_analyze(N_nodes, data.edge_index)
    
    want_stats = 1
    if want_stats:
        # np.save('viz_distri_xxx.npy', sorted(degs_ori, reverse = True))
        np.save('viz_distri_v0_cora.npy', sorted(degs_ori, reverse = True))
        stats = gen_rec_for_table1_stats(N_nodes, degs_ori)
        print(stats)

    if not use_special_split:
        small_deg_idx = get_partial_sorted_idx(degs_dst, 'top3')
        large_deg_idx = get_partial_sorted_idx(degs_dst, 'bottom3')
        data.small_deg_idx = small_deg_idx
        data.large_deg_idx = large_deg_idx
        data.small_deg_mask = torch.tensor([False]*N_nodes,device=data.x.device)
        data.large_deg_mask = torch.tensor([False]*N_nodes,device=data.x.device)
        data.small_deg_mask[small_deg_idx] = True
        data.large_deg_mask[large_deg_idx] = True
    else:
        _idx = get_partial_sorted_idx(degs_dst, 'top6')
        interm = np.array(degs_dst)[_idx].argsort()
        _idx = _idx[interm]
        zero_deg_idx = _idx[:len(_idx)//2]
        small_deg_idx = _idx[len(_idx)//2:]
        large_deg_idx = get_partial_sorted_idx(degs_dst, 'bottom3')
        data.zero_deg_idx = zero_deg_idx
        data.small_deg_idx = small_deg_idx
        data.large_deg_idx = large_deg_idx

        data.zero_deg_mask = torch.tensor([False]*N_nodes,device=data.x.device)
        data.small_deg_mask = torch.tensor([False]*N_nodes,device=data.x.device)
        data.large_deg_mask = torch.tensor([False]*N_nodes,device=data.x.device)

        data.zero_deg_mask[zero_deg_idx] = True
        data.small_deg_mask[small_deg_idx] = True
        data.large_deg_mask[large_deg_idx] = True
        print(f'\n\n\n  isolation ratio is:   {len(zero_deg_idx)/N_nodes*100:.2f} %')

        craft_isolation_v2(data)

    want_plot = 0
    if want_plot:
        ttl = 'ogbn-arxiv >> node degree distribution'
        degs_ori = np.asarray(degs_ori)
        degs_ori = degs_ori[degs_ori<=80]
        plot_dist(degs_ori, ttl, saveFig_fname=ttl, bins=80)

    return 

def craft_isolation_v2(data):
    zero_deg_mask = data.zero_deg_mask
    edge_index_np = tonp(data.edge_index)

    edge_index_crafted = []
    cnt = 0
    for idx in range(edge_index_np.shape[1]):
        ori, dst = edge_index_np[:,idx]
        if (ori!=dst) and (zero_deg_mask[ori] or zero_deg_mask[dst]):
        # if (zero_deg_mask[ori] or zero_deg_mask[dst]) :
            # print(f'now removing/crafting, edge is {(ori,dst)}, labels = {(node_flag[ori],node_flag[dst])}')
            cnt += 1
        else:
            edge_index_crafted.append([ori, dst])
    edge_index_crafted = torch.tensor(edge_index_crafted).t().to(data.edge_index.device)
    print(f'removed < {cnt} > edge; shape change: {data.edge_index.shape} ›› {edge_index_crafted.shape}')
    print('-'*20,'\n\n')

    # --- to return ---
    data.edge_index_bkup = data.edge_index
    data.edge_index = edge_index_crafted
    return

def calc_score(h_emb, t_emb):
    # DistMult
    score = th.sum(h_emb * t_emb, dim=-1)
    return score

def linkp_loss_eva(h_emb, t_emb, nh_emb, nt_emb):

    num_p = h_emb.shape[0]
    num_n = nh_emb.shape[0]
    N = num_n + num_p

    pos_score = calc_score(h_emb, t_emb).view(-1,1)
    neg_score = calc_score(nh_emb, nt_emb).view(-1,1)

    score = th.cat([pos_score, neg_score])
    label = th.cat([th.full((pos_score.shape), 1.0), th.full((neg_score.shape), 0.0)]).to(score.device)

    mrr = cal_MRR(pos_score.view(-1), neg_score.view(-1))
    predict_loss = F.binary_cross_entropy_with_logits(score, label)

    return predict_loss, mrr

def cal_MRR(pos_score, neg_score):
    num_neg_per_pos = len(neg_score)//len(pos_score)
    drop_end = num_neg_per_pos*len(pos_score)
    pos_neg = torch.cat([pos_score.reshape(-1,1), neg_score[:drop_end].reshape(len(pos_score), num_neg_per_pos)], dim=1)

    _, indices = th.sort(pos_neg, dim=1, descending=True)
    indices_correct = th.nonzero(indices == 0, as_tuple=False)  # get all locations where value is zero (all rows treated equally)
    rankings = indices_correct[:, 1].view(-1) + 1   # indices[:, 0] is the row-coordinate, indices[:, 1] is the column-coordinate
    rankings = rankings.cpu().detach().numpy()
    mrr_sum = 0
    mrr_len = 0
    for ranking in rankings:
        mrr_sum += 1.0 / ranking
        mrr_len += 1
    mrr = mrr_sum / mrr_len
    return mrr

class DoNothing(nn.Module):
    def __init__(self,*a):
        super().__init__()
    def forward(self, *a):
        return a[0]
class MyDataset(torch_geometric.data.data.Data):
    def __init__(self):
        super().__init__()

def viz_tsne(data, color=None, perplexity=30):
    from viz_graph_tsne_nx import viz_graph
    data.device = data.x.device
    data = data.to('cpu')
    xe = viz_graph(data.x, data.edge_index, color, perplexity)

    data = data.to(data.device)
    return

def run_pureLP(res):
    from Label_propagation_model.outcome_correlation import general_outcome_correlation,label_propagation,double_correlation_autoscale, double_correlation_fixed,gen_normalized_adjs,process_adj
    res.lpStep_alpha = 0.5
    res.lpStep_num_propagations = 50
    adj, D_isqrt = process_adj(res.data)
    DAD, DA, AD = gen_normalized_adjs(adj, D_isqrt)
    res.adj = DAD
    alpha_term = 1
    lp_dict = {
            'train_only': True,
            'display': False,
            'device': res.device,

            # below: lp only
            'idxs': ['intersection'],  # these is used for selecting res.split_idx: res.split_idx is a dict, key is whatever inside this 'idxs', value is the node indices (1D integer vec), these nodes will be the source, to propagate to other nodes
            'alpha': res.lpStep_alpha,
            'num_propagations': res.lpStep_num_propagations,
            'A': res.adj,

            # below: gat
            'labels': ['train'],
        }
    
    if res.args.LP_use_softmax:
        print('          LP_use_softmax      =     1')
        res.out = general_outcome_correlation(adj=res.adj, y=res.embs_LPGraph, alpha=res.lpStep_alpha, num_propagations=res.lpStep_num_propagations, post_step=lambda x:torch.clamp(x,1e-6,1), alpha_term=alpha_term, device='cpu', display=False)
        res.out = torch.exp(res.out)
    else:
        res.out = general_outcome_correlation(adj=res.adj, y=res.embs_LPGraph, alpha=res.lpStep_alpha, num_propagations=res.lpStep_num_propagations, post_step=lambda x:x, alpha_term=alpha_term, device='cpu', display=False)

    if res.verbose:
        analyze_LP(res)

    return res

def analyze_LP(res):
    embs_pre = res.embs_LPGraph
    embs_after = res.out
    print('\nin analyze_LP')
    print(f'\t num-propagation = {res.lpStep_num_propagations}')
    print(f'\t embs_pre.shape={embs_pre.shape}, embs_after.shape={embs_after.shape}')
    print(f'\t zero-pre = {len(torch.where(embs_pre.sum(dim=1)==0)[0]) }')
    print(f'\t zero-after = {len(torch.where(embs_after.sum(dim=1)==0)[0]) }')

    return

class D:
    def __repr__(self):
        values = []
        def append_values_iterative(obj, prefix=''):
            for att in dir(obj):
                if type(eval(f'obj.{att}')) is D:
                    values.append(prefix + f'{att:16} = \n')
                    append_values_iterative(eval(f'obj.{att}'), prefix='\t')
                elif not att.startswith('__'):
                    v = eval(f'obj.{att}')
                    values.append(prefix + f'{att:16} = {v}\n')
            values.append('\n')
            return
        append_values_iterative(self)
        values = ''.join(values)
        sep = '-'*40 + '\n'
        reprs = sep + values
        return reprs
C = D
choice = np.random.choice
join = os.path.join

def AcontainsB(A,listB):
    # A: string; listB: list of strings
    for s in listB:
        if s in A: return True
    return False

def getMLP(neurons, activation=nn.GELU, bias=True, dropout=0.1, last_dropout=False, normfun='layernorm'):
    # How to access parameters in module: replace printed < model.0.weight > to < model._modules['0'].weight >
    # neurons: all n+1 dims from input to output
    # len(neurons) = n+1
    # num of params layers = n
    # num of activations = n-1
    if len(neurons) in [0,1]:
        return nn.Identity()
    if len(neurons) == 2:
        return nn.Linear(*neurons)

    nn_list = []
    n = len(neurons)-1
    for i in range(n-1):
        if normfun=='layernorm':
            norm = nn.LayerNorm(neurons[i+1])
        elif normfun=='batchnorm':
            norm = nn.BatchNorm1d(neurons[i+1])
        nn_list.extend([nn.Linear(neurons[i], neurons[i+1], bias=bias), norm, activation(), nn.Dropout(dropout)])
    
    nn_list.extend([nn.Linear(neurons[n-1], neurons[n], bias=bias)])
    if last_dropout:
        nn_list.extend([nn.Dropout(dropout)])
    return nn.Sequential(*nn_list)

def get_partial_sorted_idx(arr, mode='top25'):
    # mode: top25, bottom25, top50, bottom50; top = smaller
    arr = tonp(arr).reshape(-1)
    if 'top' in mode:
        idx = np.where(arr<=np.median(arr))[0]
    else:
        idx = np.where(arr>=np.median(arr))[0]
    
    arr1 = arr[idx]
    if mode in ['top25','top12','top6','top3']:
        idx = np.where(arr<=np.median(arr1))[0]
    elif mode in ['bottom25','bottom12','bottom6','bottom3']:
        idx = np.where(arr>=np.median(arr1))[0]
    
    arr1 = arr[idx]
    if mode in ['top12','top6','top3']:
        idx = np.where(arr<=np.median(arr1))[0]
    elif mode in ['bottom12','bottom6','bottom3']:
        idx = np.where(arr>=np.median(arr1))[0]
    
    arr1 = arr[idx]
    if mode in ['top6','top3']:
        idx = np.where(arr<=np.median(arr1))[0]
    elif mode in ['bottom6','bottom3']:
        idx = np.where(arr>=np.median(arr1))[0]

    arr1 = arr[idx]
    if mode in ['top3']:
        idx = np.where(arr<=np.median(arr1))[0]
    elif mode in ['bottom3']:
        idx = np.where(arr>=np.median(arr1))[0]

    return idx

def tonp(arr):
    if type(arr) is torch.Tensor:
        return arr.detach().cpu().data.numpy()
    else:
        return np.asarray(arr)

def toitem(arr,round=True):
    arr1 = tonp(arr)
    value = arr1.reshape(-1)[0]
    if round:
        value = np.round(value,3)
    assert arr1.size==1
    return value

def save_model(net, cwd): # June 2021
    torch.save(net.state_dict(), cwd)
    print(f"‹‹‹‹‹‹‹---  Saved @ :{cwd}\n\n\n")

def load_model(net, cwd, verbose=True, strict=True, multiGPU=False):
    def load_multiGPUModel(network, cwd):
        network_dict = torch.load(cwd)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in network_dict.items():
            namekey = k[7:] # remove `module.`
            new_state_dict[namekey] = v
        # load params
        network.load_state_dict(new_state_dict)

    def load_singleGPUModel(network, cwd):
        network_dict = torch.load(cwd, map_location=lambda storage, loc: storage)
        network.load_state_dict(network_dict, strict=strict)

    if os.path.exists(cwd):
        if not multiGPU:
            load_singleGPUModel(net, cwd)
        else:
            load_multiGPUModel(net, cwd)

        if verbose: print(f"---››››  LOAD success! from {cwd}\n\n\n")
    else:
        if verbose: print(f"---››››  !!! FileNotFound when load_model: {cwd}\n\n\n")

def viz(net, ttl=''):
    def numParamsOf(net):
        return sum(param.numel() for param in net.parameters())

    viz_ = []
    flop_est = 0
    for i, (name, p) in enumerate(net.named_parameters()):
        print(f'{name:36}  {list(p.size())}')
        _size = list(p.size())
        viz_.append((name, p.numel(), _size))
        if len(_size)==2: flop_est += _size[0]*_size[1]

    ttl = str(type(net)) if ttl=='' else ttl
    print(f'\nAbove is viz for: {ttl}.\n\tDevice is: {p.device}\n\tN_groups = {len(viz_)}\n\tTotal params = {numParamsOf(net)}\n\tMLP FLOP ~= {flop_est}')
    
    return

def wzRec(datas, ttl='', want_save_npy=False, npy_dir='', save_history_fig=True):
    # this function save the input datas into two places: ___wIns___.pdf and a record history in wIns folder;
    # datas: 1D or 2D of the same meaning, multiple collections
    # two options: 
    #     want_save_npy: save data or not
    #     save_history_fig: save history record fig or not

    if type(datas) is torch.Tensor:
        datas = datas.detach().cpu().data.numpy()

    if save_history_fig:
        os.makedirs(f'wIns',exist_ok=1)
    if want_save_npy:
        npy_fname = 'some_arr' if ttl == '' else ttl
        recDir = join('wIns/Recs', npy_dir)
        fDirName = f'{recDir}/{npy_fname}.npy'
        os.makedirs(recDir, exist_ok=1)
        np.save(fDirName, datas)
    else:
        fDirName = 'data not saved'

    datas = np.asarray(datas)
    plt.close('all')
    plt.figure()
    if len(datas.shape)==1:
        min_v = min(datas)
        plt.plot(datas)
        plt.title(ttl+f', min = {min_v:5.4f}\n')
        plt.xlabel('step')
    elif len(datas.shape)==2:
        min_s = np.min(datas, axis=1)
        mean_min = np.mean(min_s)
        std_min = np.std(min_s)
        min_str = f'(avg={mean_min:5.4f},std={std_min:5.4f})'
        plot_ci(datas, ttl=ttl+f', min = {min_str}\n', xlb='step')
    else:
        raise ValueError('dim should be 1D or 2D')

    lt2 = time.strftime("%Y-%m-%d--%H_%M_%S", time.localtime())
    figDirName = 'fig not saved'
    if save_history_fig: 
        figDirName = f'wIns/{ttl}__{lt2}.jpg'
        plt.savefig(figDirName)
    plt.savefig('___wIns___.pdf',bbox_inches='tight')
    plt.show()

    return figDirName, fDirName

def figure():
    plt.figure(); plt.pause(0.01)

def plot_many(arr_list, legends, ttl='', save_history_fig=True, have_line_yon='y', marker_size_bos='s'):
    # arr_list is a list of 1D arrays, lengths can be different
    assert len(arr_list)==len(legends)
    for i in range(len(arr_list)):
        plt.plot(arr_list[i], random_line_marker(have_line_yon=have_line_yon, marker_size_bos=marker_size_bos), label = legends[i],linewidth=1.2, markersize=5)
    plt.legend()
    plt.title(ttl)
    if save_history_fig:
        os.makedirs(f'wIns',exist_ok=1)
        lt2 = time.strftime("%Y-%m-%d--%H_%M_%S", time.localtime())
        figDirName = f'wIns/{ttl}__{lt2}.jpg'
        plt.savefig(figDirName)
    plt.savefig('___wIns___.pdf', bbox_inches='tight')
    return

def random_line_marker(have_line_yon='y', marker_size_bos = 's'):
    # have_line_yon take values from: ['y', 'n', 'o']; 'o' means doesn't care.
    # marker_size_bos: ['b',s','o'] for 'big', 'small', 'not care'

    if marker_size_bos=='s':
        mk = ['', 'x','.',',','1','2','3','4','*','+','|','_']
    elif marker_size_bos=='o':
        mk = ['', 'x','.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_']
    elif marker_size_bos=='b':
        mk = ['o','v','^','<','>','s','p','h','H','x','D','d']

    if have_line_yon=='y':
        sty = ['-.','-','--',':']
    elif have_line_yon=='o':
        sty = ['','-.','-','--',':']
    elif have_line_yon=='n':
        sty = ['']

    return np.random.choice(mk)+np.random.choice(sty)

class graphUtils:
    # example usage:
        # graphUtils.example_run()
        # graphUtils.demo_bipt_graph()

    edge_index = example_edge_index = torch.tensor([[0,0,1,1,1,2],[0,1,0,1,2,2]]) # 3 nodes, 6 edges; not symmetric.
    # ---- edge_index <-> A example code ----
    # a = graphUtils.edge_index_to_A(graphUtils.example_edge_index, want_tensor=1)
    # a.to_dense()
    # a = graphUtils.edge_index_to_A(graphUtils.example_edge_index, want_tensor=0)
    # a.todense()

    # # ---- remove_isolated_nodes example code ----
    # edge_index = example_edge_index = torch.tensor([[0,0,1,1,1,4,3],[0,1,0,1,4,4,3]]) # 3 nodes, 6 edges; not symmetric.
    # a = graphUtils.edge_index_to_A(example_edge_index, want_tensor=1)
    # a.to_dense()

    # from torch_geometric.utils import remove_self_loops, to_undirected, to_networkx, negative_sampling, remove_isolated_nodes
    # edge_index2, edge_attr, mask = remove_isolated_nodes(edge_index, edge_attr=None, num_nodes=None)
    # a2 = graphUtils.edge_index_to_A(edge_index2, want_tensor=1)

    # a2.to_dense()

    edge_index_bipt = example_edge_index_bipt = torch.tensor([[0,0,1,1,1,2],[3,4,3,4,5,5]]) # num_nodes=(3,6).
    # ---- bipt graph neg sample example code ----
    # neg_edge_samp = negative_sampling(graphUtils.edge_index_bipt, num_nodes=(3,6), num_neg_samples=20, force_undirected=False)
    # neg_edge_samp, _ = remove_self_loops(neg_edge_samp)

    @staticmethod
    def edge_index_2_edgeset(edge_index):
        edge_index = tonp(edge_index)
        return set(zip(*edge_index)) # {(0,0), (1,4), ... }

    @staticmethod
    def remove_self_loops(edge_index):
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        return edge_index

    @staticmethod
    def add_self_loops(edge_index, num_nodes=None):
        if num_nodes is None:
            num_nodes = int(edge_index.max()+1)
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        return edge_index
    
    @staticmethod
    def edge_index_to_A(edge_index, num_nodes=None, edge_weight=None, want_tensor=0):
        # want_tensor set to 1 or 2 means different output data type: 1=torch.Tensor, 2=torch.sparse tensor; recommend=1

        # example usage
            # from w import graphUtils
            # A = graphUtils.edge_index_to_A(graphUtils.example_edge_index)
            # graphUtils.edge_index_to_A(graphUtils.example_edge_index, want_tensor=2)

        edge_index = tonp(edge_index)
        import scipy.sparse as ssp
        if num_nodes is None:
            num_nodes = edge_index.max()+1
        if edge_weight is None:
            edge_weight = np.ones(edge_index.shape[1])
        edge_weight = tonp(edge_weight) # shape: [N_e] or [N_e, dim]

        def get_A(edge_weight, edge_index):
            if not want_tensor:
                A = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
                return A
            elif want_tensor==1:
                A = torch.sparse_coo_tensor(edge_index, edge_weight, [num_nodes, num_nodes]).coalesce()
                return A
            elif want_tensor==2:
                from torch_sparse import SparseTensor
                edge_index = torch.tensor(edge_index)
                edge_weight = torch.tensor(edge_weight)
                A = SparseTensor(row=edge_index[1], col=edge_index[0], value=edge_weight, sparse_sizes=[num_nodes, num_nodes], is_sorted=True)
                return A

        if len(edge_weight.shape)==1:
            return get_A(edge_weight, edge_index)

        else:
            return [get_A(edge_weight[:,d], edge_index) for d in range(edge_weight.shape[1])]
            # As = []
            # for d in range(edge_weight.shape[1]):
            #     A = ssp.csr_matrix((edge_weight[:,d], (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
            #     As.append(A)
            # return As

    @staticmethod
    def A_to_edge_index(As):
        # input should be torch tensor, not ssp.csr_matrix
        if type(As) is list:
            edge_index = As[0].coalesce().indices()
            edge_weight = np.array([tonp(As[i].coalesce().values()) for i in range(len(As))]).T
        else:
            edge_index = tonp(As.coalesce().indices())
            edge_weight = tonp(As.coalesce().values()).T
        return edge_index, edge_weight

    @staticmethod
    def edge_index_to_dict(edge_index, want_reverse=1):
        l = tonp(edge_index)
        from collections import defaultdict
        node2nei = defaultdict(list)
        for ie in range(edge_index.shape[1]):
            ori, dst = edge_index[:,ie]
            node2nei[ori].append(dst)
            if want_reverse:
                node2nei[dst].append(ori)
        return node2nei

    @staticmethod
    def edge_index_to_sparse_adj(edge_index, num_nodes=None, edge_weight=None):
        # input is torch tensor
        # Related: The way to convert dense matric to edge_index:
            # adj_NN = adj_NN.to_sparse().indices()
        if num_nodes is None:
            num_nodes = int(edge_index.max()+1)
        if edge_weight is None:
            v = [1] * edge_index.shape[1]
        adj = torch.sparse_coo_tensor(edge_index, v, size=(num_nodes, num_nodes), device=edge_index.device).float()
        return adj
    @staticmethod
    def edge_index_to_dense_numpy(edge_index, num_nodes=None, edge_weight=None):
        # output is N*N numpy array
        if num_nodes is None:
            num_nodes = int(edge_index.max()+1)
        if edge_weight is None:
            v = [1] * edge_index.shape[1]
        adj = torch.sparse_coo_tensor(edge_index, v, size=(num_nodes, num_nodes), device=edge_index.device).float().numpy()
        return adj

    @staticmethod
    def normalize_adj(edge_index, num_nodes=None):
        # math:
            # A is without self-loop
            # A_hat = A + I
            # Dii = A_hat.sum(dim=1)
            # A_tilde = D^(-1/2) <dot> A_hat <dot> D^(-1/2)
        if num_nodes is None:
            num_nodes = int(edge_index.max()+1)
        edge_index = graphUtils.remove_self_loops(edge_index)
        edge_index = graphUtils.add_self_loops(edge_index, num_nodes)
        Adj = graphUtils.edge_index_to_sparse_adj(edge_index, num_nodes)
        D_mtxv = torch.sparse.sum(Adj, dim=1).values()
        assert len(D_mtxv)==num_nodes
        D_mtxinv = torch.diag(D_mtxv**(-1/2)).to_sparse()
        A_tilde = torch.sparse.mm(D_mtxinv, torch.sparse.mm(Adj, D_mtxinv))
        return A_tilde.coalesce()
    @staticmethod
    def sparse_power(x, N):
        x0 = x
        assert N>0
        for n in range(N-1):
            x = torch.sparse.mm(x, x0)
        return x.coalesce()
    @staticmethod
    def subgraph(subset, edge_index, edge_attr=None, relabel_nodes=True,
                 num_nodes=None):
        device = edge_index.device
        if isinstance(subset, list) or isinstance(subset, tuple):
            subset = torch.tensor(subset, dtype=torch.long)
        if num_nodes is None:
            num_nodes = int(edge_index.max()+1)
        n_mask = torch.zeros(num_nodes, dtype=torch.bool)
        n_mask[subset] = 1  # convert idx or mask to mask
        if relabel_nodes:
            n_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
            n_idx[subset] = torch.arange(subset.shape[0], device=device)
        mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask] if edge_attr is not None else None
        if relabel_nodes:
            edge_index = n_idx[edge_index]
        return edge_index, edge_attr
    @staticmethod
    def crop_adj_to_subgraph(adj_mtx, subset_idx):
        if isinstance(subset_idx, list) or isinstance(subset_idx, tuple):
            subset_idx = torch.tensor(subset_idx, dtype=torch.long)
        edge_index, edge_attr = adj_mtx.indices(), adj_mtx.values()
        edge_index, edge_attr = graphUtils.subgraph(subset_idx, edge_index, edge_attr, relabel_nodes=True, num_nodes=adj_mtx.shape[0])
        n2 = len(subset_idx)
        adj2 = torch.sparse_coo_tensor(edge_index, edge_attr, size=(n2, n2), device=adj_mtx.device).float()
        return adj2
    @staticmethod
    def edge_index_to_symmetric(edge_index):
        adj = graphUtils.edge_index_to_sparse_adj(edge_index)
        adjT = graphUtils.edge_index_to_sparse_adj(edge_index[[1,0]])
        adj = (adj + adjT).coalesce()
        edge_index = adj.indices()
        return edge_index
    @staticmethod
    def homo_g_to_data(g):

        # x = g.ndata['x']
        # y = g.ndata['y']

        edges_two_tuple = g.edges()
        data = D()
        # data.x = x
        # data.y = y
        data.edge_index = torch.stack(edges_two_tuple)

        return data
    @staticmethod
    def intersect_idx(idx_list):
        N_nodes = max([idx.max()+1 for idx in idx_list])
        mask_list = [graphUtils.idx2mask(idx, N_nodes) for idx in idx_list]
        intersect_mask = graphUtils.intersect_mask(mask_list)
        return graphUtils.mask2idx(intersect_mask)
    @staticmethod
    def intersect_mask(mask_list):
        intersection = mask_list[0]
        for i in range(1,len(mask_list)):
            mask = mask_list[i]
            intersection *= mask
        return intersection
    @staticmethod
    def mask2idx(mask):
        return torch.where(mask==True)[0].to(mask.device)
    @staticmethod
    def idx2mask(idx, N_nodes=None):
        if N_nodes is None:
            N_nodes = idx.max()+1
        mask = torch.tensor([False]*N_nodes, device=idx.device)
        mask[idx] = True
        return mask

    @staticmethod
    def add_As(A1,A2):
        if type(A1) is list:
            return [A1[i] + A2[i] for i in range(len(A1))]
        else:
            return A1+A2
    # ------------- below are demos -------------

    def example_run():
        print('--------- demo normalize ---------')
        edge_index = graphUtils.example_edge_index
        adj = graphUtils.normalize_adj(edge_index)

        print('--------- demo subgraph ---------')
        adj2 = graphUtils.crop_adj_to_subgraph(adj, [0,1])
        
        print('--------- demo to-dense ---------')
        adj2_dense = adj2.to_dense()
        adj2 = adj2_dense.to_sparse()
        
        print('--------- demo A^K ---------')
        adj3 = graphUtils.sparse_power(adj, 3)
        print('--------- demo to-symmetric ---------')
        edge_index2 = graphUtils.edge_index_to_symmetric(edge_index)
        print(edge_index, '\n\n',edge_index2)
        return adj

    def demo_bipt_graph():
        import dgl

        edge_in_query = torch.tensor([0, 0, 1])
        edge_in_asin = torch.tensor([1, 2, 4])
        N_nodes_query = 6
        N_nodes_asin = 7
        dim_feature = 10

        
        # edge_in_query = datas.edge_in_query
        # edge_in_asin = datas.edge_in_asin
        # N_nodes_query = datas.N_nodes_query
        # N_nodes_asin = datas.N_nodes_asin
        # dim_feature = 768

        x_asin = torch.randn(N_nodes_asin, dim_feature)
        x_query = torch.randn(N_nodes_query, dim_feature)
        y_asin = torch.randn(N_nodes_asin, 1)
        y_query = torch.randn(N_nodes_query, 1)

        graph_data = {
            ('query', 'qa', 'asin'): (edge_in_query, edge_in_asin),
            ('asin', 'qareverse', 'query'): (edge_in_asin, edge_in_query),
            }
        num_nodes_dict = {'query': N_nodes_query, 'asin': N_nodes_asin}
        hg = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)


        hg.nodes('asin')  # -> full index tensor

        hg.nodes['asin'].data['x'] = x_asin
        hg.nodes['query'].data['x'] = x_query
        hg.nodes['asin'].data['y'] = y_asin
        hg.nodes['query'].data['y'] = y_query

        g = dgl.to_homogeneous(hg, ndata=['x', 'y'])
        data = graphUtils.homo_g_to_data(g)


        print('g', g, 'data.edge_index', data.edge_index.shape)
        return hg, g, data
