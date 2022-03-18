from collections import defaultdict
import torch
import numpy as np
from torch_sparse import SparseTensor
import copy

from utils import toitem, tonp
from Label_propagation_model.outcome_correlation import general_outcome_correlation_YAG, gen_normalized_adjs


def normalize_adj_v2(edge_index, num_nodes):
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0
    DAD, DA, AD = gen_normalized_adjs(adj, D_isqrt)
    adj = DAD
    return adj

def normalize_adj_v3(edge_index, num_nodes):
    adj = torch.sparse_coo_tensor(edge_index, edge_index[0]*0+1, [num_nodes, num_nodes]).coalesce().float()
    def get_degree_vector(adj):
        _deg = torch.sparse.sum(adj, dim=1).to(torch.float)
        _idx = _deg.indices()
        _v = _deg.values()
        N = toitem(adj.shape[0])
        deg = torch.zeros(N, device=adj.device)
        deg[_idx] = _v
        return deg
    deg = get_degree_vector(adj)
    D_isqrt = deg.pow(-1)
    D_isqrt[D_isqrt == float('inf')] = 0
    diag_e = torch.tensor([np.arange(len(D_isqrt)),np.arange(len(D_isqrt))], device=edge_index.device)
    D_isqrt = torch.sparse_coo_tensor(diag_e, D_isqrt).coalesce().float()
    adj = torch.sparse.mm(D_isqrt, adj)
    return adj

def build_edge_adj(edge_index):
    # edge_adj is the 'edge_index_of_edge_index'
    # this function return the "edge_index of edges". If two edges share a node, then they are connected; other wise they are not. The ID of the edges are dedicated by the input edge_index.
    edge_adj = [[i,i] for i in range(edge_index.shape[1])]  # add self loop
    node2edgeset = get_node2edge(edge_index)
    N_nodes = edge_index.max()+1
    for ino in range(N_nodes):
        edge_set = node2edgeset[ino]
        edge_mix = mutual_intermix(edge_set)
        edge_adj.extend(edge_mix)
    edge_adj = torch.tensor(edge_adj, device=edge_index.device).t()
    return edge_adj, node2edgeset

def run_logitLP(edge_index, LP_device, alpha, num_propagations,
    pos_train_pred, pos_valid_pred, pos_test_pred, neg_train_pred, neg_valid_pred, neg_test_pred):
    # input:
        # edge_logits: logits for all edges (to be computed again using 'h')
        # A matrix (original)

    # ---- construct G and Y0
    train_val_test_nums = len(pos_train_pred), len(pos_valid_pred), len(pos_test_pred), len(neg_train_pred), len(neg_valid_pred), len(neg_test_pred)
    logits = torch.cat([pos_train_pred, pos_valid_pred, pos_test_pred, neg_train_pred, neg_valid_pred, neg_test_pred]) # shape: [N_edges]
    Y0 = torch.sigmoid(logits.reshape(-1,1))
    G = torch.zeros(Y0.shape, device=Y0.device)
    G[:train_val_test_nums[0]] += 1
    G[train_val_test_nums[0]:sum(train_val_test_nums[:3])] += 0.5

    # ---- construct edge-graph-A
    edge_adj, node2edgeset = build_edge_adj(edge_index)
    adj = normalize_adj_v2(edge_adj, len(logits))

    # ---- LP step ----
    out = general_outcome_correlation_YAG(Y0, adj, G, alpha, num_propagations, device=LP_device)

    # ---- obtain logits from embedding
    edge_logits = invsigmoid(out.reshape(-1))
    pos_train_pred, pos_valid_pred, pos_test_pred, neg_train_pred, neg_valid_pred, neg_test_pred = separate_logits(edge_logits, train_val_test_nums)
    return pos_train_pred, pos_valid_pred, pos_test_pred, neg_train_pred, neg_valid_pred, neg_test_pred

def run_embLP(embs, edge_index, LP_device, alpha, num_propagations,
        pos_train_edge, pos_valid_edge, pos_test_edge, neg_train_edge, neg_valid_edge, neg_test_edge):
    # ---- construct G and Y0
    train_val_test_nums = len(pos_train_edge), len(pos_valid_edge), len(pos_test_edge), len(neg_train_edge), len(neg_valid_edge), len(neg_test_edge)
    edges = torch.cat([pos_train_edge, pos_valid_edge, pos_test_edge, neg_train_edge, neg_valid_edge, neg_test_edge]) # shape: [N_edges, 2]

    edge_embs = torch.zeros([ len(edges), embs.shape[1]*2 ], device=edge_index.device).float()
    for ie in range(len(edges)):
        src, dst = edges[ie]
        edge_embs[ie] = torch.cat([embs[src],embs[dst]])

    # ---- construct edge-graph-A
    edge_adj, node2edgeset = build_edge_adj(edge_index)
    adj = normalize_adj_v2(edge_adj, len(edges))

    # ---- LP step ----
    Y0 = edge_embs
    G = Y0.clone()
    out = general_outcome_correlation_YAG(Y0, adj, G, alpha, num_propagations, device=LP_device)

    # ---- obtain logits from embedding
    out = out.view(len(edges), 2, embs.shape[1])
    edge_logits = (out[:,0,:] * out[:,1,:]).sum(axis=1)
    
    pos_train_pred, pos_valid_pred, pos_test_pred, neg_train_pred, neg_valid_pred, neg_test_pred = separate_logits(edge_logits, train_val_test_nums)
    return pos_train_pred, pos_valid_pred, pos_test_pred, neg_train_pred, neg_valid_pred, neg_test_pred

def run_xmcLP(edge_index, num_nodes, LP_device, alpha, num_propagations,
        pos_train_pred, pos_valid_pred, pos_test_pred, neg_train_pred, neg_valid_pred, neg_test_pred,
        pos_train_edge, pos_valid_edge, pos_test_edge, neg_train_edge, neg_valid_edge, neg_test_edge):
    # input:
        # edge_logits: logits for all edges (to be computed again using 'h')
        # A matrix (original)
    # ---- construct G and Y0
    train_val_test_nums_O = len(pos_train_pred), len(pos_valid_pred), len(pos_test_pred), len(neg_train_pred), len(neg_valid_pred), len(neg_test_pred)
    edges_O = torch.cat([pos_train_edge, pos_valid_edge, pos_test_edge, neg_train_edge, neg_valid_edge, neg_test_edge]).to(LP_device) # shape: [N_edges, 2]
    logits_O = torch.cat([pos_train_pred, pos_valid_pred, pos_test_pred, neg_train_pred, neg_valid_pred, neg_test_pred]).to(LP_device) # shape: [N_edges]

    def remove_duplicate(edges_O, logits_O, train_val_test_nums_O):
        # inputs are tensors on cpu/gpu
        train_val_test_nums_O = list(copy.copy(train_val_test_nums_O))
        edge2loc = dict()
        edges, logits, train_val_test_nums, duplicate_pos = [],[], [], []
        duplicate_mask = np.array([False]*len(logits_O))
        landmark = 0
        cnt = -1
        for i,e in enumerate([tuple(x) for x in tonp(edges_O)]):
            cnt += 1
            if e not in edge2loc:
                edge2loc[e] = i
                edges.append(e)
                logits.append(logits_O[i].clone())
            else:
                duplicate_mask[i] = True
                duplicate_pos.append(edge2loc[e])
            if cnt+1 == train_val_test_nums_O[0]:
                train_val_test_nums.append(len(edges) - landmark)
                landmark = len(edges)
                train_val_test_nums_O.pop(0)
                cnt = -1

        edges = torch.tensor(edges).to(edges_O.device)
        logits = torch.stack(logits)
        return edges, logits, train_val_test_nums, duplicate_mask, duplicate_pos

    edges, logits, train_val_test_nums, duplicate_mask, duplicate_pos = remove_duplicate(edges_O, logits_O, train_val_test_nums_O)

    num_nodes = toitem(num_nodes)
    Gvalues = torch.zeros(sum(train_val_test_nums), device=LP_device)
    Gvalues[:train_val_test_nums[0]] += 1
    Gvalues[train_val_test_nums[0]:sum(train_val_test_nums[:3])] += 1

    Y0 = torch.sparse_coo_tensor(edges.T, torch.sigmoid(logits), [num_nodes, num_nodes]).coalesce().to(LP_device)
    G = torch.sparse_coo_tensor(edges.T, Gvalues, [num_nodes, num_nodes]).coalesce().to(LP_device)

    # ---- construct edge-graph-A
    adj = normalize_adj_v3(edge_index.to(LP_device), num_nodes)

    # ---- LP step ----
    out = general_outcome_correlation_YAG(Y0, adj, G, alpha, num_propagations, device=LP_device, use_sparse_mult=True)

    # ---- obtain logits from embedding
    edges = [tuple(x) for x in tonp(edges)]
    edge_logits = invsigmoid(torch.stack([out[e] for e in edges]))

    def add_duplicate(edge_logits, duplicate_mask, duplicate_pos):
        edge_logits_O = torch.zeros(duplicate_mask.shape, device=edge_logits.device)
        edge_logits_O[~duplicate_mask] = edge_logits
        edge_logits_O[duplicate_mask] = edge_logits[duplicate_pos]
        return edge_logits_O

    edge_logits_O = add_duplicate(edge_logits, duplicate_mask, duplicate_pos)
    pos_train_pred, pos_valid_pred, pos_test_pred, neg_train_pred, neg_valid_pred, neg_test_pred = separate_logits(edge_logits_O, train_val_test_nums_O)
    return pos_train_pred, pos_valid_pred, pos_test_pred, neg_train_pred, neg_valid_pred, neg_test_pred

def separate_logits(edge_logits, train_val_test_nums):
    acc = np.array(train_val_test_nums)
    acc[1:] += train_val_test_nums[0]
    acc[2:] += train_val_test_nums[1]
    acc[3:] += train_val_test_nums[2]
    acc[4:] += train_val_test_nums[3]
    acc[5:] += train_val_test_nums[4]

    pos_train_pred = edge_logits[:acc[0]]
    pos_valid_pred = edge_logits[acc[0]:acc[1]]
    pos_test_pred = edge_logits[acc[1]:acc[2]]
    neg_train_pred = edge_logits[acc[2]:acc[3]]
    neg_valid_pred = edge_logits[acc[3]:acc[4]]
    neg_test_pred = edge_logits[acc[4]:acc[5]]
    return pos_train_pred, pos_valid_pred, pos_test_pred, neg_train_pred, neg_valid_pred, neg_test_pred

def invsigmoid(y):
    eps = 1e-9
    return - torch.log(1/(y+eps)-1)

def mutual_intermix(edge_set):  
    # input a set of edge_id (that connects to a same node), output the connectivity of these edges
    edge_list = list(edge_set)
    edge_mix = []
    for ie1 in range(len(edge_list)):
        for ie2 in range(ie1+1, len(edge_list)):
            edge_mix.append([ie1,ie2])
            edge_mix.append([ie2,ie1])
    return edge_mix

def get_node2edge(edge_index):
    # the returned dict is composed of purely integers, no torch.tensor.
    node2edge = defaultdict(set)
    for ie in range(edge_index.shape[1]):
        e = edge_index[:,ie]
        node2edge[int(e[0])].add(ie)
        node2edge[int(e[1])].add(ie)
    return node2edge
