import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.utils import negative_sampling, add_self_loops, train_test_split_edges
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader

import scipy.sparse as ssp
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

def eva_heuristics_v2_dec25(which_heuristic, data, edge_index):
    # which_heuristic support: 'CN', 'AA', 'PPR'
    # 'edge_index' is the thing that need to be evaluated, predict either true or false, by giving a number.

    if 'edge_attr' in data:
        edge_weight = data.edge_attr.view(-1).cpu()
    elif 'edge_weight' in data:
        edge_weight = data.edge_weight.view(-1).cpu()
    else:
        edge_weight = torch.ones(data.edge_index.shape[1], dtype=int).cpu()

    if 'A' not in data:
        tmp_edge_index = data.edge_index.cpu()
        print('check : ', tmp_edge_index.shape, edge_weight.shape)
        data.A = ssp.csr_matrix((edge_weight, (tmp_edge_index[0], tmp_edge_index[1])), shape=(data.num_nodes, data.num_nodes))

    pred_scores, ei = eval(which_heuristic)(data.A, torch.tensor(tonp(edge_index)))
    pred_scores = tonp(pred_scores)

    return pred_scores

def eva_heuristics(args, data, split_edge):
    num_nodes = data.num_nodes
    if 'edge_weight' in data:
        edge_weight = data.edge_weight.view(-1)
    else:
        edge_weight = torch.ones(data.edge_index.shape[1], dtype=int)

    A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])), shape=(num_nodes, num_nodes))

    pos_val_edge, neg_val_edge = get_pos_neg_edges('valid', split_edge, 
                                                   data.edge_index, 
                                                   data.num_nodes)
    pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge, 
                                                     data.edge_index, 
                                                     data.num_nodes)
    pos_val_pred, pos_val_edge = eval(args.use_heuristic)(A, pos_val_edge)
    neg_val_pred, neg_val_edge = eval(args.use_heuristic)(A, neg_val_edge)
    pos_test_pred, pos_test_edge = eval(args.use_heuristic)(A, pos_test_edge)
    neg_test_pred, neg_test_edge = eval(args.use_heuristic)(A, neg_test_edge)

    if args.eval_metric == 'hits':
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'auc':
        val_pred = torch.cat([pos_val_pred, neg_val_pred])
        val_true = torch.cat([torch.ones(pos_val_pred.shape[0], dtype=int), 
                              torch.zeros(neg_val_pred.shape[0], dtype=int)])
        test_pred = torch.cat([pos_test_pred, neg_test_pred])
        test_true = torch.cat([torch.ones(pos_test_pred.shape[0], dtype=int), 
                              torch.zeros(neg_test_pred.shape[0], dtype=int)])
        results = evaluate_auc(val_pred, val_true, test_pred, test_true)

def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()
        if split == 'train':
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.shape[1])
        else:
            neg_edge = split_edge[split]['edge_neg'].t()
        # subsample for pos_edge
        np.random.seed(123)
        num_pos = pos_edge.shape[1]
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        np.random.seed(123)
        num_neg = neg_edge.shape[1]
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.shape[0], 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        np.random.seed(123)
        num_source = source.shape[0]
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.shape[1]
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target), 
                                target_neg.view(-1)])
    return pos_edge, neg_edge

def CN(A, edge_index, batch_size=100000):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.shape[1]), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    return torch.FloatTensor(np.concatenate(scores, 0)), edge_index

def AA(A, edge_index, batch_size=100000):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.shape[1]), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index

def PPR(A, edge_index):
    # The Personalized PageRank heuristic score.
    # Need install fast_pagerank by "pip install fast-pagerank"
    # Too slow for large datasets now.
    from fast_pagerank import pagerank_power
    num_nodes = A.shape[0]
    src_index, sort_indices = torch.sort(edge_index[0])
    dst_index = edge_index[1, sort_indices]
    edge_index = torch.stack([src_index, dst_index])
    #edge_index = edge_index[:, :50]
    scores = []
    visited = set([])
    j = 0
    for i in tqdm(range(edge_index.shape[1])):
        if i < j:
            continue
        src = edge_index[0, i]
        personalize = np.zeros(num_nodes)
        personalize[src] = 1
        ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
        j = i
        while edge_index[0, j] == src:
            j += 1
            if j == edge_index.shape[1]:
                break
        all_dst = edge_index[1, i:j]
        cur_scores = ppr[all_dst]
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index

def tonp(arr):
    if type(arr) is torch.Tensor:
        return arr.detach().cpu().data.numpy()
    else:
        return np.asarray(arr)
