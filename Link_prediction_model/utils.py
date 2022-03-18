# -*- coding: utf-8 -*-
import torch
import numpy as np
from .negative_sample import global_neg_sample, global_perm_neg_sample, local_neg_sample
from utils import cal_recall

def get_pos_neg_edges(split, split_edge, edge_index=None, num_nodes=None, neg_sampler_name=None, num_neg=None):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge']
    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        pos_edge = torch.stack([source, target]).t()

    if split == 'train':
        if neg_sampler_name == 'local':
            neg_edge = local_neg_sample(
                pos_edge,
                num_nodes=num_nodes,
                num_neg=num_neg)
        elif neg_sampler_name == 'global':  # pos_edge: [720132, 2] ; neg_edge: [720132, 3, 2]
            neg_edge = global_neg_sample(
                edge_index,
                num_nodes=num_nodes,
                num_samples=pos_edge.size(0),
                num_neg=num_neg)
        else:
            neg_edge = global_perm_neg_sample(
                edge_index,
                num_nodes=num_nodes,
                num_samples=pos_edge.size(0),
                num_neg=num_neg)
    else:
        if 'edge' in split_edge['train']:
            neg_edge = split_edge[split]['edge_neg']
        elif 'source_node' in split_edge['train']:
            target_neg = split_edge[split]['target_node_neg']
            neg_per_target = target_neg.size(1)
            neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                    target_neg.view(-1)]).t()
    return pos_edge, neg_edge

def evaluate_hits(evaluator, pos_val_pred, neg_val_pred,
                  pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results

def evaluate_mrr(evaluator, pos_val_pred, neg_val_pred,
                 pos_test_pred, neg_test_pred):
    # neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_val_pred = neg_val_pred.view(neg_val_pred.shape[0], -1)
    # neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(neg_test_pred.shape[0], -1)
    results = {}
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (valid_mrr, test_mrr)

    return results

def evaluate_recall_my(pos_train_pred, neg_train_pred,
                    pos_val_pred, neg_val_pred,
                    pos_test_pred, neg_test_pred, topk=None):
    results = {}
    recall_train = cal_recall(pos_train_pred, neg_train_pred, topk=topk)
    recall_valid = cal_recall(pos_val_pred, neg_val_pred, topk=topk)
    recall_test = cal_recall(pos_test_pred, neg_test_pred, topk=topk)
    results['recall@100%'] = (recall_train, recall_valid, recall_test)

    return results

def gcn_normalization(adj_t):
    adj_t = adj_t.set_diag()
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
    return adj_t

def adj_normalization(adj_t):
    deg = adj_t.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj_t = deg_inv_sqrt.view(-1, 1) * adj_t
    return adj_t

def generate_neg_dist_table(num_nodes, adj_t, power=0.75, table_size=1e8):
    table_size = int(table_size)
    adj_t = adj_t.set_diag()
    node_degree = adj_t.sum(dim=1).to(torch.float)
    node_degree = node_degree.pow(power)

    norm = float((node_degree).sum())  # float is faster than tensor when visited
    node_degree = node_degree.tolist()  # list has fastest visit speed
    sample_table = np.zeros(table_size, dtype=np.int32)
    p = 0
    i = 0
    for j in range(num_nodes):
        p += node_degree[j] / norm
        while i < table_size and float(i) / float(table_size) < p:
            sample_table[i] = j
            i += 1
    sample_table = torch.from_numpy(sample_table)
    return sample_table
