import argparse
import time
import torch
import os
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_sparse import coalesce, SparseTensor
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from Link_prediction_model.logger import Logger
from Link_prediction_model.model import BaseModel
from Link_prediction_model.utils import gcn_normalization, adj_normalization
import numpy as np
from torch_geometric.utils import to_undirected

from utils import down_sample_graph_with_node_perm, graphUtils, target_seeded_by_source, cal_union, graph_analyze, tonp, random_mask, init_split_edge_unified_impl, plot_dist




def argument():

    _data_name = [
                    'betr~UK,AU',
                    'betr~UK,DE',  # ['SG', 'DE', 'FR', 'NL', 'SE', 'UK']
                    'betr~UK,FR',
                    'betr~UK,NL',
                    'betr~UK,PL',
                    'betr-AU',      # passed, bkup
                    'betr-UK2AU',   # passed, bkup

                    'ogbl-citation2',
                    'ogbl-collab',

                    ][-2]

    _eval_metric = [
                    'hits', 'mrr', 
                    'recall_my@0.8', 'recall_my@1', 'recall_my@1.25', 'recall_my@0'
                    ][4]

    _encoder = ['SAGE',
                # 'GCN',
                'MLP',
                'CN',   # heuristics baseline (common neighbors)
                'AA',   # heuristics baseline (academic adar)
                'PPR',  # heuristics baseline (personalized page rank)
                ][0]

    _edge_lp_mode = ['emb', 'logit', 'xmc', '' ][1]
    _LP_device = ['cpu', 'cuda:4'][1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--public_data_convert_overlapped_subgraph', type=bool, default=True)
    parser.add_argument('--transfer_setting', type=str, default='i2t', choices=['t2t', 'u2t', 'i2t', 'u', 'i', ''])
    parser.add_argument('--linkpred_baseline', type=str, default='', choices=['', 'EGI', 'DGI'])
    parser.add_argument('--edge_lp_mode', type=str, default=_edge_lp_mode)
    parser.add_argument('--ELP_alpha', type=str, default=0.995)
    parser.add_argument('--num_propagations', type=str, default=5)
    parser.add_argument('--LP_device', type=str, default=_LP_device)
    parser.add_argument('--exp_on_cold_edge', type=bool, default=False)
    parser.add_argument('--encoder', type=str, default=_encoder)
    parser.add_argument('--predictor', type=str, default='DOT', choices=['MLP', 'DOT'])
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss_func', type=str, default='ce_loss', choices=['AUC', 'ce_loss', 'log_rank_loss', 'info_nce_loss'])
    parser.add_argument('--neg_sampler', type=str, default='global')
    parser.add_argument('--data_name', type=str, default=_data_name)
    parser.add_argument('--data_path', type=str, default='dataset')
    parser.add_argument('--eval_metric', type=str, default=_eval_metric)
    parser.add_argument('--res_dir', type=str, default='')
    parser.add_argument('--pretrain_emb', type=str, default='')
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--mlp_num_layers', type=int, default=2)
    parser.add_argument('--emb_hidden_channels', type=int, default=256)
    parser.add_argument('--gnn_hidden_channels', type=int, default=256)
    parser.add_argument('--mlp_hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--grad_clip_norm', type=float, default=2.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--num_neg', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--year', type=int, default=2010)
    parser.add_argument('--linkpred_device', type=int, default=1)
    parser.add_argument('--use_node_feats', type=str2bool, default=False)
    parser.add_argument('--use_coalesce', type=str2bool, default=False)
    parser.add_argument('--train_node_emb', type=str2bool, default=True)
    parser.add_argument('--train_on_subgraph', type=str2bool, default=True)
    parser.add_argument('--use_valedges_as_input', type=str2bool, default=True)
    parser.add_argument('--eval_last_best', type=str2bool, default=True)

    args = parser.parse_args()

    def post_process(args):
        if args.linkpred_baseline in ['EGI', 'DGI']:
            args.encoder='MLP'
            args.use_node_feats = True
        return args
    args = post_process(args)
    return args



def get_data_split_edge_citation2(data, args):
    # node has year info.
    # t2t: graph = {node year >= lo} (lo <= node <= hi is partial node, partial edge),   test split = {n1,n2 >= hi}
    # u2t: graph = {all nodes}
    # i2t: graph = {node year >= lo} (full edges)

    data.node_year = data.node_year.view(-1)    # [edge_year.min(), edge_year.max(), np.median(edge_year)] = [tensor(1901), tensor(2019), 2006.0]; nodes before 1960 are very few.
    if 0:
        plot_dist(data.node_year, ttl = 'node_year-citation2', bins = 100, saveFig_fname = 'node_year-citation2')

    drop_rate = 0.1
    data = down_sample_graph_with_node_perm(data, drop_rate=drop_rate)

    lo = 2014
    hi = 2016
    drop_shared_edge_prob = 0.8
    
    # ---- convert to t/u/i data ----
    if args.transfer_setting=='t2t':
        # ---- convert to t2t data ----
        node_idx_target = torch.where(data.node_year >= lo)[0]
        data = down_sample_graph_with_node_perm(data, perm=node_idx_target)
        # ---- edge sparsify ----
        shared_node_mask = data.node_year <= hi
        shared_edge_mask = shared_node_mask[data.edge_index[0]] & shared_node_mask[data.edge_index[1]]
        drop_shared_edge_mask = shared_edge_mask & torch.tensor(random_mask(len(shared_edge_mask), drop_shared_edge_prob))
        data.edge_index = data.edge_index[:,~drop_shared_edge_mask]

    elif args.transfer_setting=='u2t':
        data = data # do nothing

    elif args.transfer_setting=='i2t':
        node_idx_target = torch.where(data.node_year >= lo)[0]
        data = down_sample_graph_with_node_perm(data, perm=node_idx_target)

    elif args.transfer_setting=='s':
        node_idx_target = torch.where(data.node_year <= hi)[0]
        data = down_sample_graph_with_node_perm(data, perm=node_idx_target)

    elif args.transfer_setting=='i':
        node_idx_target = torch.where((data.node_year <= hi) & (data.node_year >= lo))[0]
        data = down_sample_graph_with_node_perm(data, perm=node_idx_target)

    # ---- split edge ----
    if args.exp_on_cold_edge:
        degs_o, degs_d = graph_analyze(data.num_nodes, data.edge_index)
        degs_o, degs_d = tonp(degs_o), tonp(degs_d)
        cold_edge_mask = degs_o[data.edge_index[0,:]] + degs_d[data.edge_index[1,:]] <= 3
        data.is_unique_in_targetG_edge_mask = cold_edge_mask
    else:
        data.is_unique_in_targetG_mask = data.node_year >= hi
    split_edge = init_split_edge_unified_impl(data, is_bipt=False)
    data.adj_t = graphUtils.edge_index_to_A(data.edge_index, want_tensor=2)
    
    return data, split_edge

def get_data_split_edge_collab(data, args):
    # edge has year info.
    # t2t: graph = {edge year >= lo},   test split = {e >= hi}
    # u2t: graph = {all nodes}
    # i2t: graph = {edge year >= lo} -> node mask, then all edges on this node set
    import pandas as pd
    fyear = 'dataset/ogbl_collab/raw/edge_year.csv.gz'
    data.edge_year = pd.read_csv(fyear, compression='gzip', header = None).values.reshape(-1).astype(np.int64) # (num_edge,);   [edge_year.min(), edge_year.max(), np.median(edge_year)] = [1963, 2017, 2011.0]
    # ---- below operation is referred from .../python3.6/site-packages/ogb/io/read_graph_raw.py
    data.edge_year = torch.tensor(np.repeat(data.edge_year, 2))

    if 0:
        plot_dist(data.edge_year, ttl = 'edge_year', bins = 100, saveFig_fname = 'edge_year')

    drop_rate=0.1
    data = down_sample_graph_with_node_perm(data, drop_rate=drop_rate)

    lo = 2015
    hi = 2016

    # ---- convert to t/u/i data ----
    if args.transfer_setting=='t2t':
        target_edge_mask = data.edge_year >= lo
        data.edge_index = data.edge_index[:,target_edge_mask]
        data.edge_weight = data.edge_weight[target_edge_mask]
        data.edge_year = data.edge_year[target_edge_mask]

    elif args.transfer_setting=='u2t':
        data = data

    elif args.transfer_setting=='i2t':
        target_edge_mask_tmp = data.edge_year >= lo
        node_idx = sorted(set(tonp(data.edge_index[:,target_edge_mask_tmp].reshape(-1)).tolist()))
        data = down_sample_graph_with_node_perm(data, perm=node_idx)

    elif args.transfer_setting=='s':
        target_edge_mask_tmp = data.edge_year <= hi
        node_idx = sorted(set(tonp(data.edge_index[:,target_edge_mask_tmp].reshape(-1)).tolist()))
        data = down_sample_graph_with_node_perm(data, perm=node_idx)

    elif args.transfer_setting=='i':
        target_edge_mask_tmp = (lo <= data.edge_year) & (data.edge_year <= hi)
        node_idx = sorted(set(tonp(data.edge_index[:,target_edge_mask_tmp].reshape(-1)).tolist()))
        data = down_sample_graph_with_node_perm(data, perm=node_idx)


    # ---- split edge ----
    data.is_unique_in_targetG_edge_mask = data.edge_year >= hi
    split_edge = init_split_edge_unified_impl(data, is_bipt=False)
    data.adj_t = graphUtils.edge_index_to_A(data.edge_index, want_tensor=2)

    return data, split_edge

class trainer:
    def __init__(self, args, which_run):
        self.args = args

    def main(self):
        args = self.args
        device = f'cuda:{args.linkpred_device}' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        
        # ---- when open source, delete the 3 lines below, and cancel the if-else below.
        if 'betr' in args.data_name: # this means loading amazon proprietary datasets
            from _NOT_release_load_betr_proprietary_data import get_betr_paper_datasets
            data, split_edge = get_betr_paper_datasets(args)

        else:
            dataset = PygLinkPropPredDataset(name=args.data_name, root=args.data_path)
            data = dataset[0]
            if args.public_data_convert_overlapped_subgraph:
                if args.data_name=='ogbl-citation2':
                    data, split_edge = get_data_split_edge_citation2(data, args)
                elif args.data_name=='ogbl-collab':
                    data, split_edge = get_data_split_edge_collab(data, args)

            else:
                data = T.ToSparseTensor()(data)
                split_edge = dataset.get_edge_split()

        if hasattr(data, 'edge_weight'):
            if data.edge_weight is not None:
                data.edge_weight = data.edge_weight.view(-1).to(torch.float)

        row, col, _ = data.adj_t.coo()
        data.edge_index = torch.stack([col, row], dim=0)

        if hasattr(data, 'num_features'):
            num_node_feats = data.num_features
        else:
            num_node_feats = 0

        if hasattr(data, 'num_nodes'):
            num_nodes = data.num_nodes
        else:
            num_nodes = data.adj_t.size(0)

        if hasattr(data, 'x'):
            if data.x is not None:
                data.x = data.x.to(torch.float)

        if args.transfer_setting == '':
            if args.data_name == 'ogbl-citation2':
                data.adj_t = data.adj_t.to_symmetric()
            if args.data_name == 'ogbl-collab':
                # only train edges after specific year
                if args.year > 0 and hasattr(data, 'edge_year'):
                    selected_year_index = torch.reshape(
                        (split_edge['train']['year'] >= args.year).nonzero(as_tuple=False), (-1,))
                    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
                    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
                    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
                    train_edge_index = split_edge['train']['edge'].t()
                    # create adjacency matrix
                    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
                    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
                    data.adj_t = SparseTensor(row=new_edge_index[0],
                                            col=new_edge_index[1],
                                            value=new_edge_weight.to(torch.float32))
                    data.edge_index = new_edge_index

                # Use training + validation edges
                if args.use_valedges_as_input:
                    full_edge_index = torch.cat([split_edge['valid']['edge'].t(), split_edge['train']['edge'].t()], dim=-1)
                    full_edge_weight = torch.cat([split_edge['train']['weight'], split_edge['valid']['weight']], dim=-1)
                    # create adjacency matrix
                    new_edges = to_undirected(full_edge_index, full_edge_weight, reduce='add')
                    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
                    data.adj_t = SparseTensor(row=new_edge_index[0],
                                            col=new_edge_index[1],
                                            value=new_edge_weight.to(torch.float32))
                    data.edge_index = new_edge_index

                    if args.use_coalesce:
                        full_edge_index, full_edge_weight = coalesce(full_edge_index, full_edge_weight, num_nodes, num_nodes)

                    # edge weight normalization
                    split_edge['train']['edge'] = full_edge_index.t()
                    deg = data.adj_t.sum(dim=1).to(torch.float)
                    deg_inv_sqrt = deg.pow(-0.5)
                    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                    split_edge['train']['weight'] = deg_inv_sqrt[full_edge_index[0]] * full_edge_weight * deg_inv_sqrt[
                        full_edge_index[1]]

                # reindex node ids on sub-graph
                if args.train_on_subgraph:
                    # extract involved nodes
                    row, col, edge_weight = data.adj_t.coo()
                    subset = set(row.tolist()).union(set(col.tolist()))
                    subset, _ = torch.sort(torch.tensor(list(subset)))
                    # For unseen node we set its index as -1
                    n_idx = torch.zeros(num_nodes, dtype=torch.long) - 1
                    n_idx[subset] = torch.arange(subset.size(0))
                    # Reindex edge_index, adj_t, num_nodes
                    data.edge_index = n_idx[data.edge_index]
                    data.adj_t = SparseTensor(row=n_idx[row], col=n_idx[col], value=edge_weight)
                    num_nodes = subset.size(0)
                    if hasattr(data, 'x'):
                        if data.x is not None:
                            data.x = data.x[subset]
                    # Reindex train valid test edges
                    split_edge['train']['edge'] = n_idx[split_edge['train']['edge']]
                    split_edge['valid']['edge'] = n_idx[split_edge['valid']['edge']]
                    split_edge['valid']['edge_neg'] = n_idx[split_edge['valid']['edge_neg']]
                    split_edge['test']['edge'] = n_idx[split_edge['test']['edge']]
                    split_edge['test']['edge_neg'] = n_idx[split_edge['test']['edge_neg']]



        data = data.to(device)

        if args.encoder.upper() == 'GCN':
            # Pre-compute GCN normalization.
            data.adj_t = gcn_normalization(data.adj_t)

        if args.encoder.upper() == 'WSAGE':
            data.adj_t = adj_normalization(data.adj_t)

        if args.encoder.upper() == 'TRANSFORMER':
            row, col, edge_weight = data.adj_t.coo()
            data.adj_t = SparseTensor(row=row, col=col)

        model = BaseModel(
            args, data,
            lr=args.lr,
            dropout=args.dropout,
            grad_clip_norm=args.grad_clip_norm,
            gnn_num_layers=args.gnn_num_layers,
            mlp_num_layers=args.mlp_num_layers,
            emb_hidden_channels=args.emb_hidden_channels,
            gnn_hidden_channels=args.gnn_hidden_channels,
            mlp_hidden_channels=args.mlp_hidden_channels,
            num_nodes=num_nodes,
            num_node_feats=num_node_feats,
            gnn_encoder_name=args.encoder,
            predictor_name=args.predictor,
            loss_func=args.loss_func,
            optimizer_name=args.optimizer,
            device=device,
            use_node_feats=args.use_node_feats,
            train_node_emb=args.train_node_emb,
            pretrain_emb=args.pretrain_emb)


        total_params = sum(p.numel() for param in model.para_list for p in param)
        total_params_print = f'Total number of model parameters is {total_params}'
        print(total_params_print)

        evaluator = Evaluator(name=args.data_name) if 'ogbl' in args.data_name else None

        if args.eval_metric == 'hits':
            loggers = {
                'Hits@20': Logger(args.runs, args),
                'Hits@50': Logger(args.runs, args),
                'Hits@100': Logger(args.runs, args),
            }
        elif args.eval_metric == 'mrr':
            loggers = {
                'MRR': Logger(args.runs, args),
            }
        elif 'recall_my' in args.eval_metric:
            loggers = {
                'recall@100%': Logger(args.runs, args),
            }

        for run in range(args.runs):
            model.param_init()
            start_time = time.time()
            for epoch in range(1, 1 + args.epochs):
                if epoch==1 and args.linkpred_baseline in ['EGI', 'DGI']:
                    from Link_prediction_baseline.run_airport import gen_baseline_embs
                    model.embs = gen_baseline_embs(data.edge_index, data.x, args.linkpred_baseline)
                    
                loss = model.train(data, split_edge,
                                batch_size=args.batch_size,
                                neg_sampler_name=args.neg_sampler,
                                num_neg=args.num_neg)
                if epoch % args.eval_steps == 0:
                    results = model.test(data, split_edge,
                                        batch_size=args.batch_size,
                                        evaluator=evaluator,
                                        eval_metric=args.eval_metric,
                                        )
                    for key, result in results.items():
                        loggers[key].add_result(run, result)
                    if epoch % args.log_steps == 0:
                        spent_time = time.time() - start_time
                        for key, result in results.items():
                            if 'recall_my' not in args.eval_metric:
                                valid_res, test_res = result
                                to_print = (f'Run: {run + 1:02d}, '
                                            f'Epoch: {epoch:02d}, '
                                            f'Loss: {loss:.4f}, '
                                            f'Valid: {100 * valid_res:.2f}%, '
                                            f'Test: {100 * test_res:.2f}%')
                            else: 
                                to_print = (f'Run: {run + 1:02d}, '
                                            f'Epoch: {epoch:02d}, '
                                            f'Loss: {loss:.4f}, \n'
                                            f'result = {result}')


                            print(key)
                            print(to_print)
                        print('---')
                        print(
                            f'Training Time Per Epoch: {spent_time / args.eval_steps: .4f} s')
                        print('---')
                        start_time = time.time()
        return loss

def analyze_data(data):
    print(f'\n\n -------------\n nodes: {data.num_nodes}\n  edge: {data.edge_index.shape[1]}')
    degs_o, degs_d = graph_analyze(data.num_nodes, data.edge_index)
    print(f' mean deg = {np.mean(degs_o), np.mean(degs_d)}')
    print(f' median deg = {np.median(degs_o), np.median(degs_d)} \n ----------\n\n')
    return


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    args = argument()
    trnr = trainer(args, 0)
    trnr.main()


