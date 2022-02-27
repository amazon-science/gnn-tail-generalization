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
                print('inequal!  ',ino)


    return degs_ori, degs_dst


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
    # score = score.view(-1)
    # _, indices = th.sort(score, descending=True)
    # where_p = th.where(indices < num_p)[0]
    # sum_p = where_p.sum()
    # normed_score =  ( sum_p - (num_p-1)*num_p/2 ) / ( (2*N-num_p)*num_p/2 ) * 100
    num_neg_per_pos = len(neg_score)//len(pos_score)
    drop_end = num_neg_per_pos*len(pos_score)
    pos_neg = torch.cat([pos_score.reshape(-1,1), neg_score[:drop_end].reshape(len(pos_score), num_neg_per_pos)], dim=1)
    # print('mrr matrix shape: ', pos_neg.shape)



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
    # plt.scatter(xe[data.train_mask,0],xe[data.train_mask,1], 'x')
    data = data.to(data.device)
    return



def run_pureLP(res): #-p3
    from LP.outcome_correlation import general_outcome_correlation,label_propagation,double_correlation_autoscale, double_correlation_fixed,gen_normalized_adjs,process_adj
    res.lpStep_alpha = 0.5
    res.lpStep_num_propagations = 50
    adj, D_isqrt = process_adj(res.data)
    DAD, DA, AD = gen_normalized_adjs(adj, D_isqrt)
    res.adj = DAD
    alpha_term = 1
    lp_dict = {
            'train_only': True,
            # 'alpha1': res.lpStep.alpha1, 
            # 'alpha2': res.lpStep.alpha2,
            # 'A1': eval(res.lpStep.A1),
            # 'A2': eval(res.lpStep.A2),
            # 'num_propagations1': res.lpStep.num_propagations1,
            # 'num_propagations2': res.lpStep.num_propagations2,
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
    # res.out_LP = label_propagation(res.data, res.split_idx, **lp_dict)
    # lp_fn = [double_correlation_autoscale, double_correlation_fixed][0]
    # res.out_LP = lp_fn(res.data, res.emb, res.split_idx, **lp_dict)
    
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
    # plt.show()
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
    example_edge_index = torch.tensor([[0,0,1,1,1,2],[0,1,0,1,2,2]]) # 3 nodes, 6 edges; not symmetric.
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
    def edge_index_to_sparse_adj(edge_index, num_nodes=None, edge_weight=None):
        if num_nodes is None:
            num_nodes = int(edge_index.max()+1)
        if edge_weight is None:
            v = [1] * edge_index.shape[1]
        adj = torch.sparse_coo_tensor(edge_index, v, size=(num_nodes, num_nodes), device=edge_index.device).float()
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

