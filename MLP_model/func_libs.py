# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import numpy.linalg as la
import os
from time import time as timer
import time
import pickle
import copy
import gc
import numpy.linalg as la
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class C: pass
class D: pass
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
        nn_list.extend([nn.Linear(neurons[i], neurons[i+1], bias=bias), norm, nn.Dropout(dropout)])
    
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

