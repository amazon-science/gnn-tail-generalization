# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .outcome_correlation import *
from .diffusion_feature import *
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor, matmul
from typing import Callable, Optional

class LabelPropagation_Adj(nn.Module):
    def __init__(self, args, data, train_mask):
        super().__init__()
        self.train_cnt = 0
        self.args = args
        self.num_layers = args.num_layers
        self.alpha = args.lpStep.alpha
        self.num_classes = args.num_classes
        self.num_nodes = data.num_nodes
        self.edge_index = data.edge_index
        self.train_mask = train_mask
        self.preStep = PreStep(args)
        self.midStep = None
        self.lpStep = None
        self.embs_step1 = None
        self.x_after_step2 = None
        self.data_cpu = copy.deepcopy(data).to('cpu')
        self.data = data

    def train_net(self, input_dict):
        # only complete ONE-TIME backprop/update for all nodes
        self.train_cnt += 1
        device, split_masks = input_dict['device'], input_dict['split_masks']
        if self.embs_step1 is None: # only preprocess ONCE; has to be on cpu
            self.embs_step1 = self.preStep(self.data_cpu).to(device)

        data = self.data_cpu.to(device)
        x, y = data.x, data.y
        loss_op = input_dict['loss_op']
        train_mask = split_masks['train']

        if self.midStep is None:
            self.midStep = MidStep(self.args, self.embs_step1, self.data).to(device)
            self.optimizer = torch.optim.Adam(self.midStep.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        if self.lpStep is None:
            self.lpStep = LPStep(self.args, data, split_masks)

        self.x_after_step2, train_loss = self.midStep.train_forward(self.embs_step1, y, self.optimizer, loss_op, split_masks) # only place that require opt
        
        if self.train_cnt>20:
            print()
            acc = cal_acc_logits(self.x_after_step2[split_masks['test']], data.y[split_masks['test']])

        self.out = self.lpStep(self.x_after_step2, data)
        self.out,y,train_mask = to_device([self.out,y,train_mask], 'cpu')
        total_correct = int(self.out[train_mask].argmax(dim=-1).eq(y[train_mask]).sum())
        train_acc = total_correct / int(train_mask.sum())
        return train_loss, train_acc

    def inference(self, input_dict):
        return self.out

    @torch.no_grad()
    def forward_backup(
        self, y: Tensor, edge_index: Adj, mask: Optional[Tensor] = None,
        edge_weight: OptTensor = None,
        post_step: Callable = lambda y: y.clamp_(0., 1.)
    ) -> Tensor:
        """"""
        if y.dtype == torch.long:
            y = F.one_hot(y.view(-1)).to(torch.float)
        out = y
        if mask is not None:
            out = torch.zeros_like(y)
            out[mask] = y[mask]
        if isinstance(edge_index, SparseTensor) and not edge_index.has_value():
            edge_index = gcn_norm(edge_index, add_self_loops=False)
        elif isinstance(edge_index, Tensor) and edge_weight is None:
            edge_index, edge_weight = gcn_norm(edge_index, num_nodes=y.size(0),
                                               add_self_loops=False)
        res = (1 - self.alpha) * out
        for _ in range(self.num_layers):
            # propagate_type: (y: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight,
                                 size=None)
            out.mul_(self.alpha).add_(res)
            out = post_step(out)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(num_layers={}, alpha={})'.format(self.__class__.__name__,
                                                    self.num_layers,
                                                    self.alpha)

class LPStep(nn.Module):
    """two papers:
    http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf
    https://github.com/CUAI/CorrectAndSmooth
    """
    def __init__(self, args, data, split_masks):
        super().__init__()
        self.train_cnt = 0
        self.args = args
        self.train_idx = torch.where(split_masks['train']==True)[0].to(args.device)
        self.valid_idx = torch.where(split_masks['valid']==True)[0].to(args.device)
        self.test_idx = torch.where(split_masks['test']==True)[0].to(args.device)
        self.split_idx = {'train': self.train_idx, 'valid': self.valid_idx, 'test': self.test_idx}
        self.no_prep = args.lpStep.no_prep
        adj, D_isqrt = process_adj(data)
        DAD, DA, AD = gen_normalized_adjs(adj, D_isqrt)

        self.lp_dict = {
                'train_only': True,
                'alpha1': args.lpStep.alpha1, 
                'alpha2': args.lpStep.alpha2,
                'A1': eval(args.lpStep.A1),
                'A2': eval(args.lpStep.A2),
                'num_propagations1': args.lpStep.num_propagations1,
                'num_propagations2': args.lpStep.num_propagations2,
                'display': False,
                'device': args.device,

                # below: lp only
                'idxs': ['train'],
                'alpha': args.lpStep.alpha,
                'num_propagations': args.lpStep.num_propagations,
                'A': eval(args.lpStep.A),
            }
        self.fn = eval(self.args.lpStep.fn)
        return

    def forward(self, model_out, data):
        # need to pass 'data.y' through 'data'
        self.train_cnt += 1
        if self.args.lpStep.lp_force_on_cpu:
            self.split_idx, data, model_out = to_device([self.split_idx, data, model_out], 'cpu')
        else:
            self.split_idx, data, model_out = to_device([self.split_idx, data, model_out], self.args.device)

        if self.no_prep:
            out = label_propagation(data, self.split_idx, **self.lp_dict)
        else:
            _, out = self.fn(data, model_out, self.split_idx, **self.lp_dict)

        self.split_idx, data, model_out = to_device([self.split_idx, data, model_out], self.args.device)
        return out

class PreStep(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        return

    def forward(self, data):
        embs = []
        if 'diffusion' in self.args.preStep.pre_methods:
            embs.append(preprocess(data, 'diffusion', self.args.preStep.num_propagations, post_fix=self.args.dataset))
        if 'spectral' in self.args.preStep.pre_methods:
            embs.append(preprocess(data, 'spectral', self.args.preStep.num_propagations, post_fix=self.args.dataset))
        if 'community' in self.args.preStep.pre_methods:
            embs.append(preprocess(data, 'community', self.args.preStep.num_propagations, post_fix=self.args.dataset))

        embeddings = torch.cat(embs, dim=-1)
        return embeddings

class MidStep(nn.Module):
    def __init__(self, args, embs, data):
        super().__init__()
        self.args = args
        self.train_cnt = 0
        self.best_valid = 0.
        self.data = data
        if args.midStep.model == 'mlp':
            self.model = MLP(embs.size(-1)+args.num_feats,args.midStep.hidden_channels, args.num_classes, args.midStep.num_layers, 0.5, args.dataset == 'Products').to(args.device)
        elif args.midStep.model=='linear':
            self.model = MLPLinear(embs.size(-1)+args.num_feats, args.num_classes).to(args.device)
        elif args.midStep.model=='plain':
            self.model = MLPLinear(embs.size(-1)+args.num_feats, args.num_classes).to(args.device)
        return

    def forward(self, x):
        return self.model(x)

    def train_forward(self, embs, y, optimizer, loss_op, split_masks):
        self.train_cnt += 1
        x = torch.cat(to_device([self.data.x, embs], self.args.device), dim=-1)

        y = self.data.y.to(self.args.device)
        train_mask = split_masks['train']
        valid_mask = split_masks['valid']
        test_mask = split_masks['test']

        optimizer.zero_grad()
        out = self.model(x)
        if isinstance(loss_op, torch.nn.NLLLoss):
            out = F.log_softmax(out, dim=-1)

        loss = loss_op(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        valid_acc = cal_acc_logits(out[valid_mask], y[valid_mask])

        print('step2 test_acc = ',cal_acc_logits(out[test_mask], y[test_mask]))

        if valid_acc > self.best_valid:
            self.best_valid = valid_acc
            self.best_out = out.exp()
            print('!!! best val', self.train_cnt, f'={self.best_valid*100:.2}')
        loss = float(loss.item())
        return self.best_out, loss

def cal_acc_logits(output, labels):
    # work with model-output-logits, not model-output-indices
    assert len(output.shape)==2 and output.shape[1]>1
    labels = labels.reshape(-1).to('cpu')
    indices = torch.max(output, dim=1)[1].to('cpu')
    correct = float(torch.sum(indices == labels)/len(labels))
    return correct

def cal_acc_indices(output, labels):
    assert (len(output.shape)==2 and output.shape[1]==1) or len(output.shape)==1
    labels = labels.reshape(-1).to('cpu')
    output = output.reshape(-1).to('cpu')
    correct = float(torch.sum(output == labels)/len(labels))
    return correct

def to_device(list1d, device):
    newl = []
    for x in list1d:
        if type(x) is dict:
            for k,v in x.items():
                x[k] = v.to(device)
        else:
            x = x.to(device)
        newl.append(x)
    return newl
