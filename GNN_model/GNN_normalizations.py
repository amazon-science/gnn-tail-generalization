# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from torch import nn

from .norm_tricks import *
from .GCN import TricksComb
from func_libs import *






class TeacherGNN(nn.Module):
    # This class is the teacher GCN model (with structural embedding) for cold brew
    def __init__(self, args, proj2class=None):
        super().__init__()
        proj2class = proj2class or nn.Identity()
        args.num_classes_bkup = args.num_classes
        args.num_classes = args.dim_commonEmb
        self.args = args

        if self.args.dim_learnable_input>0:
            embs = torch.randn(args.N_nodes, args.dim_learnable_input)*0.001
            self.embs = nn.Parameter(embs, requires_grad=True)
            self.args.num_feats_bkup = self.args.num_feats
            self.args.num_feats = self.args.dim_learnable_input

        from GNN_model.GNN_normalizations import GNN_norm as GNN_trickComb
        self.model = GNN_trickComb(args)


        self.proj2linkp = getMLP(args.TeacherGNN.neurons_proj2linkp).to(args.device)
        self.proj2class = proj2class
        self.dglgraph = None

    def forward(self, x, edge_index):        
        if self.args.TeacherGNN.change_to_featureless:
            x = x*0

        if self.args.dim_learnable_input>0:
            x = self.embs
        
        commonEmb, self.se_reg_all = self.model(x, edge_index)
        self.out = commonEmb
        return commonEmb

    def get_3_embs(self, x, edge_index, mask=None, want_heads=True):
        commonEmb = self.forward(x, edge_index)
        emb4classi_full = self.proj2class(commonEmb)
        if want_heads:
            if mask is not None:
                emb4classi = emb4classi_full[mask]
            else:
                emb4classi = emb4classi_full

            emb4linkp = self.proj2linkp(commonEmb)
        else:
            emb4linkp = emb4classi = None
        res = D()
        res.commonEmb, res.emb4classi, res.emb4classi_full, res.emb4linkp = commonEmb, emb4classi, emb4classi_full, emb4linkp

        return res

    def get_emb4linkp(self, x, edge_index, mask=None):
        # return ALL nodes
        _, _, emb4linkp = self.get_3_embs(x, edge_index, want_heads=True)
        return emb4linkp

    def graph2commonEmb(self, x, edge_index, train_mask):
        commonEmb = self.forward(x, edge_index)
        commonEmb_train = commonEmb[train_mask]
        return commonEmb_train, commonEmb




class GNN_norm(nn.Module):
    def __init__(self, args):
        super(GNN_norm, self).__init__()

        self.model = TricksComb(args)
        
    def forward(self, x, edge_index):
        return self.model.forward(x, edge_index)

