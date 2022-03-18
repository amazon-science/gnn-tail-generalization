from utils import *

class StudentBaseMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if args.StudentBaseMLP.dim_model==-1:
            dim_model = None
        else:
            dim_model = args.StudentBaseMLP.dim_model
        self.model = BlockResMLP(dims_in_out = args.StudentBaseMLP.dims_in_out, dim_model=dim_model, skip_conn_period=args.StudentBaseMLP.skip_conn_period, num_blocks=args.StudentBaseMLP.num_blocks)

    def forward(self, x, edge_index=None, mask=None):
        if mask is not None:
            x = x[mask]
        return self.model(x)
    def get_emb4linkp(self, x, edge_index, mask=None):
        # return ALL nodes
        return self.model(x)

class BlockResMLP(nn.Module):
    def __init__(self, dims_in_out, num_blocks, skip_conn_period=2, dim_hidden=None, dim_model=None, activation=nn.GELU, bias=True, dropout=0.1):
        # dims_in_out: it is a 2-element list
        super().__init__()
        
        self.dims_in_out = dims_in_out
        self.dim_model = dim_model or min(max(dims_in_out), 256)
        self.dim_hidden = dim_hidden or int(self.dim_model*1.5)+2
        

        self.num_blocks = num_blocks

        self.in_proj = nn.Identity() if self.dim_model==dims_in_out[0] else nn.Linear(dims_in_out[0], self.dim_model)
        self.out_proj = nn.Identity() if self.dim_model==dims_in_out[1] else nn.Linear(self.dim_model, dims_in_out[1])

        neurons = [self.dim_model] + [self.dim_hidden]*(skip_conn_period-1) + [self.dim_model]
        self.blocks = nn.ModuleList([getMLP(neurons, activation=activation, bias=bias, dropout=dropout, last_dropout=True) for _ in range(self.num_blocks-1)])
        self.blocks.append(getMLP(neurons, activation=activation, bias=bias, dropout=dropout, last_dropout=False))
        return

    def forward(self, x):
        x = self.in_proj(x)
        for block in self.blocks:
            h = x
            x = block(x)
            x = h + x
        x = self.out_proj(x)
        return x

class SEMLP(nn.Module):
    # The implementation of Cold Brew's MLP.
    def __init__(self, args, data, teacherGNN):
        super().__init__()
        self.hidden_dim = 256
        self.args = args
        self.train_mask = data.train_mask
        self.test_mask = data.test_mask
        self.train_idx = data.train_idx # torch.where(self.train_mask==True)[0]
        self.test_idx = data.test_idx   # torch.where(self.test_mask==True)[0]
        if self.args.batch_size>len(self.train_idx):
            print(f'\n\n    Batch size too large...\n Changing batch_size from {self.args.batch_size} to {len(self.train_idx)}!\n\n')
            self.args.batch_size = len(self.train_idx)
        self.has_NCloss = False
        self.adj_pow = None
        self.topK_2_replace = args.SEMLP_topK_2_replace

        self.teacherGNN = teacherGNN
        self.part1 = None
        self.part2 = None
        self.alphas = nn.Parameter(torch.tensor([0.0001,0.0001]), requires_grad=True)
        if self.has_NCloss:
            self.out_proj = nn.Linear(self.hidden_dim, args.num_classes_bkup)
        return

    def forward_part1(self, x, edge_index=None, batch_idx=None):        
        if self.has_NCloss and self.adj_pow is None:
            adj = graphUtils.normalize_adj(edge_index)
            self.adj_pow = graphUtils.sparse_power(adj, self.args.graphMLP_r)

        if self.part1 is None:
            if self.args.StudentMLP__dim_model==-1:
                dim_model = None
            else:
                dim_model = self.args.StudentBaseMLP.dim_model
            # ----- build part1 MLP module -----
            neurons_io = [self.args.num_feats, self.teacherGNN.model.model.get_se_dim(x, edge_index)]
            if self.args.SEMLP_part1_arch == 'residual': # options: 'residual', '2layer', '3layer', '4layer'
                self.part1 = BlockResMLP(dims_in_out = neurons_io, dim_model=dim_model, skip_conn_period=self.args.StudentBaseMLP.skip_conn_period, num_blocks=self.args.StudentBaseMLP.num_blocks).to(self.args.device)
            else:
                nlayer = int(self.args.SEMLP_part1_arch[0])
                neurons = [neurons_io[0]] + [256]*(nlayer-1) + [neurons_io[1]]
                self.part1 = getMLP(neurons, dropout=self.args.dropout_MLP).to(self.args.device)
            self.opt = self.optfun(self.parameters(),lr=self.args.lr, weight_decay=self.args.weight_decay)

        if batch_idx is not None:
            x = x[batch_idx]
        le_guess = self.part1(x)
        return le_guess

    def forward_part2(self, x, batch_idx=None, edge_index=None, ):
        if batch_idx is not None:
            x = x[batch_idx]
        if self.args.SEMLP__downgrade_to_MLP:
            part2_in = x
        else:
            part1_out = self.forward_part1(x, batch_idx).detach()*self.alphas[0]
            replaced = self.replacement(part1_out)*self.alphas[1]

            if self.args.SEMLP__include_part1out:
                part2_in = torch.cat([x, replaced, part1_out], dim=-1)
            else:
                part2_in = torch.cat([x[batch_idx], replaced], dim=-1)

        if self.part2 is None:
            if self.args.StudentMLP__dim_model==-1:
                dim_model = None
            else:
                dim_model = self.args.StudentBaseMLP.dim_model
            neurons_io = [part2_in.shape[-1], self.args.num_classes_bkup]
            
            if self.args.train_which=='GraphMLP':
                self.part2 = GraphMLP(self.args, self.train_mask).to(self.args.device)
            elif self.args.train_which=='StudentBaseMLP':
                self.part2 = BlockResMLP(dims_in_out = [self.args.num_feats, self.args.num_classes_bkup], dim_model=dim_model, skip_conn_period=self.args.StudentBaseMLP.skip_conn_period, num_blocks=self.args.StudentBaseMLP.num_blocks).to(self.args.device)
            else:
                neurons = [part2_in.shape[1], 256, self.args.num_classes_bkup]
                self.part2 = getMLP(neurons, dropout=self.args.dropout_MLP).to(self.args.device)

            self.opt = self.optfun(self.parameters(),lr=self.args.lr, weight_decay=self.args.weight_decay)
       
        if self.args.train_which=='GraphMLP':
            res = self.part2(part2_in, edge_index=edge_index, batch_idx=batch_idx)
            y = res.emb
            self.loss_NContrastive = res.loss_NContrastive
        else:
            y = self.part2(part2_in)
        return y

    def forward(self, x, edge_index=None):
        return 

    def replacement(self, le_guess, node_idx=None):
        le_guess = le_guess.detach()
        res_N_feat = []
        teacherSE_T = self.teacherSE.transpose(0,1)
        if node_idx is None:
            node_idx = np.arange(len(le_guess))
        for idx in node_idx:
            attn_1N = torch.matmul(le_guess[[idx]], teacherSE_T)
            sortidx = attn_1N.argsort()[0]
            select = sortidx[-self.topK_2_replace:]
            attn_1N = F.softmax(attn_1N[:,select], dim=1)
            z_1_feat = torch.matmul(attn_1N, self.teacherSE[select])
            res_N_feat.append(z_1_feat)
        return torch.cat(res_N_feat, dim=0).detach()

class GraphMLP(nn.Module):
    # pytorch re-implementation of GRAPHMLP: https://arxiv.org/pdf/2106.04051.pdf
    def __init__(self, args, train_mask):
        super().__init__()
        self.dropout = 0.6          # reported in the paper
        self.hidden_dim = 256       # reported in the paper
        self.args = args
        neurons = [args.num_feats, self.hidden_dim, self.hidden_dim]
        self.model = getMLP(neurons, dropout=self.dropout).to(args.device)
        self.out_proj = nn.Linear(self.hidden_dim, args.num_classes_bkup)
        self.train_mask = train_mask
        self.train_idx = torch.where(self.train_mask==True)[0]
        if self.args.batch_size>len(self.train_idx):
            print(f'\n\n    Batch size too large...\n Changing batch_size from {self.args.batch_size} to {len(self.train_idx)}!\n\n')
            self.args.batch_size = len(self.train_idx)
        self.adj_pow = None

    def forward(self, x, edge_index=None, batch_idx=None):
        if self.adj_pow is None:
            adj = graphUtils.normalize_adj(edge_index)
            self.adj_pow = graphUtils.sparse_power(adj, self.args.graphMLP_r)
        z = self.model(x)
        info = D()
        info.loss_NContrastive = get_neighbor_contrastive_loss(z, self.adj_pow, batch_idx, self.args.graphMLP_tau)
        info.emb = self.out_proj(z)
        return info

    def get_emb4linkp(self, x, edge_index, mask=None):
        # return ALL nodes
        raise NotImplementedError
        return self.model(x)

def get_neighbor_contrastive_loss(z, adj_pow, batch_idx, tau):
    mask = torch.eye(len(z)).to(z.device)
    simz = (1 - mask) * torch.exp(cosine_sim(z)/tau)    # shape: [B, B]
    adj_pow = graphUtils.crop_adj_to_subgraph(adj_pow, batch_idx).to_dense() # shape: [B, B]
    numerator = (adj_pow*simz).sum(dim=1, keepdim=False) # 1D tensor
    denominator = simz.sum(dim=1, keepdim=False) # 1D tensor
    nonzero = torch.where(numerator!=0)[0]
    loss_NContrastive = - torch.mean(torch.log(numerator[nonzero]/denominator[nonzero]))
    return loss_NContrastive

def cosine_sim(x):
    # This function returns the pair-wise cosine semilarity.
    # x.shape = [N_nodes, 256]
    # returned shape: [N_nodes, N_nodes]
    x_dis = x @ x.T
    x_sum = torch.norm(x, p=2, dim=1, keepdim=True)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    return x_dis
