from torch.utils.data import DataLoader
from .layer import *
from .loss import *
from .utils import *
from .edge_LP import run_embLP, run_xmcLP, run_logitLP

class BaseModel(object):
    """
        Parameters
        ----------
        lr : double
            Learning rate
        dropout : double
            dropout probability for gnn and mlp layers
        gnn_num_layers : int
            number of gnn layers
        mlp_num_layers : int
            number of gnn layers
        *_hidden_channels : int
            dimension of hidden
        num_nodes : int
            number of graph nodes
        num_node_feats : int
            dimension of raw node features
        gnn_encoder_name : str
            gnn encoder name
        predictor_name: str
            link predictor name
        loss_func: str
            loss function name
        optimizer_name: str
            optimization method name
        device: str
            device name: gpu or cpu
        use_node_feats: bool
            whether to use raw node features as input
        train_node_emb: bool
            whether to train node embeddings based on node id
        pretrain_emb: str
            whether to load pretrained node embeddings
    """

    def __init__(self, args, data, lr, dropout, grad_clip_norm, gnn_num_layers, mlp_num_layers, emb_hidden_channels,
                 gnn_hidden_channels, mlp_hidden_channels, num_nodes, num_node_feats, gnn_encoder_name,
                 predictor_name, loss_func, optimizer_name, device, use_node_feats, train_node_emb, pretrain_emb=None):
        # self.is_empty = args.encoder in ['EGI', 'CN', 'AA', 'PPR']
        self.encoder_name = gnn_encoder_name
        self.args = args

        self.loss_func_name = loss_func
        self.num_nodes = num_nodes
        self.num_node_feats = num_node_feats
        self.use_node_feats = use_node_feats
        self.train_node_emb = train_node_emb
        self.clip_norm = grad_clip_norm
        self.device = device

        # Input Layer
        self.input_channels, self.emb = create_input_layer(num_nodes=num_nodes,
                                                           num_node_feats=num_node_feats,
                                                           hidden_channels=emb_hidden_channels,
                                                           use_node_feats=use_node_feats,
                                                           train_node_emb=train_node_emb,
                                                           pretrain_emb=pretrain_emb)
        if self.emb is not None:
            self.emb = self.emb.to(device)

        # GNN Layer
        self.encoder = create_gnn_layer(input_channels=self.input_channels,
                                        hidden_channels=gnn_hidden_channels,
                                        num_layers=gnn_num_layers,
                                        dropout=dropout,
                                        encoder_name=gnn_encoder_name, data=data).to(device)

        # Predict Layer
        self.predictor = create_predictor_layer(hidden_channels=mlp_hidden_channels,
                                                num_layers=mlp_num_layers,
                                                dropout=dropout,
                                                predictor_name=predictor_name).to(device)

        # Parameters and Optimizer
        self.para_list = list(self.encoder.parameters()) + list(self.predictor.parameters())
        if self.emb is not None:
            self.para_list += list(self.emb.parameters())
        if optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.para_list, lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.para_list, lr=lr)

    def param_init(self):
        self.encoder.reset_parameters()
        self.predictor.reset_parameters()
        if self.emb is not None:
            torch.nn.init.xavier_uniform_(self.emb.weight)

    def create_input_feat(self, data):
        if self.use_node_feats:
            if self.args.linkpred_baseline in ['EGI', 'DGI']:
                input_feat = self.embs.to(self.device)
            else:
                input_feat = data.x.to(self.device)
            if self.train_node_emb:
                input_feat = torch.cat([self.emb.weight, input_feat], dim=-1)
        else:
            input_feat = self.emb.weight
        return input_feat

    def calculate_loss(self, pos_out, neg_out, num_neg, margin=None):
        if self.loss_func_name == 'ce_loss':
            loss = ce_loss(pos_out, neg_out)
        elif self.loss_func_name == 'info_nce_loss':
            loss = info_nce_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == 'log_rank_loss':
            loss = log_rank_loss(pos_out, neg_out, num_neg)
        elif self.loss_func_name == 'adaptive_auc_loss' and margin is not None:
            loss = adaptive_auc_loss(pos_out, neg_out, num_neg, margin)
        else:
            loss = auc_loss(pos_out, neg_out, num_neg)
        return loss

    def train(self, data, split_edge, batch_size, neg_sampler_name, num_neg):
        if self.encoder_name in ['CN', 'AA', 'PPR']:
            return -1.
            return torch.tensor([0.], requires_grad=True, device=self.device)

        self.encoder.train()
        self.predictor.train()

        # pos_train_edge contains full positive train edge; neg_train_edge is controlled to be containing the same number of neg edges
        pos_train_edge, neg_train_edge = get_pos_neg_edges('train', split_edge,
                                                           edge_index=data.edge_index,
                                                           num_nodes=self.num_nodes,
                                                           neg_sampler_name=neg_sampler_name,
                                                           num_neg=num_neg)

        pos_train_edge, neg_train_edge = pos_train_edge.to(self.device), neg_train_edge.to(self.device)

        if 'weight' in split_edge['train']:
            edge_weight_margin = split_edge['train']['weight'].to(self.device)
        else:
            edge_weight_margin = None

        total_loss = total_examples = 0
        for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
            self.optimizer.zero_grad()

            input_feat = self.create_input_feat(data)
            h = self.encoder(input_feat, data.adj_t)  # input_feat & h both have shape: [N_nodes_full, dim=256]
            pos_edge = pos_train_edge[perm].t()
            neg_edge = torch.reshape(neg_train_edge[perm], (-1, 2)).t()

            pos_out = self.predictor(h[pos_edge[0]], h[pos_edge[1]])  # shape: [num_pos_edges]
            neg_out = self.predictor(h[neg_edge[0]], h[neg_edge[1]])

            weight_margin = edge_weight_margin[perm] if edge_weight_margin is not None else None

            loss = self.calculate_loss(pos_out, neg_out, num_neg, margin=weight_margin)
            loss.backward()

            if self.clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip_norm)
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.clip_norm)

            self.optimizer.step()
            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
        
        return total_loss / total_examples

    @torch.no_grad()
    def batch_predict(self, h, edges, batch_size): 
        # this function take the entire split of edges, use the model predictor head, and output the score for the entire split of edges. Used only in test, not during training. 
        # # h has shape: [N_total_nodes, dim] ; 
        # edges shape: [N_total_edge_pos_or_neg, 2], pred shape: [N_total_edge_pos_or_neg]
        if self.encoder_name in ['CN', 'AA', 'PPR']:
            pred = self.encoder.get_score(edges.T)
            pred = torch.tensor(pred, device=edges.device)
        else:
            preds = []
            for perm in DataLoader(range(edges.size(0)), batch_size):
                edge = edges[perm].t()
                preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
            pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test(self, data, split_edge, batch_size, evaluator, eval_metric):
        args = self.args
        self.encoder.eval()
        self.predictor.eval()
        input_feat = self.create_input_feat(data)
        h = self.encoder(input_feat, data.adj_t)

        pos_train_edge, neg_train_edge = get_pos_neg_edges('train', split_edge, # neg_train_edge shape = [N_edge, num_neg, 2]
                                                           edge_index=data.edge_index,
                                                           num_nodes=self.num_nodes,
                                                           neg_sampler_name='global',
                                                           num_neg=self.args.num_neg)
        neg_train_edge = neg_train_edge.reshape([-1,2])
        pos_valid_edge, neg_valid_edge = get_pos_neg_edges('valid', split_edge) # shape=[60084, 2]; [100000, 2]; contain all pn edges in split_edge.
        pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge)

        pos_train_edge, neg_train_edge = pos_train_edge.to(self.device), neg_train_edge.to(self.device)
        pos_valid_edge, neg_valid_edge = pos_valid_edge.to(self.device), neg_valid_edge.to(self.device)
        pos_test_edge, neg_test_edge = pos_test_edge.to(self.device), neg_test_edge.to(self.device)

        if args.edge_lp_mode=='emb':
            pos_train_pred, pos_valid_pred, pos_test_pred, neg_train_pred, neg_valid_pred, neg_test_pred = run_embLP(h, data.edge_index, args.LP_device, args.ELP_alpha, args.num_propagations,
                pos_train_edge, pos_valid_edge, pos_test_edge, neg_train_edge, neg_valid_edge, neg_test_edge)
        else:
            pos_train_pred = self.batch_predict(h, pos_train_edge, batch_size) # pos_train_edge/neg_valid_edge/... : torch.tensor on cuda, shape=[N_edge, 2] (transposed from edge_index)
            neg_train_pred = self.batch_predict(h, neg_train_edge, batch_size)

            pos_valid_pred = self.batch_predict(h, pos_valid_edge, batch_size) # [N_edge, dim] -> [N_edge, ]
            neg_valid_pred = self.batch_predict(h, neg_valid_edge, batch_size)

            pos_test_pred = self.batch_predict(h, pos_test_edge, batch_size)
            neg_test_pred = self.batch_predict(h, neg_test_edge, batch_size)


        if args.edge_lp_mode == 'logit':
                # edge LP requires:
                    # h: already learned node embeddings
                    # edge_logits: logits for all edges (to be computed again using 'h')
                    # A matrix (original)
                # edge LP algorithm inputs:
                    # A_edgeLP: computed by input A
                    # Y0: computed by 'h' or edge_logits
                    # G: computed by 'h' or edge_logits
                pos_train_pred, pos_valid_pred, pos_test_pred, neg_train_pred, neg_valid_pred, neg_test_pred = run_logitLP(
                    data.edge_index, args.LP_device, args.ELP_alpha, args.num_propagations,
                    pos_train_pred, pos_valid_pred, pos_test_pred, 
                    neg_train_pred, neg_valid_pred, neg_test_pred)
        if args.edge_lp_mode == 'xmc':
                pos_train_pred, pos_valid_pred, pos_test_pred, neg_train_pred, neg_valid_pred, neg_test_pred = run_xmcLP(
                    data.edge_index, data.num_nodes, args.LP_device, args.ELP_alpha, args.num_propagations,
                    pos_train_pred, pos_valid_pred, pos_test_pred, neg_train_pred, neg_valid_pred, neg_test_pred,
                    pos_train_edge, pos_valid_edge, pos_test_edge, neg_train_edge, neg_valid_edge, neg_test_edge)

        if eval_metric == 'hits':
            results = evaluate_hits(
                evaluator,
                pos_valid_pred,
                neg_valid_pred,
                pos_test_pred,
                neg_test_pred)
        
        elif eval_metric == 'mrr':
            results = evaluate_mrr(
                evaluator,
                pos_valid_pred,
                neg_valid_pred,
                pos_test_pred,
                neg_test_pred)

        elif 'recall_my' in eval_metric:
            results = evaluate_recall_my(
                pos_train_pred,
                neg_train_pred,
                pos_valid_pred,
                neg_valid_pred,
                pos_test_pred,
                neg_test_pred, topk=eval_metric.split('@')[1])

        return results

def create_input_layer(num_nodes, num_node_feats, hidden_channels, use_node_feats=True,
                       train_node_emb=False, pretrain_emb=None):
    emb = None
    if use_node_feats:
        input_dim = num_node_feats
        if train_node_emb:
            emb = torch.nn.Embedding(num_nodes, hidden_channels)
            input_dim += hidden_channels
        elif pretrain_emb is not None and pretrain_emb != '':
            weight = torch.load(pretrain_emb)
            emb = torch.nn.Embedding.from_pretrained(weight)
            input_dim += emb.weight.size(1)
    else:
        if pretrain_emb is not None and pretrain_emb != '':
            weight = torch.load(pretrain_emb)
            emb = torch.nn.Embedding.from_pretrained(weight)
            input_dim = emb.weight.size(1)
        else:
            emb = torch.nn.Embedding(num_nodes, hidden_channels)
            input_dim = hidden_channels
    return input_dim, emb

def create_gnn_layer(input_channels, hidden_channels, num_layers, dropout=0, encoder_name='SAGE', data=None):
    if encoder_name.upper() == 'GCN':
        return GCN(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'WSAGE':
        return WSAGE(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'TRANSFORMER':
        return Transformer(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'SAGE':
        return SAGE(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name.upper() == 'MLP':
        return MLP(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
    elif encoder_name in ['CN', 'AA', 'PPR']:
        return Heuristics(encoder_name, data)
    else:
        raise NotImplementedError(f'encoder_name {encoder_name} not implemented')

def create_predictor_layer(hidden_channels, num_layers, dropout=0, predictor_name='MLP'):
    predictor_name = predictor_name.upper()
    if predictor_name == 'DOT':
        return DotPredictor()
    elif predictor_name == 'BIL':
        return BilinearPredictor(hidden_channels)
    elif predictor_name == 'MLP':
        return MLPPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout)
    elif predictor_name == 'MLPDOT':
        return MLPDotPredictor(hidden_channels, 1, num_layers, dropout)
    elif predictor_name == 'MLPBIL':
        return MLPBilPredictor(hidden_channels, 1, num_layers, dropout)
    elif predictor_name == 'MLPCAT':
        return MLPCatPredictor(hidden_channels, hidden_channels, 1, num_layers, dropout)
