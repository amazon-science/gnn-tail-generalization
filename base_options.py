import argparse
import numpy as np
import os
import torch

from secrets import choice

class BaseOptions():
    def get_arguments(self):
        _exp_mode = ['coldbrew', 'I2_GTL'][1]
        if _exp_mode=='coldbrew':
            _lr = 0.005
        elif _exp_mode=='I2_GTL':
            _lr = 0.001

        # ------- build up common parameters for Cold Brew and I2-GTL -------
        parser = argparse.ArgumentParser(description='Tail and cold start generalization')
        parser.add_argument('--exp_mode', type=str, default=_exp_mode)
        parser.add_argument("--lr", type=float, default=_lr, help="learning rate")
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument('--batch_size', type=int, default=64 * 1024)
        parser.add_argument("--epochs", type=int, default=1500)

        # ------- build up the parameters for node classification (Cold Brew) -------
        parser.add_argument('--samp_size_p', type=int, default=200)
        parser.add_argument('--samp_size_n_train', type=int, default=200)
        parser.add_argument('--samp_size_n_test_times_p', type=int, default=20)
        parser.add_argument('--dim_learnable_input', type=int, default=0, help="This arguments controls the featureless mode. If set to 0, no change to original GNN; otherwise, discard input features, and use learnable embeddings of the specified dimension.")
        parser.add_argument('--unify_mlps', type=int, default=0, help="auxiliary function for batch training: if set to True, reset the MLP type methods' coefficient.")
        parser.add_argument('--force_set_to_best_config', type=int, default=1, help="set dataset dependent configs.")
        parser.add_argument('--want_headtail', type=int, default=1, help="wheter to add head and tail evaluation results as output.")
        parser.add_argument('--num_layers', type=int, default=2, help="used for TeacherGNN")
        parser.add_argument('--studentMLP__skip_conn_T_and_res_blks', type=str, default='', help="architecture options for arch search of studentMLP. Use default is not doing architecture search.")
        parser.add_argument('--StudentMLP__dim_model', type=int, default=-1)
        parser.add_argument('--studentMLP__opt_lr', type=str, default='', help="optimization configuration for the studentMLP, better use default.")
        parser.add_argument('--LP__which_corr_and_DAD', type=str, default='', help="two hyperpatameters in label propagation; better use default.")
        parser.add_argument('--LP__num_propagations', type=int, default=-1, help="number of propagations in label propagation.")
        parser.add_argument('--LP__alpha', type=float, default=-1, help="the alpha coefficient for label propagation")
        parser.add_argument("--SEMLP_topK_2_replace", type=int, default=2, help="the hyper parameter used to replace with the top K best neighbors")
        parser.add_argument("--SEMLP__include_part1out", type=int, default=1, help="whether the part1 of cold brew's MLP is concatenated during part 2 training.")
        parser.add_argument("--dropout_MLP", type=float, default=0.2, help="dropout rate for Cold Brew MLP (both part1 and part2) and StudentBaseMLP modules.")
        parser.add_argument("--SEMLP_part1_arch", type=str, default='2layer', choices=['residual', '2layer', '3layer', '4layer'], help="architecture of part1 of Cold Brew MLP")
        parser.add_argument('--has_proj2class', type=int, default=0, help="whether cold brew's TeacherGNN has additional projection head")
        parser.add_argument("--whetherHasSE", type=str, default='000', choices=['100', '001', '111', '000'], help="whether cold brew's TeacherGNN has structural embedding.")
        parser.add_argument("--se_reg", type=float, default=10, help="regularization coefficient for cold brew's structural embedding")
        parser.add_argument("--graphMLP_reg", type=float, default=0., choices=[0., 1., 10., 100.], help="regularization coefficient in GraphMLP")
        parser.add_argument("--graphMLP_tau", type=float, default=2.0, choices=[0.5, 1.0, 2.0], help="the coefficient tau in GraphMLP")
        parser.add_argument("--graphMLP_r", type=int, default=3, choices=[2, 3, 4], help="the coefficient r (number of r-hop neighbors to consider) in GraphMLP")
        parser.add_argument("--change_to_featureless", type=int, default=0, help="whether switch to featureless graph.")
        parser.add_argument("--do_deg_analyze", type=int, default=1, help="if True, analyze graph data. has to be true, otherwise 'large_deg_mask' is not assigned and will bug.")
        parser.add_argument("--train_which", type=str, default='TeacherGNN', choices=['TeacherGNN', 'SEMLP', 'LP','StudentBaseMLP','GraphMLP','proj2class'])
        parser.add_argument("--task", type=str, default='nodeC')
        parser.add_argument("--dataset", type=str, default='', choices=['Cora','Citeseer','Pubmed','ogbn-arxiv','chameleon','ACTOR','squirrel','WISCONSIN','CORNELL','TEXAS'])
        parser.add_argument("--use_special_split", type=int, default=1)
        parser.add_argument('--optfun', type=str, default='torch.optim.Adam', choices=['torch.optim.Adam', 'torch.optim.SGD'])
        parser.add_argument('--manual_assign_GPU', type=int, default=-9999, help="default=-9999, means to choose bestGPU")
        parser.add_argument('--random_seed', type=int, default=100)
        parser.add_argument('--N_exp', type=int, default=1)
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument("--cuda", type=bool, default=True, required=False,
                            help="whether run on cuda or not (bool value)")
        parser.add_argument('--cuda_num', type=int, default=0, help="GPU number")
        parser.add_argument('--records_desc', type=str, default='res_connection',
                            help="file name of training records, try to make your exp settings directly readable from this name, e.g., gcn_PairNorm")
        parser.add_argument('--records_path', type=str, default='.', help="saving location of training records")
        parser.add_argument('--compare_model', type=int, default=0,
                            help="0 means compare single trick, 1 means compare model, 2 means compare trick combinations")
        parser.add_argument('--type_model', type=str, default="GCN",
                            choices=['GCN', 'GAT', 'SGC', 'GCNII', 'DAGNN', 'GPRGNN', 'APPNP', 'JKNet', 'DeeperGCN'])
        parser.add_argument('--type_trick', type=str, default='Initial+BatchNorm', help="type of residual/dropout/normalization trics used in the TeacherGNN of Cold Brew. Can be one of ['Initial', 'Jumping', 'Residual', ''] + one of ['GroupNorm', 'BatchNorm', 'NoNorm']")
        parser.add_argument('--layer_agg', type=str, default='concat',
                            choices=['concat', 'maxpool', 'attention', 'mean'],
                            help='aggregation function for skip connections')
        parser.add_argument('--res_alpha', type=float, default=0.1,
                            help='the trade off parameter of some res connections')
        parser.add_argument('--patience', type=int, default=100,
                            help="patience step for early stopping")  # 5e-4
        parser.add_argument("--multi_label", type=bool, default=False,
                            help="multi_label or single_label task")
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help="weight decay")  # 5e-4
        parser.add_argument('--dim_hidden', type=int, default=64)
        parser.add_argument('--transductive', type=bool, default=True,
                            help='transductive or inductive setting')
        parser.add_argument('--float_or_double', type=str, default="float", required=False,
                            help='do you want to train your model with float or double precision')
        parser.add_argument('--type_norm', type=str, default="None")
        parser.add_argument('--adj_dropout', type=float, default=0.5,
                            help="dropout rate in APPNP")  # 5e-4
        parser.add_argument('--edge_dropout', type=float, default=0.2,
                            help="dropout rate in EdgeDrop")  # 5e-4
        parser.add_argument('--node_norm_type', type=str, default="n", choices=['n', 'v', 'm', 'srv', 'pr'])
        parser.add_argument('--skip_weight', type=float, default=None)
        parser.add_argument('--num_groups', type=int, default=None)
        parser.add_argument('--prog', type=str, default='', help="support for batch running mode: progress")
        parser.add_argument('--rexName', type=str, default="res.npy",
                            help="support for batch running mode: record's name")
        parser.add_argument('--graph_dropout', type=float, default=0.2,
                            help="graph dropout rate (for dropout tricks)")  # 5e-4
        parser.add_argument('--layerwise_dropout', action='store_true', default=False)


        # ------- build up the parameters for link prediction transfer learning (I2-GTL) -------
        parser.add_argument('--public_data_convert_overlapped_subgraph', type=bool, default=True)
        parser.add_argument('--transfer_setting', type=str, default='i2t', choices=['t2t', 'u2t', 'i2t', 'u', 'i', ''])
        parser.add_argument('--linkpred_baseline', type=str, default='', choices=['', 'EGI', 'DGI'])
        parser.add_argument('--edge_lp_mode', type=str, default='logit', choices=['emb', 'logit', 'xmc', '' ])
        parser.add_argument('--ELP_alpha', type=str, default=0.995)
        parser.add_argument('--num_propagations', type=str, default=5)
        parser.add_argument('--LP_device', type=str, default='cuda:4', choices=['cpu', 'cuda:0', 'cuda:4'])
        parser.add_argument('--exp_on_cold_edge', type=bool, default=False)
        parser.add_argument('--encoder', type=str, default='SAGE', choices=['SAGE','MLP','CN','AA','PPR'])
        parser.add_argument('--predictor', type=str, default='DOT', choices=['MLP', 'DOT'])
        parser.add_argument('--optimizer', type=str, default='Adam')
        parser.add_argument('--loss_func', type=str, default='ce_loss', choices=['AUC', 'ce_loss', 'log_rank_loss', 'info_nce_loss'])
        parser.add_argument('--neg_sampler', type=str, default='global')
        parser.add_argument('--data_name', type=str, default='ogbl-citation2', choices=['ogbl-citation2','ogbl-collab'])
        parser.add_argument('--data_path', type=str, default='dataset')
        parser.add_argument('--eval_metric', type=str, default='recall_my@1.25', choices=['hits', 'mrr','recall_my@0.8', 'recall_my@1', 'recall_my@1.25', 'recall_my@0'])
        parser.add_argument('--res_dir', type=str, default='')
        parser.add_argument('--pretrain_emb', type=str, default='')
        parser.add_argument('--gnn_num_layers', type=int, default=2)
        parser.add_argument('--mlp_num_layers', type=int, default=2)
        parser.add_argument('--emb_hidden_channels', type=int, default=256)
        parser.add_argument('--gnn_hidden_channels', type=int, default=256)
        parser.add_argument('--mlp_hidden_channels', type=int, default=256)
        parser.add_argument('--grad_clip_norm', type=float, default=2.0)
        parser.add_argument('--num_neg', type=int, default=3)
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

        if args.unify_mlps:
            unify_mlps(args)
        args = self.reset_dataset_dependent_parameters(args)
        args = self.ini_records_saver(args)

        if args.manual_assign_GPU != -9999:
            args.cuda_num = args.manual_assign_GPU
        elif torch.cuda.is_available():
            args.cuda_num = bestGPU(True)

        set_labprop_configs(args)
        if args.force_set_to_best_config:
            force_set_to_best_config(args)

        print(
            f'\nConfigs: \n\tdataset = < {args.dataset} >\n\ttrain_which = < {args.train_which} >\n\ttype_trick = < {args.type_trick} >\n\tnum_layers = < {args.num_layers} >\n\tdim_hidden = < {args.dim_hidden} >\n\tGPU actually use = < {args.cuda_num} >\n\n')

        # ---- setup some manual hyperparameters ----
        if args.exp_mode=='coldbrew':
            args.has_loss_component_nodewise = True
            args.has_loss_component_edgewise = False
        elif args.exp_mode=='I2_GTL':
            args.has_loss_component_nodewise = False
            args.has_loss_component_edgewise = True
            if args.linkpred_baseline in ['EGI', 'DGI']:
                args.encoder='MLP'
                args.use_node_feats = True

        return args

    def ini_records_saver(self, args):
        records_file = os.path.join(args.records_path, args.records_desc)
        if os.path.exists(records_file):
            records_file_backup = os.path.join(args.records_path, args.records_desc + ' - backup')
            print(
                f'\n\n !!! Warning !!! assigned records_file < {records_file} > already exists, now re-name the previous one to < {records_file_backup} >\n\n')
            os.rename(records_file, records_file_backup)

        args.records_file = records_file
        assert not os.path.exists(records_file)
        return args

    ## setting the common hyperparameters used for comparing different methods of a trick
    def reset_dataset_dependent_parameters(self, args):
        if 'betr' in args.dataset:
            args.num_classes = self.betr_gcn_dout
            args.num_feats = self.betr_num_feats
            args.dropout = 0.4
            args.weight_decay = 5e-4
            args.dim_hidden = 128
            args.activation = 'relu'

        elif args.dataset == 'Cora':
            args.num_feats = 1433
            args.num_classes = 7
            args.N_nodes = 2708
            args.dropout = 0.6
            args.weight_decay = 5e-4
            args.patience = 100
            args.dim_hidden = 64
            args.activation = 'relu'

        elif args.dataset == 'Pubmed':
            args.num_feats = 500
            args.num_classes = 3
            args.N_nodes = 19717
            args.dropout = 0.5
            args.weight_decay = 5e-4
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

        elif args.dataset == 'Citeseer':
            args.num_feats = 3703
            args.num_classes = 6
            args.N_nodes = 3327

            args.dropout = 0.6
            args.weight_decay = 5e-4
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'

            args.res_alpha = 0.2

        elif args.dataset == 'ogbn-arxiv':
            args.num_feats = 128
            args.num_classes = 40
            args.N_nodes = 169343

            args.dropout = 0.1
            args.weight_decay = 0.
            args.patience = 200
            args.dim_hidden = 256

        # ==============================================
        # ========== below are other datasets ==========
        elif args.dataset == 'chameleon':
            args.num_feats = 128
            args.num_classes = 6
            args.N_nodes = 2277

            args.dropout = 0.5
            args.weight_decay = 5e-4
            args.dim_hidden = 256
            args.activation = 'relu'

        elif args.dataset == 'squirrel':
            args.num_feats = 128
            args.num_classes = 5
            args.N_nodes = 5201

            args.dropout = 0.5
            args.weight_decay = 5e-4
            args.dim_hidden = 256
            args.activation = 'relu'

        elif args.dataset == 'TEXAS':
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'
            args.num_feats = 1703
            args.num_classes = 5
            args.dropout = 0.6
            args.weight_decay = 5e-4
            args.res_alpha = 0.9
            args.N_nodes = 183

        elif args.dataset == 'WISCONSIN':
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'
            args.num_feats = 1703
            args.num_classes = 5
            args.dropout = 0.6
            args.weight_decay = 5e-4
            args.res_alpha = 0.9
            args.N_nodes = 251

        elif args.dataset == 'CORNELL':
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'
            args.num_feats = 1703
            args.num_classes = 5
            args.dropout = 0.
            args.weight_decay = 5e-4
            args.res_alpha = 0.9
            args.N_nodes = 183

        elif args.dataset == 'ACTOR':
            args.patience = 100
            args.dim_hidden = 256
            args.activation = 'relu'
            args.N_nodes = 7600
            args.num_feats = 932
            args.num_classes = 5
            args.dropout = 0.
            args.weight_decay = 5e-4
            args.res_alpha = 0.9

        return args

def best_alpha_or_agg(args):
    idx_1 = {'Citeseer': 0, 'Pubmed': 1, 'ogbn-arxiv': 2}
    idx_2 = {'GCN': 0, 'SGC': 1}
    idx_3 = {2: 0, 16: 1, 32: 2}
    idx_4 = {'Residual': 0, 'Initial': 1, 'Dense': 0, 'Jumping': 1}

    alpha_dict = {0: 0.1, 1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8}
    agg_dict = {0: 'concat', 1: 'maxpool', 2: 'attention'}

    if args.type_trick in ['Residual', 'Initial']:
        arr = np.load('res_init.npy')
        dltn = arr[..., 1] * 100
        tdln = dltn.transpose(1, 0, 2, 3, 4)
        alpha_matrix = np.argmax(tdln, axis=-1)
        idx = alpha_matrix[idx_1[args.dataset], idx_2[args.type_model], idx_3[args.num_layers], idx_4[args.type_trick]]
        return alpha_dict[idx], args.layer_agg
    elif args.type_trick in ['Jumping', 'Dense']:
        arr = np.load('dense_jumping.npy')
        dltn = arr[..., 1] * 100
        tdln = dltn.transpose(1, 0, 2, 3, 4)
        alpha_matrix = np.argmax(tdln, axis=-1)
        idx = alpha_matrix[idx_1[args.dataset], idx_2[args.type_model], idx_3[args.num_layers], idx_4[args.type_trick]]
        return args.res_alpha, agg_dict[idx]
    else:
        return args.res_alpha, args.layer_agg

def bestGPU(gpu_verbose=False, **w):
    import GPUtil
    import numpy as np

    Gpus = GPUtil.getGPUs()
    Ngpu = 4
    mems, loads = [], []
    for ig, gpu in enumerate(Gpus):
        memUtil = gpu.memoryUtil * 100
        load = gpu.load * 100
        mems.append(memUtil)
        loads.append(load)
        if gpu_verbose: print(f'gpu-{ig}:   Memory: {memUtil:.2f}%   |   load: {load:.2f}% ')
    bestMem = np.argmin(mems)
    bestLoad = np.argmin(loads)
    best = bestMem
    if gpu_verbose: print(f'//////   Will Use GPU - {best}  //////')

    return int(best)

def set_labprop_configs(args):
    class C: pass
    args.preStep = C()
    args.lpStep = C()
    args.midStep = C()

    args.lp_has_prep = 1

    args.preStep.num_propagations = 10
    args.preStep.p = 1
    args.preStep.alpha = 0.5
    args.preStep.pre_methods = 'diffusion+spectral' # options: sgc , diffusion , spectral , community

    args.midStep.model = ['mlp', 'linear', 'plain', 'gat'][0]
    args.midStep.hidden_channels = 256
    args.midStep.num_layers = 3

    if args.LP__which_corr_and_DAD == '':
        args.lpStep.A = 'DAD'
    else:
        args.lpStep.A = args.LP__which_corr_and_DAD

    if args.LP__num_propagations == -1:
        args.lpStep.num_propagations = 50
    else:
        args.lpStep.num_propagations = args.LP__num_propagations

    if args.LP__alpha == -1.:
        args.lpStep.alpha = 0.5
    else:
        args.lpStep.alpha = args.LP__alpha

    args.lpStep.fn = [  'double_correlation_fixed',  # 'lpStep.fn' ONLY apply to 'with MLP' case; not applicable to LP-only case.
                        'double_correlation_autoscale',
                        'only_outcome_correlation',
                        ][1]

    args.lpStep.A1 = 'DA'
    args.lpStep.A2 = 'AD'
    args.lpStep.alpha1 = 0.9791632871592579
    args.lpStep.alpha2 = 0.7564990804200602
    args.lpStep.num_propagations1 = 50
    args.lpStep.num_propagations2 = 50
    args.lpStep.lp_force_on_cpu = True  # fixed due to hard coding in C&S. please never change this.

    args.lpStep.no_prep = 1
    # if the above 'lpStep.no_prep' is set to 1, it means the 'LP-only' case. what will happen:
        # there will be no preprocessing (self.preStep);
        # no MLP (self.midStep);
        # the node features are never considered;
        # it will only take the label-propagation, with initialization of zero vectors at test nodes, and true labels at train nodes.

def force_set_to_best_config(args):
    print('-'*30,'\n\n\n   Now reseting configs !!! \n\n\n','-'*30)

    d2i = {"Cora":0, "Citeseer":1, "Pubmed":2, "ogbn-arxiv":3, "chameleon":4, "ACTOR":5, "squirrel":6, "WISCONSIN":7, "CORNELL":8, "TEXAS":9,}
    if args.dataset not in d2i.keys():
        return

    if args.train_which in ['SEMLP', 'StudentBaseMLP', 'TeacherGNN']:
        args.best_config_performance = [86.9639468690702, 72.44, 75.96000000000001, 71.5367364154476, 68.50877192982458, 31.947368421052637, 59.78866474543709, 65.09803921568627, 61.08108108108108, 81.62162162162163]
        TeacherGNN_arr1=(2, 4, 8, 16, 64)
        TeacherGNN_arr2=('NoRes', 'Initial', 'Dense', 'Residual')
        TeacherGNN_arr3=('NoNorm', 'GroupNorm', 'BatchNorm', 'PairNorm', 'NodeNorm')
        best_config_TeacherGNN = np.array([[0, 0, 4], [0, 0, 1], [4, 1, 2], [2, 1, 2], [1, 1, 3], [0, 0, 2], [0, 1, 4], [1, 3, 0], [2, 3, 3], [2, 3, 1]])
        whichcf = best_config_TeacherGNN[d2i[args.dataset]]
        x1 = TeacherGNN_arr1[whichcf[0]]
        x2 = TeacherGNN_arr2[whichcf[1]]
        x3 = TeacherGNN_arr3[whichcf[2]]
        args.type_trick = x2+x3

    if args.train_which in ['SEMLP', 'StudentBaseMLP']:

        arr1=("2&1", "2&4", "2&16", "2&32", "4&2", "4&8")
        arr2=(128, 256)
        arr3=('torch.optim.Adam&0.001', 'torch.optim.Adam&0.005', 'torch.optim.Adam&0.02', 'torch.optim.SGD&0.005')
        best_config = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 3], [1, 1, 0], [2, 0, 0], [0, 1, 2], [2, 1, 2], [0, 1, 0], [0, 1, 3], [0, 0, 2]])
        whichcf = best_config[d2i[args.dataset]]
        x1 = arr1[whichcf[0]]
        x2 = arr2[whichcf[1]]
        x3 = arr3[whichcf[2]]

        args.studentMLP__skip_conn_T_and_res_blks=x1
        args.StudentMLP__dim_model=x2
        args.studentMLP__opt_lr='torch.optim.Adam&0.005'

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

def unify_mlps(args):
    # This function is an auxiliary function for batch training: it reset the MLP type methods' coefficient.
    args.studentMLP__skip_conn_T_and_res_blks = '2&2'
    args.StudentMLP__dim_model = 128
    args.studentMLP__opt_lr = 'torch.optim.Adam&0.005'
    args.SEMLP__include_part1out = 1

    if args.train_which == 'SEMLP':
        args.SEMLP_topK_2_replace = 3

    elif args.train_which == 'GraphMLP':
        args.graphMLP_reg = 10
        args.graphMLP_tau = 1
        args.graphMLP_r = 3

    elif args.train_which in 'SEMLP_MLP':
        args.SEMLP_topK_2_replace = -99
        args.train_which = 'SEMLP'

    elif args.train_which == 'GraphMLP_MLP':
        args.graphMLP_reg = 0
        args.train_which = 'GraphMLP'
