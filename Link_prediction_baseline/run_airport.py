import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from IPython import embed
import scipy.sparse as sp
from collections import defaultdict
from torch.autograd import Variable
from tqdm import tqdm
import pickle
from collections import defaultdict
from sklearn.manifold import SpectralEmbedding

try:
    from models.dgi import DGI, MultiClassifier
    from models.subgi import SubGI
except ModuleNotFoundError:
    from Link_prediction_baseline.models.dgi import DGI, MultiClassifier
    from Link_prediction_baseline.models.subgi import SubGI

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def spectral_feature(graph, args):
    A = np.zeros([graph.number_of_nodes(), graph.number_of_nodes()])
    a,b = graph.all_edges()
    
    for id_a, id_b in zip(a.numpy().tolist(), b.numpy().tolist()):
        A[id_a, id_b] = 1
    embedding = SpectralEmbedding(n_components=args.n_hidden)
    features = torch.FloatTensor(embedding.fit_transform(A))
    return features

def degree_bucketing(graph, args, degree_emb=None, max_degree = 10):
    max_degree = args.n_hidden
    features = torch.zeros([graph.number_of_nodes(), max_degree])
    for i in range(graph.number_of_nodes()):
        try:
            features[i][min(graph.in_degree(i), max_degree-1)] = 1
        except:
            features[i][0] = 1
    return features

def tonp(arr):
    if type(arr) is torch.Tensor:
        return arr.detach().cpu().data.numpy()
    else:
        return np.asarray(arr)

def createTraining(labels, valid_mask = None, train_ratio=0.8):
    train_mask = torch.zeros(labels.shape, dtype=torch.bool)
    test_mask = torch.ones(labels.shape, dtype=torch.bool)
    
    num_train = int(labels.shape[0] * train_ratio)
    all_node_index = list(range(labels.shape[0]))
    np.random.shuffle(all_node_index)
    #for i in range(len(idx) * train_ratio):
    # embed()
    train_mask[all_node_index[:num_train]] = 1
    test_mask[all_node_index[:num_train]] = 0
    if valid_mask is not None:
        train_mask *= valid_mask
        test_mask *= valid_mask
    return train_mask, test_mask

def read_struct_net(args):
    #g = DGLGraph()
    g = nx.Graph()
    #g.add_nodes(1000)
    with open(args.file_path) as IN:
        for line in IN:
            tmp = line.strip().split()
            # print(tmp[0], tmp[1])
            g.add_edge(int(tmp[0]), int(tmp[1]))
    labels = dict()
    with open(args.label_path) as IN:
        IN.readline()
        for line in IN:
            tmp = line.strip().split(' ')
            labels[int(tmp[0])] = int(tmp[1])
    return g, labels

def read_struct_net_v2(edge_index, num_nodes):
    # edge_index in np form
    edge_index = tonp(edge_index)
    g = nx.Graph()
    for i in range(edge_index.shape[1]):
        g.add_edge(int(edge_index[0,i]), int(edge_index[1,i]))
    labels = {i: 0 for i in range(num_nodes)}
    return g, labels 

def constructDGL(graph, labels):
    node_mapping = defaultdict(int)
    relabels = []
    for node in sorted(list(graph.nodes())):
        node_mapping[node] = len(node_mapping)
        relabels.append(labels[node])
    assert len(node_mapping) == len(labels)
    new_g = DGLGraph()
    new_g.add_nodes(len(node_mapping))
    for i in range(len(node_mapping)):
        new_g.add_edge(i, i)
    for edge in graph.edges():
        new_g.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])
        new_g.add_edge(node_mapping[edge[1]], node_mapping[edge[0]])
    
    return new_g, relabels

def output_adj(graph):
    A = np.zeros([graph.number_of_nodes(), graph.number_of_nodes()])
    a,b = graph.all_edges()
    for id_a, id_b in zip(a.numpy().tolist(), b.numpy().tolist()):
        A[id_a, id_b] = 1
    return A

# dump the best run
def main_bkup():

    _gpu=-1
    _label_path = 'data/labels-usa-airports.txt'
    _file_path = 'data/usa-airports.edgelist'

    _n_classifier_epochs = [100, 2][1]
    _n_dgi_epochs = [100, 2][1]

    parser = argparse.ArgumentParser(description='DGI')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=_gpu,
                        help="gpu")
    parser.add_argument("--dgi-lr", type=float, default=1e-2,
                        help="dgi learning rate")
    parser.add_argument("--classifier-lr", type=float, default=1e-2,
                        help="classifier learning rate")
    parser.add_argument("--n-dgi-epochs", type=int, default=_n_dgi_epochs,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=_n_classifier_epochs,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=32,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=20,
                        help="early stop patience condition")
    parser.add_argument("--model", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--self-loop", action='store_false',
                        help="graph self-loop (default=)")
    parser.add_argument("--model-type", type=int, default=2,
                    help="graph self-loop (default=False)")
    parser.add_argument("--graph-type", type=str, default="DD",
                    help="graph self-loop (default=False)")
    parser.add_argument("--data-id", type=str,default='',
                    help="[usa, europe, brazil]")
    parser.add_argument("--data-src", type=str, default='',
                    help="[usa, europe, brazil]")
    parser.add_argument("--file-path", type=str,default=_file_path,
                        help="graph path")
    parser.add_argument("--label-path", type=str, default=_label_path,
                        help="label path")
    parser.add_argument("--model-id", type=int, default=2,
                    help="[0, 1, 2, 3]")

    args = parser.parse_args()
    print(args)
    
    if False:
        graphs = create(args)
        max_num, max_id = 0,-1
        for idx, g in enumerate(graphs):
            if g.number_of_edges() > max_num:
                max_num = g.number_of_edges()
                max_id = idx
    torch.manual_seed(2)
    test_acc = []
    for runs in tqdm(range(10)):

        g,labels = read_struct_net(args)
        valid_mask = None
        if True:
            g.remove_edges_from(nx.selfloop_edges(g))

        g, labels = constructDGL(g, labels)

        labels = torch.LongTensor(labels)
        
        degree_emb = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1, [100, args.n_hidden])), requires_grad=False)

        if True:
            features = degree_bucketing(g, args, degree_emb)
        else:
            features = spectral_feature(g, args)

        train_mask, test_mask = createTraining(labels, valid_mask)

        if True:
            if hasattr(torch, 'BoolTensor'):
                train_mask = torch.BoolTensor(train_mask)
                #val_mask = torch.BoolTensor(val_mask)
                test_mask = torch.BoolTensor(test_mask)
            else:
                train_mask = torch.ByteTensor(train_mask)
                #val_mask = torch.ByteTensor(val_mask)
                test_mask = torch.ByteTensor(test_mask)
        # embed()
        in_feats = features.shape[1]
        n_classes = labels.max().item() + 1
        n_edges = g.number_of_edges()

        if args.gpu < 0:
            cuda = False
        else:
            cuda = True
            torch.cuda.set_device(args.gpu)
            features = features.cuda()
            labels = labels.cuda()

        g.readonly()
        n_edges = g.number_of_edges()

        # create DGI model
        if args.model_type == 1:
            dgi = VGAE(g,
                in_feats,
                args.n_hidden,
                args.n_hidden,
                args.dropout)
            
            dgi.prepare()
            
            dgi.adj_train = sp.csr_matrix(output_adj(g))

        elif args.model_type == 0:
            dgi = DGI(g,
                    in_feats,
                    args.n_hidden,
                    args.n_layers,
                    nn.PReLU(args.n_hidden),
                    args.dropout)
        elif args.model_type == 2:
            dgi = SubGI(g,
                    in_feats,
                    args.n_hidden,
                    args.n_layers,
                    nn.PReLU(args.n_hidden),
                    args.dropout,
                    args.model_id)

        if cuda:
            dgi.cuda()

        dgi_optimizer = torch.optim.Adam(dgi.parameters(),
                                        lr=args.dgi_lr,
                                        weight_decay=args.weight_decay)

        cnt_wait = 0
        best = 1e9
        best_t = 0
        dur = []
        g.ndata['features'] = features
        for epoch in range(args.n_dgi_epochs):
            train_sampler = dgl.contrib.sampling.NeighborSampler(g, 256, 5,
                                                                    neighbor_type='in', num_workers=1,
                                                                    add_self_loop=False,
                                                                    num_hops=args.n_layers + 1, shuffle=True)
            dgi.train()
            if epoch >= 3:
                t0 = time.time()
            
            loss = 0.0
            # VGAE mode
            if args.model_type == 1:
                dgi.optimizer = dgi_optimizer
                dgi.train_sampler = train_sampler
                dgi.features = features
                loss = dgi.train_model()
            # EGI mode
            elif args.model_type == 2:
                #if True:
                for nf in train_sampler:
                    dgi_optimizer.zero_grad()
                    l = dgi(features, nf)
                    l.backward()
                    loss += l
                    dgi_optimizer.step()
            # DGI mode
            elif args.model_type == 0:
                dgi_optimizer.zero_grad()
                loss = dgi(features)
                loss.backward()
                dgi_optimizer.step()
                #loss = loss.item()
            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                torch.save(dgi.state_dict(), 'best_classification_{}.pkl'.format(args.model_type))
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!')
                break

            if epoch >= 3:
                dur.append(time.time() - t0)



        # create classifier model
        classifier = MultiClassifier(args.n_hidden, n_classes)
        if cuda:
            classifier.cuda()

        classifier_optimizer = torch.optim.Adam(classifier.parameters(),
                                                lr=args.classifier_lr,
                                                weight_decay=args.weight_decay)

        # flags used for transfer learning
        if args.data_src != args.data_id:
            pass
        else:
            dgi.load_state_dict(torch.load('best_classification_{}.pkl'.format(args.model_type)))

        with torch.no_grad():
            if args.model_type == 1:
                _, embeds, _ = dgi.forward(features)
            elif args.model_type == 2:
                embeds = dgi.encoder(features, corrupt=False)
            elif args.model_type == 0:
                embeds = dgi.encoder(features)
            else:
                dgi.eval()
                test_sampler = dgl.contrib.sampling.NeighborSampler(g, g.number_of_nodes(), -1,  # 0,
                                                                            neighbor_type='in', num_workers=1,
                                                                            add_self_loop=False,
                                                                            num_hops=args.n_layers + 1, shuffle=False)
                for nf in test_sampler:
                    nf.copy_from_parent()
                    embeds = dgi.encoder(nf, False)
                    print("test flow")

        embeds = embeds.detach()

        dur = []
        for epoch in range(args.n_classifier_epochs):
            classifier.train()
            if epoch >= 3:
                t0 = time.time()

            classifier_optimizer.zero_grad()
            preds = classifier(embeds)
            loss = F.nll_loss(preds[train_mask], labels[train_mask])
            loss.backward()
            classifier_optimizer.step()
            
            if epoch >= 3:
                dur.append(time.time() - t0)
        acc = evaluate(classifier, embeds, labels, test_mask)
        
        test_acc.append(acc)
        
    print("Test Accuracy {:.4f}, std {:.4f}".format(np.mean(test_acc), np.std(test_acc)))



def gen_baseline_embs(edge_index, x, alg):
    print(f' started to gen embs from EGI/DGI ... ')
    def dgi_default_args(emb_dim, alg):
        class D: pass
        args = D()
        args.dropout=0.0
        args.gpu=-1
        args.dgi_lr=1e-2
        args.classifier_lr=1e-2
        args.n_hidden = emb_dim
        args.n_layers=1
        args.weight_decay=0.
        args.patience=20
        args.model=False
        args.self_loop=True
        args.model_type = {'EGI':2, 'DGI':0, 'VGAE':1,   }[alg]
        args.graph_type="DD"
        args.data_id=''
        args.data_src=''
        args.model_id=2
        args.n_dgi_epochs = [100, 2][0]
        args.n_classifier_epochs = [100, 2][0]

        return args


    args = dgi_default_args(x.shape[1], alg)

    g,labels = read_struct_net_v2(edge_index, len(x))
    valid_mask = None
    if True:
        g.remove_edges_from(nx.selfloop_edges(g))

    g, labels = constructDGL(g, labels)

    labels = torch.LongTensor(labels)
    
    degree_emb = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1, [100, args.n_hidden])), requires_grad=False)

    if True:
        features = degree_bucketing(g, args, degree_emb)
    else:
        features = spectral_feature(g, args)
    

    train_mask, test_mask = createTraining(labels, valid_mask)

    if True:
        if hasattr(torch, 'BoolTensor'):
            train_mask = torch.BoolTensor(train_mask)
            test_mask = torch.BoolTensor(test_mask)
        else:
            train_mask = torch.ByteTensor(train_mask)
            test_mask = torch.ByteTensor(test_mask)

    in_feats = features.shape[1]
    n_classes = labels.max().item() + 1
    n_edges = g.number_of_edges()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()

    

    g.readonly()
    n_edges = g.number_of_edges()

    # create DGI model
    if args.model_type == 1:
        dgi = VGAE(g,
            in_feats,
            args.n_hidden,
            args.n_hidden,
            args.dropout)
        dgi.prepare()
        dgi.adj_train = sp.csr_matrix(output_adj(g))

    elif args.model_type == 0:
        dgi = DGI(g,
                in_feats,
                args.n_hidden,
                args.n_layers,
                nn.PReLU(args.n_hidden),
                args.dropout)
    elif args.model_type == 2:
        dgi = SubGI(g,
                in_feats,
                args.n_hidden,
                args.n_layers,
                nn.PReLU(args.n_hidden),
                args.dropout,
                args.model_id)

    if cuda:
        dgi.cuda()

    dgi_optimizer = torch.optim.Adam(dgi.parameters(),
                                    lr=args.dgi_lr,
                                    weight_decay=args.weight_decay)

    cnt_wait = 0
    best = 1e9
    best_t = 0
    dur = []
    g.ndata['features'] = features
    for epoch in range(args.n_dgi_epochs):
        train_sampler = dgl.contrib.sampling.NeighborSampler(g, 256, 5,
                                                                neighbor_type='in', num_workers=1,
                                                                add_self_loop=False,
                                                                num_hops=args.n_layers + 1, shuffle=True)
        dgi.train()
        if epoch >= 3:
            t0 = time.time()
        
        loss = 0.0
        # VGAE mode
        if args.model_type == 1:
            dgi.optimizer = dgi_optimizer
            dgi.train_sampler = train_sampler
            dgi.features = features
            loss = dgi.train_model()
        # EGI mode
        elif args.model_type == 2:
            #if True:
            for nf in train_sampler:
                dgi_optimizer.zero_grad()
                l = dgi(features, nf)
                l.backward()
                loss += l
                dgi_optimizer.step()
        # DGI mode
        elif args.model_type == 0:
            dgi_optimizer.zero_grad()
            loss = dgi(features)
            loss.backward()
            dgi_optimizer.step()
            #loss = loss.item()
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(dgi.state_dict(), 'best_classification_{}.pkl'.format(args.model_type))
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

    with torch.no_grad():
        if args.model_type == 1:
            _, embeds, _ = dgi.forward(features)
        elif args.model_type in [0,2]:
            embeds = dgi.encoder(features, corrupt=False)
    embeds = embeds.detach()

    print(f' gen embs from EGI/DGI finished ! ! ')
    
    return embeds


if __name__ == '__main__':
    # the function to be called is gen_baseline_embs()
    main_bkup()