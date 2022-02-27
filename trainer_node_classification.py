from networkx.convert_matrix import to_numpy_array
from numpy.lib import npyio
from utils import *
from GNN_model.GNN_normalizations import TeacherGNN
from MLP_model import SEMLP, GraphMLP


class trainer:
    # This class manages the entire process. It will first load data in the trainer.__init__() method, then accomplish training in the trainer.main() method.
    def main(self):
        want_viz_tsne = 0
        if want_viz_tsne:
            viz_tsne(self.data)
        if self.args.do_deg_analyze:
            save_graph_analyze(self.args.N_nodes, self.data, self.args.use_special_split)
        if self.args.train_which in ['TeacherGNN']:
            results_arr2D = self.train_teacherGNN()
        elif self.args.train_which in ['StudentBaseMLP']:
            self.args.SEMLP__downgrade_to_MLP = 1
            results_arr2D = self.train_seMLP_part2()
        elif self.args.train_which in ['LP']:
            results_arr2D = self.run_pureLP()
        elif self.args.train_which in ['SEMLP']:
            if not self.args.SEMLP__downgrade_to_MLP:
                self.train_seMLP_part1()
            results_arr2D = self.train_seMLP_part2()
        elif self.args.train_which in ['GraphMLP']:
            self.args.SEMLP__downgrade_to_MLP = 1
            results_arr2D = self.train_seMLP_part2()
        return  results_arr2D


    def run_pureLP(self):
        from Label_propagation_model.outcome_correlation import label_propagation,gen_normalized_adjs,process_adj

        self.args.lpStep_alpha = 0.5
        self.args.lpStep_num_propagations = 50

        adj, D_isqrt = process_adj(self.data)
        DAD, DA, AD = gen_normalized_adjs(adj, D_isqrt)

        lp_dict = {
                'train_only': True,
                'display': False,
                'device': self.args.device,

                # below: lp only
                'idxs': ['train'],
                'alpha': self.args.lpStep_alpha,
                'num_propagations': self.args.lpStep_num_propagations,
                'A': DAD,

                # below: gat
                'labels': ['train'],
            }
        out = label_propagation(self.data, self.split_idx, **lp_dict)

        acc_train = evaluate(out, self.data.y, self.data.train_mask)
        acc_test = evaluate(out, self.data.y, ~self.data.train_mask)
        acc_train, acc_test = np.round(acc_train*100,2), np.round(acc_test*100,2)
        print('train,test acc = ', acc_train, acc_test)

        return np.array([[acc_train, acc_test]])


    def train_seMLP_part1(self,):
        # This function train the first part of Cold-Brew: map node features to teacherGNN embeddings.
        # This function does not load teacherGNN; it train teacherGNN inside this function for itsef.

        print('-'*30,'\n         Training TeacherGNN before train SEMLP\n','-'*30)
        self.train_teacherGNN()
        self.load_teacherGNN('best checkpoint')

        print('-'*30,'\n         Start training part-1 of SEMLP\n','-'*30)
        self.seMLP = SEMLP(self.args, self.data, self.teacherGNN).to(self.device)
        self.seMLP.train_idx = self.seMLP.train_idx.to('cpu')
        self.seMLP.train_mask = self.seMLP.train_mask.to('cpu')
        self.seMLP.test_idx = self.seMLP.test_idx.to('cpu')


        self.optimizer = None
        self.seMLP.optfun = self.optfun


        results_arr2D = []
        
        self.seMLP.teacherSE = lrn_targ = self.teacherGNN.model.model.collect_SE(self.data.x, self.data.edge_index)
        lossfun = nn.MSELoss()

        for epoch in range(self.epochs):
            self.seMLP.train()
            # ------ select a batch of nodes in the training node set ------
            batch_idx_train = np.random.choice(self.seMLP.train_idx, self.args.batch_size)
            part1_out = self.seMLP.forward_part1(self.data.x, batch_idx=batch_idx_train)
            if self.optimizer is None:
                self.optimizer = self.seMLP.opt


            loss_train = lossfun(part1_out, lrn_targ[batch_idx_train])

            self.optimizer.zero_grad()
            loss_train.backward()
            self.optimizer.step()

            if epoch%1==0:
                self.seMLP.eval()
                batch_idx_train = np.random.choice(self.seMLP.test_idx, self.args.batch_size)
                part1_out = self.seMLP.forward_part1(self.data.x, batch_idx=batch_idx_train)
                loss_test = lossfun(part1_out, lrn_targ[batch_idx_train])

                # defining what results to return:seMLP
                result = [np.log(toitem(loss_train)), np.log(toitem(loss_test))]

                results_arr2D.append(result)
                if epoch%20==0:
                    print(f'epoch {epoch}, train/test loss {toitem(loss_train):.2}, {toitem(loss_test):.2}')

        save_model(self.seMLP, join(self.modeldir,'seMLP-part-1'))
        results_arr2D = np.array(results_arr2D).T
        npy_dir = f'{self.resdir}/seMLP'
        wzRec(results_arr2D[0], f'loss_train@{npy_dir.replace("/","@")}', want_save_npy=1, npy_dir=npy_dir)
        wzRec(results_arr2D[1], f'loss_test@{npy_dir.replace("/","@")}', want_save_npy=1, npy_dir=npy_dir)
        results_arr2D_concise = results_arr2D[[1]]  # shape = [1, epochs] ; only want acc_test
        return results_arr2D_concise

    def train_seMLP_part2(self,):
        # This function finished the second part: map node embeddings and its virtual neighbors to the node logits.

        if self.args.SEMLP__downgrade_to_MLP:
            # print('-'*30,'\n         Training TeacherGNN before train SEMLP\n','-'*30)
            # self.load_teacherGNN()
            self.seMLP = SEMLP(self.args, self.data, teacherGNN=None).to(self.device)
            self.seMLP.train_idx = self.seMLP.train_idx.to('cpu')
            self.seMLP.train_mask = self.seMLP.train_mask.to('cpu')
            self.seMLP.test_idx = self.seMLP.test_idx.to('cpu')

        print('-'*30,'\n         Start training part2 of SEMLP\n','-'*30)
        
        self.optimizer = None
        self.seMLP.optfun = self.optfun
        
        results_arr2D = []
        
        lrn_targ = self.data.y
        lossfun = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            self.seMLP.train()

            # ------ select a batch of nodes in the training node set ------
            batch_idx_train = np.random.choice(self.seMLP.train_idx, self.args.batch_size)
            part2_out_train = self.seMLP.forward_part2(self.data.x, edge_index=self.data.edge_index, batch_idx=batch_idx_train)
            if self.optimizer is None:
                self.optimizer = self.seMLP.opt

            loss_train = lossfun(part2_out_train, lrn_targ[batch_idx_train])
            if self.args.train_which=='GraphMLP':
                loss_train += self.seMLP.loss_NContrastive * self.args.graphMLP_reg

            self.optimizer.zero_grad()
            loss_train.backward()
            self.optimizer.step()

            if epoch%1==0:
                self.seMLP.eval()

                batch_idx_train = np.random.choice(self.seMLP.test_idx, self.args.batch_size)
                part2_out_test = self.seMLP.forward_part2(self.data.x, edge_index=self.data.edge_index, batch_idx=batch_idx_train)

                acc_test = evaluate(part2_out_test, self.data.y[batch_idx_train], None)*100
                result = [toitem(acc_test)]

                if self.args.want_headtail:

                    batch_idx = self.data.large_deg_idx
                    train_head, test_head = self.eval_headtail__traintest_v2(self.seMLP.forward_part2(self.data.x, edge_index=self.data.edge_index, batch_idx=batch_idx), lrn_targ[batch_idx], batch_idx, cal_acc_rounded100)

                    batch_idx = self.data.small_deg_idx
                    train_tail, test_tail = self.eval_headtail__traintest_v2(self.seMLP.forward_part2(self.data.x, edge_index=self.data.edge_index, batch_idx=batch_idx), lrn_targ[batch_idx], batch_idx, cal_acc_rounded100)

                    head_tail_iso = [test_head, test_tail]
                    result.extend(head_tail_iso)

                    if self.args.use_special_split:
                        batch_idx = self.data.zero_deg_idx
                        train_iso, test_iso = self.eval_headtail__traintest_v2(self.seMLP.forward_part2(self.data.x, edge_index=self.data.edge_index, batch_idx=batch_idx), lrn_targ[batch_idx], batch_idx, cal_acc_rounded100)
                        result.extend([test_iso])

                results_arr2D.append(result)
                if epoch%20==0:
                    # print(f'epoch {epoch}, acc test {toitem(acc_test):.2}')
                    print(f'epoch {epoch}, acc test {toitem(acc_test):.2f}, head_tail_iso = {result[-3:]}')  if self.args.use_special_split else print(f'epoch {epoch}, acc test {toitem(acc_test):.2}')


        save_model(self.seMLP, join(self.modeldir,'seMLP'))
        results_arr2D = np.array(results_arr2D).T
        npy_dir = f'{self.resdir}/seMLP'
        wzRec(results_arr2D[0], f'acc_test@{npy_dir.replace("/","@")}', want_save_npy=1, npy_dir=npy_dir)

        figure()
        plot_many(results_arr2D[[0]], ['acc-test'],ttl='#ALL2#__'+npy_dir.replace("/","@"))
        if not self.args.want_headtail:
            results_arr2D_concise = results_arr2D[[0]]  # shape = [1, epochs] ; only want acc_test
        else:
            results_arr2D_concise = results_arr2D[[0,-3,-2,-1]]  # shape = [4, epochs]

        return results_arr2D_concise




    def eval_headtail__traintest(self, emb2, lrn_targ, metricfun):
        
        mask_head_train = self.data.train_mask * self.data.large_deg_mask
        mask_head_test = (~self.data.train_mask) * self.data.large_deg_mask
        mask_tail_train = self.data.train_mask * (~self.data.large_deg_mask)
        mask_tail_test = (~self.data.train_mask) * (~self.data.large_deg_mask)

        metric_head_train = metricfun(emb2[mask_head_train], lrn_targ[mask_head_train])
        metric_head_test = metricfun(emb2[mask_head_test], lrn_targ[mask_head_test])
        metric_tail_train = metricfun(emb2[mask_tail_train], lrn_targ[mask_tail_train])
        metric_tail_test = metricfun(emb2[mask_tail_test], lrn_targ[mask_tail_test])
        return [metric_head_train,metric_head_test,metric_tail_train,metric_tail_test]


    def eval_headtail__traintest_v2(self, emb2, lrn_targ, subsets, metricfun):
        # emb2/lrn_target: not full node emb, but a subset, whose index is given by indices
        # subsets can be either head or tail, in either indices or mask
        actual_train_mask = self.data.train_mask[subsets]
        on_train = torch.where(actual_train_mask)[0]
        on_test = torch.where(~actual_train_mask)[0]

        metric_train = metricfun(emb2[on_train], lrn_targ[on_train])
        metric_test = metricfun(emb2[on_test], lrn_targ[on_test])
        return metric_train, metric_test

    def load_teacherGNN(self, keyw=''):
        if self.args.has_proj2class:
            self.proj2class = getMLP(self.args.TeacherGNN.neurons_proj2class).to(self.device)
        else:
            self.proj2class = None
        self.teacherGNN = TeacherGNN(self.args, self.proj2class).to(self.device)
        if 'best' in keyw:
            load_model(self.teacherGNN, join(self.modeldir,'best-teacherGNN'))
            print('\n\n\nloading best techerGNN ckpt!\n\n\n')
        else:
            load_model(self.teacherGNN, join(self.modeldir,'teacherGNN'))

        return


    def __init__(self, args, which_run):
        self.bag = {}
        self.is_large_dataset = False
        self.which_run = which_run
        self.args = args
        self.dataset = args.dataset
        self.device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
        args.device = self.device
        if self.dataset == 'ogbn-arxiv':
            self.data, self.split_idx = load_ogbn(self.dataset)
            self.data.to(self.device)
            self.train_idx = self.split_idx['train'].to(self.device)
            self.evaluator = Evaluator(name='ogbn-arxiv')
            self.loss_fn = torch.nn.functional.nll_loss

            if not self.args.use_special_split:
                self.data.train_mask = torch.BoolTensor([False]*self.args.N_nodes).to(self.device)
                self.data.test_mask = torch.BoolTensor([False]*self.args.N_nodes).to(self.device)
                self.data.train_mask[self.train_idx] = True
                self.data.test_mask[self.split_idx['test'].to(self.device)] = True
        else:
            self.data = load_data(self.dataset, self.which_run, self)
            self.split_idx = {'train':self.data.train_mask, 'valid':self.data.val_mask, 'test': self.data.test_mask}
            self.data.train_idx = torch.where(self.data.train_mask==True)[0]
            self.loss_fn = torch.nn.functional.nll_loss
        

        self.type_model = args.type_model
        self.type_trick = args.type_trick
        self.epochs = args.epochs
        self.num_layers = args.num_layers
        self.dim_hidden = args.dim_hidden
        self.weight_decay = args.weight_decay
        self.records_path = args.records_path
        self.records_desc = args.records_desc
        self.records_file = args.records_file

        self.data.x = self.data.x.float()
        self.modeldir = f'saved_models/{args.task}/{args.dataset}'
        self.resdir = f'{self.args.task}/{self.args.dataset}'
        os.makedirs(self.modeldir,exist_ok=1)
        if args.optfun=='torch.optim.Adam':
            self.optfun = torch.optim.Adam
        elif args.optfun=='torch.optim.SGD':
            self.optfun = torch.optim.SGD

        set_arch_configs(args)
        self.args.data = self.data

        return

    def train_teacherGNN(self):
        if self.args.has_proj2class:
            self.proj2class = getMLP(self.args.TeacherGNN.neurons_proj2class).to(self.device)
        else:
            self.proj2class = None

        self.teacherGNN = TeacherGNN(self.args, self.proj2class).to(self.device)
        self.optimizer = self.optfun(self.teacherGNN.parameters(),lr=self.args.lr, weight_decay=self.weight_decay)

        best_train_loss = 100
        best_test_acc = 0.
        best_train_acc = 0.
        best_val_loss = 100.
        patience = self.args.patience
        bad_counter = 0.
        val_loss_history = []

        results_arr2D = []

        tmp = []

        for epoch in range(self.epochs):
            self.epoch = epoch

            acc_train, acc_val, acc_test, loss_train, loss_val, linkp_train, linkp_test = self.train_net()

            val_loss_history.append(loss_val)

            keep_saved_teacher_model_strongest = 'SEMLP' in self.args.train_which
            if keep_saved_teacher_model_strongest and acc_test > best_test_acc:
                best_test_acc = acc_test
                save_model(self.teacherGNN, join(self.modeldir,'best-teacherGNN'))

            # ---- defining what results to return:TeacherGNN ----
            results_arr2D.append([np.log(loss_train), acc_train*100, acc_test*100, linkp_train, linkp_test])
            if self.args.want_headtail:
                results_arr2D[-1].extend(self.bag['head_tail_iso'])

            if epoch%20 == 0:
                if self.args.has_loss_component_edgewise:
                    print(f'Ep{epoch:03d}, linkp train/test mrr: {linkp_train:.3f} / {linkp_test:.3f}')
                else:
                    print(f'Ep{epoch:03d}, acc @ train/test: {acc_train*100:.1f}, {acc_test*100:.1f} | ', f"head_tail_iso: {self.bag['head_tail_iso']}" if self.args.want_headtail else '')



        print('train_loss: {:.4f},  test_acc:{:.4f}'
              .format(best_train_loss, best_test_acc))
        save_model(self.teacherGNN, join(self.modeldir,'teacherGNN'))

        results_arr2D = np.array(results_arr2D).T
        npy_dir = f'{self.resdir}/teacherGNN'
        wzRec(results_arr2D[0], f'loss_train@{npy_dir.replace("/","@")}', want_save_npy=1, npy_dir=npy_dir)
        wzRec(results_arr2D[1], f'acc_train@{npy_dir.replace("/","@")}', want_save_npy=1, npy_dir=npy_dir)
        wzRec(results_arr2D[2], f'acc_test@{npy_dir.replace("/","@")}', want_save_npy=1, npy_dir=npy_dir)
        wzRec(results_arr2D[3], f'linkp_train@{npy_dir.replace("/","@")}', want_save_npy=1, npy_dir=npy_dir)
        wzRec(results_arr2D[4], f'linkp_test@{npy_dir.replace("/","@")}', want_save_npy=1, npy_dir=npy_dir)

        figure()
        plot_many(results_arr2D[[1,2,3,4]], ['acc_train', 'acc_test','linkp_train','linkp_test',],ttl='#ALL1#__'+npy_dir.replace("/","@"))
        figure()
        plot_many(results_arr2D[[0]], ['log_loss_train'],ttl='#ALL2#__'+npy_dir.replace("/","@"))
        if not self.args.want_headtail:
            results_arr2D_concise = results_arr2D[[2]]  # shape = [1, epochs]
        else:
            results_arr2D_concise = results_arr2D[[2,-3,-2,-1]]  # shape = [4, epochs]
        return results_arr2D_concise





    def train_net(self):
        loss_train, linkp_train, linkp_test = self.run_trainSet()
        acc_train, acc_val, acc_test, loss_val = self.run_testSet()
        return acc_train, acc_val, acc_test, loss_train, loss_val, linkp_train, linkp_test



    def run_trainSet(self):
        self.teacherGNN.train()
        loss, linkp_train, linkp_test = -1, 0, 0
        assert self.args.has_loss_component_nodewise or self.args.has_loss_component_edgewise, 'setting no node-wise and no edge-wise loss for teacherGNN! at least set one of them!'
        res = self.teacherGNN.get_3_embs(self.data.x, self.data.edge_index, self.data.train_mask)
        raw_logits, emb4classi_full, emb4linkp = res.emb4classi, res.emb4classi_full, res.emb4linkp
        if self.args.has_loss_component_nodewise:
            # ========= classification: train =========
            logits = F.log_softmax(raw_logits, 1)
            loss_semantic = self.loss_fn(logits, self.data.y[self.data.train_mask])
            loss = loss_semantic*self.args.TeacherGNN.lossa_semantic
            if self.teacherGNN.se_reg_all is not None:
                loss += self.args.se_reg * self.teacherGNN.se_reg_all
            
            result = []
            if self.args.want_headtail:
                all_node_logits = self.teacherGNN.get_3_embs(self.data.x, self.data.edge_index).emb4classi
                lrn_targ = self.data.y

                batch_idx = self.data.large_deg_idx
                train_head, test_head = self.eval_headtail__traintest_v2(all_node_logits[batch_idx], lrn_targ[batch_idx], batch_idx, cal_acc_rounded100)

                batch_idx = self.data.small_deg_idx
                train_tail, test_tail = self.eval_headtail__traintest_v2(all_node_logits[batch_idx], lrn_targ[batch_idx], batch_idx, cal_acc_rounded100)

                head_tail_iso = [test_head, test_tail]
                result.extend(head_tail_iso)

                if self.args.use_special_split:
                    batch_idx = self.data.zero_deg_idx
                    train_iso, test_iso = self.eval_headtail__traintest_v2(all_node_logits[batch_idx], lrn_targ[batch_idx], batch_idx, cal_acc_rounded100)
                    result.extend([test_iso])

            self.bag['head_tail_iso'] = result

        if self.args.has_loss_component_edgewise:
            emb4linkp = res.commonEmb  # for linkp, must use full node embs (without applying train_mask!!)
            # ======= link prediction: train =======
            loss_structure, linkp_train = self.getLinkp_loss_eva(emb4linkp, 'train')
            # ======= link prediction: eva =======
            _, linkp_test = self.getLinkp_loss_eva(emb4linkp, 'test')
            if loss is None:
                loss = loss_structure*self.args.TeacherGNN.lossa_structure
            else:
                loss = loss + loss_structure*self.args.TeacherGNN.lossa_structure

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), linkp_train, linkp_test


    def getLinkp_loss_eva(self, emb, mode):
        h_emb, t_emb, nh_emb, nt_emb = self.gen_pn_edges(emb, mode)
        loss, eva_score = linkp_loss_eva(h_emb, t_emb, nh_emb, nt_emb)
        return loss, eva_score

    def evaluate_linkp(self, model, mode):
        # mode: train / test / all
        if mode=='train':
            mask = self.data.train_mask
        elif mode=='test':
            mask = ~self.data.train_mask
        emb4linkp = model.get_emb4linkp(self.data.x, self.data.edge_index, mask) # return ALL nodes
        h_emb, t_emb, nh_emb, nt_emb = self.gen_pn_edges(emb4linkp, mode)
        _, eva_score = linkp_loss_eva(h_emb, t_emb, nh_emb, nt_emb)
        return eva_score



    def run_testSet(self):
        self.teacherGNN.eval()

        with torch.no_grad():
            if self.args.train_which=='GraphSAGE':
                if not self.epoch%100==0:
                    return -1,-1,-1,-1
                input_dict = {'x': self.data.x, 'y': self.data.y, 'device': self.args.device}
                raw_logits = self.teacherGNN.inference(input_dict)

                result = []
                if self.args.want_headtail:
                    lrn_targ = self.data.y
                    all_node_logits = raw_logits.to(lrn_targ.device)
                    
                    batch_idx = self.data.large_deg_idx
                    train_head, test_head = self.eval_headtail__traintest_v2(all_node_logits[batch_idx], lrn_targ[batch_idx], batch_idx, cal_acc_rounded100)

                    batch_idx = self.data.small_deg_idx
                    train_tail, test_tail = self.eval_headtail__traintest_v2(all_node_logits[batch_idx], lrn_targ[batch_idx], batch_idx, cal_acc_rounded100)

                    head_tail_iso = [test_head, test_tail]
                    result.extend(head_tail_iso)

                    if self.args.use_special_split:
                        batch_idx = self.data.zero_deg_idx
                        train_iso, test_iso = self.eval_headtail__traintest_v2(all_node_logits[batch_idx], lrn_targ[batch_idx], batch_idx, cal_acc_rounded100)

                        result.extend([test_iso])

                self.bag['head_tail_iso'] = result

            else:
                res = self.teacherGNN.get_3_embs(self.data.x, self.data.edge_index)
                raw_logits = res.emb4classi


        logits = F.log_softmax(raw_logits, 1)
        acc_train = evaluate(logits, self.data.y, self.data.train_mask)
        acc_val = np.nan #evaluate(logits, self.data.y, self.data.val_mask)
        acc_test = evaluate(logits, self.data.y, self.data.test_mask)
        val_loss = np.nan # self.loss_fn(logits[self.data.val_mask], self.data.y[self.data.val_mask])
        return acc_train, acc_val, acc_test, val_loss

    def filename(self, filetype='params'):
        filedir = f'./{filetype}/{self.dataset}'

        filename = self.args.type_model + f'.pth.tar'
        filename = os.path.join(filedir, filename)

        return filename



    def gen_pn_edges(self, nodes_emb, mode):
        # nodes_emb must contain all nodes
        # mode: train / test 
        # assert self.args.use_special_split  # otherwise not implemented
        if mode=='train':
            valid_edge_mask = self.data.train_mask[self.data.edge_index[0]] * self.data.train_mask[self.data.edge_index[1]]  # len = edge_index
            valid_edge_index = self.data.edge_index[:,valid_edge_mask]  # len = num of true
        elif mode=='test':
            test_mask = ~ self.data.train_mask            
            valid_edge_mask = test_mask[self.data.edge_index[0]] * test_mask[self.data.edge_index[1]]  # len = edge_index_bkup
            valid_edge_index = self.data.edge_index[:,valid_edge_mask]

        else:
            raise NotImplementedError
        samp_size_p = self.args.samp_size_p

        samp_edge_p_idx = np.random.choice(valid_edge_index.shape[1], samp_size_p)
        samp_edge_p = valid_edge_index[:,samp_edge_p_idx]
        samp_edge_n = self.my_negative_sampling(mode)


        h_emb = nodes_emb[samp_edge_p[0]]
        t_emb = nodes_emb[samp_edge_p[1]]
        nh_emb = nodes_emb[samp_edge_n[0]]
        nt_emb = nodes_emb[samp_edge_n[1]]
        return h_emb, t_emb, nh_emb, nt_emb

    def my_negative_sampling(self, mode):
        # how to sample neg edge:
        # first get neg sample for all edges in the graph, then screen them according to train/test split: for training set, neg edge samples are those ori & dst nodes all falls within training split; for test set, neg edge samples are those at least one node of ori/dst falls within the test split.
        

        sampled_all = []
        N_sampled = 0
            
        if mode=='train':
            samp_size_n = self.args.samp_size_n_train
            samp_size_n_sub = max(samp_size_n//4, 50) # do neg sample in small batches to prevend over flood
            while N_sampled<samp_size_n:
                edge_samp = negative_sampling(self.data.edge_index, num_neg_samples=samp_size_n_sub, force_undirected=True)
                fall_in_mask = self.data.train_mask[edge_samp[0]] * self.data.train_mask[edge_samp[1]]
                edge_samp = edge_samp[:,fall_in_mask]
                N_sampled += edge_samp.size(1)
                sampled_all.append(edge_samp)

        elif mode=='test':
            samp_size_n = self.args.samp_size_p*self.args.samp_size_n_test_times_p
            samp_size_n_sub = max(samp_size_n//4, 50) # do neg sample in small batches to prevend over flood
            while N_sampled<samp_size_n:
                edge_samp = negative_sampling(self.data.edge_index, num_neg_samples=samp_size_n_sub, force_undirected=True)
                fall_in_mask = ~ (self.data.train_mask[edge_samp[0]] * self.data.train_mask[edge_samp[1]])
                edge_samp = edge_samp[:,fall_in_mask]
                N_sampled += edge_samp.size(1)
                sampled_all.append(edge_samp)

        sampled_all = torch.cat(sampled_all,dim=1)
        return sampled_all






def load_ogbn(dataset='ogbn-arxiv'):
    dataset = PygNodePropPredDataset(name=dataset, root='data')
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    data.y = data.y.squeeze(1)
    
    return data, split_idx

def change_split(data, dataset, which_split=0):
    if dataset in ["CoauthorCS", "CoauthorPhysics"]:
        data = random_coauthor_amazon_splits(data)
    elif dataset in ["AmazonComputers", "AmazonPhoto"]:
        data = random_coauthor_amazon_splits(data)
    elif dataset in ["TEXAS", "WISCONSIN", "CORNELL"]:
        data = manual_split_WebKB_Actor(data, which_split)
    elif dataset == "ACTOR":
        data = manual_split_WebKB_Actor(data, which_split)
    elif dataset in ["chameleon","squirrel"]:
        data = manual_split_WebKB_Actor(data, which_split)

    else:
        data = data
    data.y = data.y.long()
    return data


    
def convert_out_to_nparray(fdirname):
    # .out file lines look like:
    # 'BX3243\t0.0564,-0.44, ... , 0.39344\n'
    res = []
    texts = []
    with open(fdirname,'r') as f:
        while 1:
            line = f.readline()
            if line=='': break
            text, x = line.split('\t')
            x = np.array(x[:-1].split(','), dtype=float)
            res.append(x)
            texts.append(text)
    resx = np.asarray(res)
    return resx, texts



def load_data(dataset, which_run, self):
    # what to load:
    #     return a dataset, which is a namespace, called 'data', 
    #     data.x: 2D tensor, on cpu; shape = [N_nodes, dim_feature].
    #     data.y: 1D tensor, on cpu; shape = [N_nodes]; values are integers, indicating the class of nodes.
    #     data.edge_index: tensor: [2, N_edge], cpu; edges contain self loop.
    #     data.train_mask: bool tensor, shape = [N_nodes], indicating the training node set.
    #     Template class for the 'data':
    #     class MyDataset(torch_geometric.data.data.Data):
    #         def __init__(self):
    #             super().__init__()

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', dataset)

    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        data = Planetoid(path, dataset, split='public', transform=T.NormalizeFeatures())[0]
        if dataset=='Cora':
            N_nodes_train = 600
            data.train_mask = torch.tensor([True]*N_nodes_train+[False]*(self.args.N_nodes-N_nodes_train))
            self.N_nodes_train = N_nodes_train
            data.test_mask = ~ data.train_mask

    elif 'PROD' in dataset:
        from load_proprietary_datasets import load_PROD
        data = load_PROD(dataset, self)
        print()

    elif dataset in ["TEXAS", "WISCONSIN", "CORNELL"]:
        data = WebKB(path, dataset, transform=T.NormalizeFeatures())[0]
        data = change_split(data, dataset, which_split=int(which_run // 10))
    elif dataset == "ACTOR":
        data = Actor(path, transform=T.NormalizeFeatures())[0]
        data = change_split(data, dataset, which_split=int(which_run // 10))
    elif dataset in ['chameleon', 'squirrel']:
        data = WikipediaNetwork(path, dataset, transform=T.NormalizeFeatures())[0]
        data = change_split(data, dataset, which_split=int(which_run // 10))
    else:
        raise ValueError(f'the dataset of {dataset} has not been implemented')

    num_nodes = data.x.size(0)
    edge_index = ensure_symmetric(data.edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
    if isinstance(edge_index, tuple):
        data.edge_index = edge_index[0]
    else:
        data.edge_index = edge_index
    if not hasattr(data, 'test_mask') or data.test_mask is None:
        data.test_mask = ~data.train_mask
    if not hasattr(data, 'train_idx') or data.train_idx is None:
        data.train_idx = torch.where(data.train_mask==True)[0]
    if not hasattr(data, 'test_idx') or data.test_idx is None:
        data.test_idx = torch.where(data.test_mask==True)[0]
    data = data.to(self.device)
    return data

def evaluate(output, labels, mask):
    output = output.to(labels.device)
    _, indices = torch.max(output, dim=1)
    if mask is None:
        correct = torch.sum(indices == labels)
        return correct.item() / len(indices)
    else:
        mask = mask.to(labels.device)
        correct = torch.sum(indices[mask] == labels[mask])
        return correct.item() * 1.0 / mask.sum().item()

def cal_acc_rounded100(output, labels):
    # output = output.to(labels.device)
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices == labels)/len(labels)
    return toitem(correct * 100)

def AcontainsB(A, listB):
    # A: string; listB: list of strings
    for s in listB:
        if s in A: return True
    return False



