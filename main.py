# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import json
import numpy as np
import os
import random
import torch

from base_options import BaseOptions
from datetime import datetime

def main():
    args = BaseOptions().get_arguments()
    if args.exp_mode == 'coldbrew':
        from trainer_node_classification import trainer
    elif args.exp_mode == 'I2_GTL':
        from trainer_link_prediction import trainer # expected to contain codes for extended work other than cold brew; coming soon.

    if args.prog: tensorRex(None, args.prog, args.rexName)
    full_recs_3D = []
    for seed in range(args.N_exp):
        print(f'seed (which_run) = <{seed}>')

        args.random_seed = seed
        set_seed(args)
        torch.cuda.empty_cache()
        trnr = trainer(args, seed)

        results_arr2D = trnr.main()

        full_recs_3D.append(results_arr2D) # dimensions: [seeds, record_type, epochs]
        del trnr
        torch.cuda.empty_cache()
        gc.collect()

    if args.prog: tensorRex(full_recs_3D, args.prog, args.rexName)

def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    import random
    random.seed(args.random_seed)

def tensorRex(dataND, prog, rexName, support_inequal_shape=True):
    # This version is better than the previous 1D version; use this. --Aug.12.2021
        # 1. Difference from previous version: previous one only deal with mean/std, which is 1D data; this one deal with N-D data;
        # 2. share in common: direct store the input data, and do NOT increase data dimensions.

    # support function for batch running
    # how to use: call this function twice, at the begining & the end; set dataND=None at the beginning;
    # it will automatically skip experiments that are already completed

    # prog contain 3 elements: tensor_indices, vector_idx, tensor_shape, eg:
    #     '1_3_2_5__//__143__//__2x4x3x6'
    #     indicies = [1,3,2,5]
    # dataND is N-Dim
    # rec[...,0] is flag (=1 means exp is finished)
    # rec[...,1:] is dataND

    indicies, idx, shape = prog.split('__//__')
    indicies = np.array(indicies.split('_'), dtype=int)
    idx = int(idx)
    shape = list(np.array(shape.split('*'), dtype=int))

    if dataND is None:  # in the first call: only check if exp has been completed
        try:  # load or init new rec
            rec = np.load(rexName, allow_pickle=1).item()
            rec_data = rec['data']
            rec_flag = rec['flag']
            if rec_flag[tuple(indicies)] == 1.:
                raise UserWarning('\n\n\nThis exp has completed already\n\n\n')
            return
        except FileNotFoundError:
            assert idx == 0, '\n\n\nFatal Error! previous experiment file deleted!\n\n\n'
            return  # first run, first exp

    else:  # calling at the end: store dataND and exit
        dataND = np.asarray(dataND)
        try:  # load or init new rec
            rec = np.load(rexName, allow_pickle=1).item()
            rec_data = rec['data']
            rec_flag = rec['flag']
        except FileNotFoundError:
            assert idx == 0, '\n\n\nFatal Error! previous experiment file deleted!\n\n\n'
            rec_data = np.zeros(shape + list(dataND.shape), dtype=float)
            rec_flag = np.zeros(shape, dtype=float)

        if support_inequal_shape and (rec_data[tuple(indicies)].shape != dataND.shape):
            to_fill = rec_data[tuple(indicies)]
            assert len(dataND.shape)==len(to_fill.shape), f'\n\nFatal Error! new exp has different number of dims ({dataND.shape}) than existing exp ({to_fill.shape})!\n\n'
            
            def tolerantly_fill_b_in_A(b, A):
                # b has dynamic shape;
                # A has fixed shape;
                # fill b to the upper frontmost corner of A
                _shape_str = []
                for _s in range(len(b.shape)):
                    ms = min(b.shape[_s], A.shape[_s])
                    _shape_str.append(f':{ms}')
                _shape_str = ','.join(_shape_str)
                # evastr = f'A[{_shape_str}]'
                exec(f'A[{_shape_str}]=b[{_shape_str}]')
                return A
            to_fill = tolerantly_fill_b_in_A(dataND, to_fill)
            rec_data[tuple(indicies)] = to_fill
            print(f'\n\n\n tolerantly fill success!!! \n\n  to_fill is:\n{to_fill}\n\n')

        else:
            rec_data[tuple(indicies)] = dataND

        rec_flag[tuple(indicies)] = 1.
        rec = {'flag':rec_flag,'data':rec_data}
        np.save(rexName, rec)
        return

def print_line_by_line(*b, tight=False):
    print('\n')
    for x in b:
        print(x)
        if not tight: print()
    print('\n')
    return

if __name__ == "__main__":
    main()
