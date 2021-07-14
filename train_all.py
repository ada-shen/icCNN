## train files
from densenet_ori_train import densenet_ori_train
from densenet_iccnn_train import densenet_single_train
from densenet_iccnn_multi_train import densenet_multi_train
#from vgg_train import
#from resnet_train import
#resnet
from resnet_iccnn_multi_train import resnet_multi_train
from resnet_iccnn_train import resnet_single_train
from resnet_ori_train import resnet_ori_train ###
#vgg
from vgg_iccnn_train import vgg_single_train
from vgg_iccnn_multi_train import vgg_multi_train
from vgg_ori_train import vgg_ori_train
##
import argparse
import random
import os
import numpy as np
import torch

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    torch.set_num_threads(5)    

    set_seed(0)

    parser = argparse.ArgumentParser()

    # add positional arguments
    parser.add_argument('-type', type=str, help='the type of train model ori/iccnn')
    parser.add_argument('-is_multi', type=int, help='single/multi 0/1')
    parser.add_argument('-model', type=str, help='vgg/resnet/densenet')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.model == 'densenet':
        if args.type == 'ori':
            densenet_ori_train()
        elif args.type == 'iccnn':
            if args.is_multi == 0:
                densenet_single_train()
            elif args.is_multi == 1:
                densenet_multi_train()
    elif args.model == 'resnet':
        if args.type == 'ori':
            resnet_ori_train()
        elif args.type == 'iccnn':
            if args.is_multi == 0:
                resnet_single_train()
            else:
                resnet_multi_train()
    elif args.model == 'vgg':
        if args.type == 'ori':
            vgg_ori_train()
        elif args.type == 'iccnn':
            if args.is_multi == 0:
                vgg_single_train()
            else:
                vgg_multi_train()
    else:
        raise Exception("Not Implemented!")

