import os
import numpy as np
import torch
import random
from learner import Learner
import argparse

# init learner
def init_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Partition-and-Debias (ICCV 2022)')
    parser.add_argument("--model", help="which network, [ResNet18, PnD]", default= 'PnD', type=str)
    parser.add_argument("--batch_size", help="batch_size", default=128, type=int)
    parser.add_argument("--lr1",help='learning rate',default=0.001, type=float)
    parser.add_argument("--lr2",help='learning rate',default=0.0005, type=float)
    parser.add_argument("--lr_decay_step1", help="learning rate decay steps", type=int, default=20)
    parser.add_argument("--lr_decay_step2", help="learning rate decay steps", type=int, default=20)
    parser.add_argument("--lr_gamma",  help="lr gamma", type=float, default=0.5)
    parser.add_argument("--temperature",help='temperature',default=0.1, type=float)
    parser.add_argument("--weight_decay",help='weight_decay',default=1e-5, type=float)
    parser.add_argument("--loss_q",help='loss_q',default=0.7, type=float)
    parser.add_argument("--alpha1",help='wdc',default=0.2, type=float)
    parser.add_argument("--alpha2",help='wdc',default=2.0, type=float)
    parser.add_argument("--beta",  help="beta", type=float, default=4.0)
    parser.add_argument("--num_workers", help="workers number", default=16, type=int)
    parser.add_argument("--device", help="cuda or cpu", default='cuda:5', type=str)
    parser.add_argument("--num_epochs1", help="# of epochs", default= 70, type=int) 
    parser.add_argument("--num_epochs2", help="# of epochs", default= 100, type=int)
    parser.add_argument("--dataset", help="data to train", default= 'bmnist', type=str)
    parser.add_argument("--percent", help="percentage of conflict", default= "1pct", type=str)
    parser.add_argument("--target_attr_idx", help="target_attr_idx", default= 0, type=int)
    parser.add_argument("--q", help="GCE parameter q", type=float, default=0.7)
    parser.add_argument("--gamma",help='gamma',default=2, type=float)
    parser.add_argument("--ema_alpha",  help="use weight mul", type=float, default=0.7)
    parser.add_argument("--exp", help='experiment name', default='debugging', type=str)   
    parser.add_argument("--log_dir", help='path for saving model', default='./log', type=str)
    parser.add_argument("--data_dir", help='path for loading data', default='dataset', type=str)
    parser.add_argument("--seed", help="seed", default=1997, type=int)
    args = parser.parse_args()


    init_seeds(args.seed)
    # init learner
    learner = Learner(args)

    # actual training
    print('Training starts ...')
    print(" # PID :", os.getpid())

    learner.train_ours_step1(args)
    learner.train_ours_step2(args)
    learner.test_ours()
