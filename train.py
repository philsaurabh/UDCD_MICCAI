from utils.utils import get_labels_frequency, set_logger
from utils.trainer import fit
from models.model import DenseNet121
from data.loader import load_dataset
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import random
import logging
import sys
import os
import argparse
import warnings
warnings.simplefilter('ignore')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../Datasets/APTOS/APTOS_images/train_images')
    parser.add_argument('--csv_file_path', type=str, default='../CSVs/')
    parser.add_argument("--logdir", type=str, required=False,
                        default="./logs/aptos/", help="Log directory path")
    parser.add_argument('--dataset', type=str, default='aptos')
    parser.add_argument('--split', type=str, default='split1')

    parser.add_argument('--n_distill', type=int, default=20,
                        help='start to use the kld loss')

    parser.add_argument('--mode', default='exact', type=str,
                        choices=['exact', 'relax', 'multi_pos'])
    parser.add_argument('--nce_p', default=1, type=int,
                        help='number of positive samples for NCE')
    parser.add_argument('--nce_k', default=4096, type=int,
                        help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float,
                        help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float,
                        help='momentum for non-parametric updates')
    parser.add_argument('--CCD_mode', type=str,
                        default="sup", choices=['sup', 'unsup'])
    parser.add_argument('--rel_weight', type=float, default=25,
                        help='whether use the relation loss')
    parser.add_argument('--ccd_weight', type=float,
                        default=0.1, help='whether use the CCD loss')

    parser.add_argument('--anchor_type', type=str,
                        default="center", choices=['center', 'class'])
    parser.add_argument('--class_anchor', default=30, type=int,
                        help='number of anchors in each class')

    parser.add_argument('--feat_dim', type=int, default=128,
                        help='reduced feature dimension')
    parser.add_argument('--s_dim', type=int, default=128,
                        help='feature dim of the student model')
    parser.add_argument('--t_dim', type=int, default=128,
                        help='feature dim of the EMA teacher')
    parser.add_argument('--n_data', type=int, default=3662,
                        help='total number of training samples.')
    parser.add_argument('--t_decay', type=float,
                        default=0.99, help='ema_decay')

    parser.add_argument('--epochs', type=int,  default=80,
                        help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch_size per gpu')
    parser.add_argument('--drop_rate', type=int,
                        default=0, help='dropout rate')
    parser.add_argument('--lr', type=float,  default=1e-4,
                        help='learning rate')
    parser.add_argument('--seed', type=int,  default=2024, help='random seed')

    parser.add_argument('--optimizer', type=str,  default='adam', help='optim')
    parser.add_argument('--scheduler', type=str,
                        default='OneCycleLR', help='sch_str')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')

    parser.add_argument('--consistency', type=float,
                        default=1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=30, help='consistency_rampup')

    args = parser.parse_args()
    return args

# Function to set the seed for all random number generators to ensure reproducibility


def set_seed(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    # Get arguments
    args = get_args()

    # Set seed
    set_seed(args.seed)

    # Set Logger
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logger = set_logger(args)
    logger.info(args)

    # Loading Data
    train_ds, test_ds = load_dataset(args, p=args.nce_p, mode=args.mode)
    n_classes = test_ds.n_classes
    class_index = train_ds.class_index
    print(n_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=12, pin_memory=True,
                          worker_init_fn=worker_init_fn)

    test_dl = DataLoader(test_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=12, pin_memory=True,
                         worker_init_fn=worker_init_fn)
    freq = get_labels_frequency(args.csv_file_path + args.dataset +
                                '/' + args.split + '_train.csv', 'diagnosis', 'id_code')
    freq = freq.values
    weights = freq.sum() / freq
    print(weights)

    # Loading Models
    student = DenseNet121(hidden_units=args.feat_dim,
                          out_size=n_classes, drop_rate=args.drop_rate)
    teacher = DenseNet121(hidden_units=args.feat_dim,
                          out_size=n_classes, drop_rate=args.drop_rate)

    for param in teacher.parameters():
        param.detach_()

    # Fit the model
    fit(student, teacher, train_dl, test_dl, weights,
        class_index, logger, args, device=args.device)
