"""Main training entrypoint for HyperMIL on equal-length aeon datasets."""

import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from aeon.datasets import load_classification
from torch.utils.data import TensorDataset

import utils as u
from train_model import train_loop

warnings.filterwarnings("ignore")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HyperMIL training")
    parser.add_argument("--dataset", default="PenDigits", type=str, help="aeon dataset name")
    parser.add_argument("--num_workers", default=4, type=int, help="DataLoader workers")
    parser.add_argument("--num_classes", default=2, type=int, help="Number of classes (autodetected)")
    parser.add_argument("--feats_size", default=256, type=int, help="Feature dimension (autodetected)")

    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--batchsize", default=64, type=int)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "multistep", "plateau", "none"])

    parser.add_argument("--dropout_patch", default=0.5, type=float)
    parser.add_argument("--dropout_node", default=0.2, type=float)
    parser.add_argument("--loss", default="cross_entropy", choices=["cross_entropy", "bce"])
    parser.add_argument("--autocast", action="store_true", help="Use AMP")
    parser.add_argument("--unscale", action="store_true", help="Unscale AMP grads before clipping")
    parser.add_argument("--epoch_des", default=10, type=int, help="Warmup epochs")

    parser.add_argument("--encoding", default="wavelet", choices=["wavelet", "sinusoidal", "none"])
    parser.add_argument("--pooling", default="cls", choices=["cls", "mean", "max", "attention", "conjunct"])
    parser.add_argument("--num_layers", default=2, type=int)

    parser.add_argument("--k_prototypes", default=8, type=int)
    parser.add_argument("--num_convs", default=1, type=int)
    parser.add_argument("--tau", default=0.2, type=float)
    parser.add_argument("--time_decay", default=6.0, type=float)
    parser.add_argument("--activity_gate", action="store_true")
    parser.add_argument("--intra_identity", action="store_true")
    parser.add_argument("--intra_embed", default=128, type=int)
    parser.add_argument("--embed", default=128, type=int)

    parser.add_argument("--gpu_index", type=int, nargs="+", default=(0,), help="GPU IDs")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save_dir", default="./savemodel/", type=str)
    parser.add_argument("--debug", action="store_true")
    return parser


def main():
    parser = build_parser()

    
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    u.set_seed(args.seed)
    
    args.save_dir = os.path.join(args.save_dir, "HyperMIL")
    if args.debug:
        args.save_dir = os.path.join(args.save_dir, 'debug')
        u.maybe_mkdir_p(args.save_dir)
        version = 'debug'

    else:
        u.maybe_mkdir_p(os.path.join(args.save_dir, f'{args.dataset}'))
        args.save_dir, version = u.make_dirs(os.path.join(args.save_dir, f'{args.dataset}'))
        u.maybe_mkdir_p(args.save_dir)
    


    # <------------- set up logging ------------->
    logging_path = os.path.join(args.save_dir, 'Train_log.log')
    logger = u.get_logger(logging_path)
        

    # <------------- save hyperparams ------------->
    option = vars(args)
    file_name = os.path.join(args.save_dir, 'option.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(option.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
            

    # <------------- load data ------------->
    
    Xtr, ytr = load_classification(name=args.dataset,split='train')

    word_to_idx = {}
    classes = np.unique(ytr)
    for i in range(len(classes)):
            word_to_idx[classes[i]]=i

    Xtr =torch.from_numpy(Xtr).permute(0,2,1).float()
    ytr = [word_to_idx[i] for i in ytr]
    ytr =  F.one_hot(torch.tensor(ytr)).float()

    trainset = TensorDataset(Xtr,ytr)


    Xte, yte = load_classification(name=args.dataset,split='test')

    Xte =torch.from_numpy(Xte).permute(0,2,1).float()
    yte = [word_to_idx[i] for i in yte]
    yte = F.one_hot(torch.tensor(yte)).float()

    testset = TensorDataset(Xte,yte)

    args.feats_size = Xte.shape[-1]
    num_classes = yte.shape[-1]
    args.num_classes =  yte.shape[-1]
    print(f"num class:{args.num_classes}")

    seq_len=max(21, Xte.shape[1])
        
    train_loop(trainset, testset, num_classes, seq_len, args, logger)    

if __name__ == '__main__':
    main()
