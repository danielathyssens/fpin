import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Model to solve the VRP for a given number of vehicles with Supervised Learning")

    # Data
    parser.add_argument('--gpus', type=int, default=1, help="Nr of GPUs to use")
    parser.add_argument('--vrp_size', type=int, default=100, help="The size of the VRP problem")
    parser.add_argument('--resume', type=bool, default=False, help="If training should be resumed from a given chechpoints")
    parser.add_argument('--model_ckp', type=str, default='/home/thyssens/job_vrp100/outs/cps/',
                        help="path to checkpoint to load")
    parser.add_argument('--save_dir', type=str, default='/home/thyssens/job_vrp100/outs/', help="path to save checkpoints and outputs")
    parser.add_argument('--checkpoint_epochs', type=int, default=10, help="epoch frequency at which to make checkpoint")
    parser.add_argument('--batch_size', type=int, default=64, help='Number of instances per batch during training')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--n_hidden', type=int, default=1024, help='Number of hidden dimension in pooling network')
    parser.add_argument('--main_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--layers', type=int, default=9, help='Embedding dimension')
    parser.add_argument('--load', type=bool, default=False, help='calculate load estimate during training - needed for load loss and direct demand loss')
    parser.add_argument('--pen_l', type=bool, default=False, help='include penalty for not meeting capacity constraints')
    parser.add_argument('--dem_l', type=bool, default=False, help='incorporate direct demand losses i.t.o. a inv. Cross Entropy formulation')
    parser.add_argument('--load_l', type=bool, default=False, help='incorporate perm. invariante load losses for not incquiring the same loads as target routes')
    
    # EVAL
    parser.add_argument('--eval_data', type=str, default='/testing/data_test_Kool/vrp20_test_seed1234.pkl',
                        help="path to data that is to be evaluated")
    parser.add_argument('--greedy', type=bool, default=True,
                        help='greedy evaluation with repair')
    parser.add_argument('--post_process', type=bool, default=True,
                        help='OR tools search on outputted solution')
    parser.add_argument('--model_dir', type=str, default='/outs20/',
                        help='Model to be evaluated')
    parser.add_argument('--guarantee_solution', type=bool, default=False,
                        help='guaranteed solution, but not with m number of vehicles (potentially more vehicles)')

    opts = parser.parse_args(args)


    return opts
