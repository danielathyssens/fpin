#! /usr/bin/env python

import os
from parse_options import get_options

# python lib imports
import pprint as pp
import numpy as np
import random
import torch
# import tqdm
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# model lib imports
# from VRP_Loss import VRPLoss
from VRP_Loss1 import VRPLoss
from VRPModel_attn1 import VRP_Net
from utils_all.basic_funcs import to_variable, zeros
from utils_all.print_out import print_func
# from data_utils.preprocess import import_trainval_data
from data_utils.preprocess1 import import_trainval_data
from utils_all.basic_funcs import sparse_dense_mul
from utils_all.learn_funcs import load_ckp


def train(opts):
    # Pretty print the run args
    pp.pprint(vars(opts))
    # handles
    v5 = False

    if opts.vrp_size == 20:
        vrp_to_solve = 'VRP20'
        nr_files = 29
        VRP50 = False
    elif opts.vrp_size == 50:
        vrp_to_solve = 'VRP50'
        nr_files = 23
        VRP50 = True
    elif opts.vrp_size == 100:
        vrp_to_solve = 'VRP100'
        nr_files = 16
    else:
        print('Error: No valid vrp_size specified')
        vrp_to_solve = None
        nr_files = 0

    # parameters
    model_params = {'with_loads': opts.load, 'attn_dotproduct': True, 'memory_efficient': True,
                    'avg_pool': False, 'residual': True, 'norm': True, 'self_pool': False, 'embedding_norm': True,
                    'weighting': True, 'fleet_dim': 4, 'cities_dim': 3, 'depot_dim': 4}
    loss_params = {'simple_loss': False, 'no_perms': False, 'size_average': True, 'with_penalty': opts.pen_l,
                   'with_loads_loss': opts.load_l}

    class HParams(object):
        def __init__(self, lr=None, wd=None, dropout=None, batch_s=None, n_hidden=None, layers=None,
                     n_SOFTlayers=None, mainDimension=None, starts_weight=None):
            self.lr = opts.lr if lr is None else lr
            self.wd = opts.wd if wd is None else wd
            self.dropout = 0.0 if dropout is None else dropout
            self.batch_s = int(opts.batch_size if batch_s is None else batch_s)
            self.n_hidden = int(opts.n_hidden if n_hidden is None else n_hidden)
            self.layers = int(opts.layers if layers is None else layers)
            self.n_SOFTlayers = int(0 if n_SOFTlayers is None or n_SOFTlayers == 0 else n_SOFTlayers)
            self.mainDimension = int(opts.main_dim if mainDimension is None or mainDimension == 0 else mainDimension)
            self.starts_weight = 0.2 if starts_weight is None else starts_weight

        def to_string(self):
            return 'learning rate: {}, weight decay: {}, batch size: {}, dropout: {}, n_hidden: {}, layers: {}, ' \
                   'n_SOFTlayers: {}, main_dim: {}, starts_weight: {}'.format(
                self.lr, self.wd, self.batch_s, self.dropout, self.n_hidden, self.layers, self.n_SOFTlayers,
                self.mainDimension, self.starts_weight)

    HP = HParams()
    # print('Chosen Hyperparameter options: ',HP.to_string())

    dat_params = {'batch_size': HP.batch_s,
                  'shuffle': True,
                  'num_workers': 2}

    dat_params_v = {'batch_size': HP.batch_s,
                    'shuffle': False,
                    'num_workers': 2}

    class Dataset_VRP(torch.utils.data.Dataset):
        """Characterizes a dataset for PyTorch"""

        def __init__(self, list_IDs, X_dat, Y_dat, YLoad_dat):
            """Initialization"""
            self.list_IDs = list_IDs
            self.X_dat = X_dat
            self.Y_dat = Y_dat
            self.YLoad_dat = YLoad_dat

        def __len__(self):
            """Denotes total nr of samples"""
            return len(self.list_IDs)

        def __getitem__(self, index):
            """Generates one sample of data"""

            # Select sample
            ID = self.list_IDs[index]
            X = self.X_dat[ID]
            y = self.Y_dat[ID]
            y_load = self.YLoad_dat[ID]

            return X, y, y_load

    # import training and validation data
    data_X, data_Y, data_YLoad = import_trainval_data(vrp_to_solve, nr_files)

    # print('data_Y[0].is_sparse', data_Y[0].is_sparse)

    # shuffle training instances
    c = list(zip(data_X, data_Y, data_YLoad))
    random.shuffle(c)
    data_X, data_Y, data_YLoad = zip(*c)
    tr = len(data_X) - 10000

    # Train Test Partition
    partition = {
        'train': [i for i in range(tr)],
        'val': [i for i in range(tr, len(data_X))]
    }
    # Generators
    training_set = Dataset_VRP(partition['train'], data_X, data_Y, data_YLoad)
    val_set = Dataset_VRP(partition['val'], data_X, data_Y, data_YLoad)
    training_generator = DataLoader(training_set, **dat_params)
    val_generator = DataLoader(val_set, **dat_params_v)

    # TRAINING PHASE STARTS

    # limit for training if there's no val_loss improvement
    NO_IMPRVMT_LIMIT = 9
    # max epochs to train for
    max_epochs = 100
    # some log lists
    training_loss_eps, validation_loss_eps, CapViolation_eps, CapViolation_mean_eps, CapViolation_mean_eps_v = [], [], [], [], []

    model = VRP_Net(HP.layers, model_params['depot_dim'], model_params['cities_dim'], model_params['fleet_dim'],
                    HP.mainDimension,
                    model_params['avg_pool'], model_params['residual'], model_params['norm'], HP.n_hidden, HP.dropout,
                    model_params['self_pool'], model_params['embedding_norm'], HP.n_SOFTlayers,
                    model_params['weighting'], model_params['with_loads'], model_params['attn_dotproduct'],
                    model_params['memory_efficient']).cuda()

    vrp_loss = VRPLoss(HP.starts_weight, loss_params['simple_loss'],
                       loss_params['no_perms'], loss_params['size_average'],
                       loss_params['with_penalty'],
                       loss_params['with_loads_loss']).cuda()

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=HP.lr, weight_decay=HP.wd)

    if opts.resume:
        model, optimizer, start_epoch = load_ckp(opts.model_ckp, model, optimizer)

    # Loop over epochs
    epochs_done = 0  # initiate epoch count for early stopping
    best_loss = 99999.0  # initiate best loss to beat
    no_imprvmt_count = 0
    # alpha = opts.penalty_weight
    for epoch in range(max_epochs):
        epochs_done += 1

        # set model to train mode
        model.train()
        # batch-wise logs
        train_loss_batches, violations_batch, violations_batch_mean, violations_batch_mean_v = [], [], [], []
        for local_batch, targets, target_loads in training_generator:
            # Unpack X_groups
            fleet_batch = local_batch[0].float()
            depot_batch = local_batch[1].float()
            custom_batch = local_batch[2].float()
            demand_batch = local_batch[3].float()
            dists_batch = local_batch[4].float()
            # sparsify targets before to_variable
            targets = targets.to_sparse()

            # Transfer to GPU
            depot_batch, custom_batch, fleet_batch, dists_batch, demand_batch, targets, target_loads = to_variable(
                depot_batch, custom_batch, fleet_batch, dists_batch, demand_batch, targets, target_loads)

            # Zero gradients
            optimizer.zero_grad()
            vrp_probs, vrp_loads, sample_path_b0 = model(depot_batch, custom_batch,
                                                         fleet_batch,
                                                         demand_batch, dists_batch)

            if vrp_loads is not None:
                # Float(batch):
                capa_violation_sum = torch.where(vrp_loads > 1.00001, vrp_loads - 1.0000, zeros(1)).sum(1)
                # mean across vehicles
                capa_violation_mean = torch.mean(torch.where(vrp_loads > 1.00001, vrp_loads - 1.0000, zeros(1)),
                                                 dim=1)

                violations_batch.append(torch.mean(capa_violation_sum).item())
                violations_batch_mean.append(torch.mean(capa_violation_mean).item())
                
                violations_batch_mean_v.append(torch.mean(capa_violation_mean).item())

            loss = vrp_loss(vrp_probs, vrp_loads, targets, target_loads)
            # record train loss performance
            train_loss_batches.append(loss.item())
            # perform a backward pass, and update the weights.
            loss.backward()
            optimizer.step()

        ###################### Validation ######################
        model.eval()
        valid_loss_batches = []
        with torch.set_grad_enabled(False):
            for local_batch, targets_v, target_loads_v in val_generator:
                # unpack X_groups
                fleet_batch_v = local_batch[0].float()
                depot_batch_v = local_batch[1].float()
                custom_batch_v = local_batch[2].float()
                demand_batch_v = local_batch[3].float()
                dists_batch_v = local_batch[4].float()
                # sparsify targets before to_variable
                targets_v = targets_v.to_sparse()

                depot_batch_v, custom_batch_v, fleet_batch_v, dists_batch_v, demand_batch_v, targets_v, target_loads_v = to_variable(
                    depot_batch_v, custom_batch_v, fleet_batch_v, dists_batch_v, demand_batch_v, targets_v,
                    target_loads_v)
                    
                vrp_probs_v, vrp_loads_v, sample_path_b0_v = model(depot_batch_v,
                                                                   custom_batch_v,
                                                                   fleet_batch_v,
                                                                   demand_batch_v,
                                                                   dists_batch_v)
                if vrp_loads_v is not None:
                    # mean across vehicles
                    capa_violation_mean_v = torch.mean(torch.where(vrp_loads_v > 1.00001, vrp_loads_v - 1.0000, zeros(1)),dim=1)
                    violations_batch_mean_v.append(torch.mean(capa_violation_mean_v).item())

                loss_v = vrp_loss(vrp_probs_v, vrp_loads_v, targets_v, target_loads_v)

                # Record val loss
                valid_loss_batches.append(loss_v.item())

        # APPENDING EPOCH LOSS
        curr_val_loss = np.mean(valid_loss_batches)
        validation_loss_eps.append(np.mean(valid_loss_batches))
        training_loss_eps.append(np.mean(train_loss_batches))

        if vrp_loads is not None:
            # APPENDING EPOCH VIOLATION
            CapViolation_eps.append(np.mean(violations_batch))
            CapViolation_mean_eps.append(np.mean(violations_batch_mean))
        if vrp_loads_v is not None:
            # APPENDING EPOCH VIOLATION FOR VALIDATION
            CapViolation_mean_eps_v.append(np.mean(violations_batch_mean_v))
        if epoch % 2 == 0:
            print('\nEpoch: {}, TRAIN loss (avg of all batches): {}'.format(epoch, np.mean(train_loss_batches)))
            # print_func(VRP_probs,sample_path_b0,VRP_loads,targets,target_loads,model_params['with_loads'],VRP50,v5)
            print('\nEpoch: {}, VALID loss (avg of all batches): {}'.format(epoch, np.mean(valid_loss_batches)))
            if not opts.vrp_size == 100:
                print_func(vrp_probs_v, sample_path_b0_v, vrp_loads_v, targets_v, target_loads_v,
                           model_params['with_loads'], VRP50, v5)

            if vrp_loads is not None:
                print('Epoch: {},MEAN CAPA VIOL (avg of all batches): {}'.format(epoch, np.mean(violations_batch_mean)))
            if vrp_loads_v is not None:
                print('Epoch: {}, MEAN CAPA VIOL VAL (avg of all batches): {}'.format(epoch, np.mean(violations_batch_mean_v)))

        ################### END OF ONE EPOCH ###################
        # simple early stopping
        if curr_val_loss < best_loss:
            best_loss = curr_val_loss
            # Update best model parameters
            best_model = model.state_dict()
            no_imprvmt_count = 0
        else:
            no_imprvmt_count += 1
            print(no_imprvmt_count)
        # save ckpts:
        if epoch % opts.checkpoint_epochs == 0:
            print('Saving model and state...')
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': vrp_loss
                },
                os.path.join(opts.save_dir, 'cps/epoch-{}.pt'.format(epochs_done))
            )
        if no_imprvmt_count >= NO_IMPRVMT_LIMIT:
            print('no Improvement:', no_imprvmt_count)
            print('last best_loss:', best_loss)
            print('curr val_loss:', curr_val_loss)
            break

    return epochs_done, best_loss, best_model, validation_loss_eps, training_loss_eps, CapViolation_eps, CapViolation_mean_eps, CapViolation_mean_eps_v


if __name__ == "__main__":
    num_eps, best_val_loss, model_params, val_loss, train_loss, CapViolation, CapViolation_mean, CapViolation_mean_v = train(get_options())
    # Save best model params
    # /home/thyssens/job_vrp100/outs/
    print('get_options().save_dir', get_options().save_dir)
    save_outs = get_options().save_dir
    # /home/thyssens/job_vrp100/outs/VRP_model_100.pth
    # with open(os.path.join(save_outs, 'VRP_train_loss_1.pt')) as f:
    torch.save(model_params, os.path.join(save_outs, 'VRP_model.pth'))
    # Save loss over epochs
    with open(os.path.join(save_outs, 'VRP_train_loss_1.pt'), "wb") as f:
        torch.save(train_loss, f)
    with open(os.path.join(save_outs, 'VRP_val_loss_1.pt'), "wb") as f:
        torch.save(val_loss, f)
    with open(os.path.join(save_outs, 'VRP_capa_viol_1.pt'), "wb") as f:
        torch.save(CapViolation, f)
    with open(os.path.join(save_outs, 'VRP_Mcapa_viol_1.pt'), "wb") as f:
        torch.save(CapViolation_mean, f)
    with open(os.path.join(save_outs, 'VRP_Mcapa_viol_v_1.pt'), "wb") as f:
        torch.save(CapViolation_mean_v, f)
