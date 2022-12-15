import torch
import math
import numpy as np
import tensorly as tl
from tensorly import tucker_to_tensor
from tensorly.random import random_tucker
import json


tl.set_backend('pytorch')

def generate_problem(ranks, shape, alpha, kappa, device):
    '''
    Randomly generate a Y and X_star pair.

    Args:
        ranks: iterable of desired low multilinear rank of X_star
        shape: iterable of desired tensor shape
        alpha: corruption fraction between 0 and 1
        kappa: condition number of X_star
        device: device to create tensors
    Output:
        Y: X_star + sparse corruptions
        X_star: low rank tensor
    '''
    core = torch.zeros(ranks, device=device)
    _, factors = random_tucker(shape = tuple(shape), rank=ranks, full=False, orthogonal=True)
    for k in range(len(factors)):
        factors[k] = factors[k].to(device)
    sing_val_decay = (1/kappa)**(1/(ranks[0] - 1))
    for i in range(ranks[0]):
        core[i, i, i] = sing_val_decay ** i
    X_star = tucker_to_tensor((core, factors))
    Y = corrupt_X(X_star, alpha, device)
    return Y, X_star

def corrupt_X(X_star, alpha, device, s_range=None):
    '''
    Sparsely corrupt a tensor.
    Adpated from https://github.com/caesarcai/LRPCA
    HanQin Cai, Jialin Liu, and Wotao Yin. Learned Robust PCA: A Scalable Deep Unfolding Approach for High-Dimensional Outlier Detection. In Advances in Neural Information Processing Systems, 34: 16977-16989, 2021.
    
    Args:
        X_star: tensor to be corrupted
        alpha: corruption fraction between 0 and 1
        device: device to create tensors
        s_range: corruptions will be randomly sampled from [-s_range, +s_range]
    Output:
        Y: X_star + sparse corruptions
        X_star: low rank tensor
    '''
    if not s_range:
        s_range = torch.mean(torch.abs(X_star))
    num_entries = np.prod(X_star.shape)
    idx = torch.randperm(num_entries, device = device)
    idx = idx[:math.floor(alpha * num_entries)]
    S_star 	= torch.rand(len(idx), dtype = torch.float32, device = device)
    S_star	= s_range * (2.0 * S_star - 1.0)
    Y = X_star.reshape(-1)
    Y[idx] = Y[idx] + S_star
    Y = Y.reshape(tuple(X_star.shape))
    return Y


def synthetic_supervised_train(model, data_generator, config, collect_metrics=False, metrics_path=None):
    '''
    Train a model using supervised learning with synthetic streaming data. 
    
    Args:
        model: pytorch model to train
        data_generator: a function that generates a (Y, X_star) pair when given ranks, shape, alpha, and kappa
        config: a dictionary of various hyperparameters
        collect_metrics: whether to collect data while training
        metrics_path: location to store metrics as a json file
    Output:
        metrics: only returned if collect_metrics is True
    '''
    model.train()
    optimizer = torch.optim.Adam([
                {'params': model.z0, 'lr': config['lr'] },
                {'params': model.z1, 'lr': config['lr'] },
                {'params': model.eta, 'lr': config['lr']},
                {'params': model.decay, 'lr': config['lr'] },
            ],)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['scheduler_steps'], gamma=config['scheduler_decay'])

    if collect_metrics:
        metrics = dict()
        metrics['loss_traj'] = []

        metrics['z0_traj'] = [model.z0.item()]
        metrics['z1_traj'] = [model.z1.item()]
        metrics['eta_traj'] = [model.eta.item()]
        metrics['decay_traj'] = [model.decay.item()]

    print("Supervised training starting.")
    for b in range(config['iterations']):
        optimizer.zero_grad()
        Y, X_star = data_generator(ranks=config['ranks'], shape=config['shape'], alpha=config['alpha'], kappa=config['kappa'])
        X, S = model(Y, config["ranks"], config["T"], epsilon=config["eps"])

        loss = ((X_star-X).norm() / X_star.norm())**2
        loss.backward()
        process_grads(model, config)
        
        optimizer.step()
        scheduler.step()

        if collect_metrics:
            metrics['loss_traj'].append(loss.item())
            metrics['z0_traj'].append(model.z0.item())
            metrics['z1_traj'].append(model.z1.item())
            metrics['eta_traj'].append(model.eta.item())
            metrics['decay_traj'].append(model.decay.item())

            if b % config['log_interval'] == config['log_interval'] - 1:
                if metrics_path:
                    with open(metrics_path, 'w') as fp:
                        json.dump(metrics, fp)
                print("Mean loss: ", np.mean(metrics['loss_traj'][-config['log_interval']:]))
    print("Supervised training complete.")
    if collect_metrics:
        return metrics

def single_self_supervised_train(model, Y, config, collect_metrics=False, X_star=None, metrics_path=None):
    '''
    Fine tune a model on one tensor using self-supervised learning. 
    
    Args:
        model: pytorch model to train
        Y: input tensor
        config: a dictionary of various hyperparameters
        collect_metrics: whether to collect data while training
        X_star: ground truth low rank tensor
        metrics_path: location to store metrics as a json file
    Output:
        metrics: only returned if collect_metrics is True
    '''
    model.train()
    optimizer = torch.optim.Adam([
                {'params': model.z0, 'lr': config['lr'] },
                {'params': model.z1, 'lr': config['lr'] },
                {'params': model.eta, 'lr': config['lr']},
                {'params': model.decay, 'lr': config['lr'] },
            ],)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['scheduler_steps'], gamma=config['scheduler_decay'])

    if collect_metrics:
        metrics = dict()
        metrics['loss_traj'] = []
        metrics['reconstruction_loss_traj'] = []
        if X_star is not None:
            metrics['X_loss_traj'] = []

        metrics['z0_traj'] = [model.z0.item()]
        metrics['z1_traj'] = [model.z1.item()]
        metrics['eta_traj'] = [model.eta.item()]
        metrics['decay_traj'] = [model.decay.item()]
    
    print("Self-supervised training starting.")
    for b in range(config['iterations']):
        optimizer.zero_grad()
        
        X, S = model(Y, config["ranks"], config["T"], epsilon=config["eps"])

        loss = (Y-X).norm(p=1) / Y.norm()**2
        loss.backward()
        process_grads(model, config)
        
        optimizer.step()
        scheduler.step()

        if collect_metrics:
            metrics['loss_traj'].append(loss.item())
            metrics['reconstruction_loss_traj'].append(((Y - X - S).norm()**2 / Y.norm()**2).item())
            if X_star is not None:
                metrics['X_loss_traj'].append(((X - X_star).norm()**2 / X_star.norm()**2).item())
            metrics['z0_traj'].append(model.z0.item())
            metrics['z1_traj'].append(model.z1.item())
            metrics['eta_traj'].append(model.eta.item())
            metrics['decay_traj'].append(model.decay.item())

            if b % config['log_interval'] == config['log_interval'] - 1:
                if metrics_path:
                    with open(metrics_path, 'w') as fp:
                        json.dump(metrics, fp)
                print("Mean loss: ", np.mean(metrics['loss_traj'][-config['log_interval']:]))
                print("Mean reconstruction loss: ", np.mean(metrics['reconstruction_loss_traj'][-config['log_interval']:]))
                if X_star is not None:
                    print("Mean X loss: ", np.mean(metrics['X_loss_traj'][-config['log_interval']:]))

    model.eval()
    with torch.no_grad():
        X, S = model(Y, config["ranks"], config["T"], epsilon=config["eps"])
        loss = (Y-X).norm(p=1) / Y.norm()**2
        metrics['loss_traj'].append(loss.item())
        metrics['reconstruction_loss_traj'].append(((Y - X - S).norm()**2 / Y.norm()**2).item())
        if X_star is not None:
            metrics['X_loss_traj'].append(((X - X_star).norm()**2 / X_star.norm()**2).item())
    
    print("Self-supervised training complete.")
    if collect_metrics:
        return metrics

def process_grads(model, config):
    '''
    Gradient processing to prevent errors/divergence.

    Args:
        model: pytorch model to train
        config: a dictionary of various hyperparameters
    '''
    if torch.isnan(model.z0.grad):
        model.z0.grad = torch.zeros_like(model.z0.grad) 
    if torch.isnan(model.z1.grad):
        model.z1.grad = torch.zeros_like(model.z0.grad)
    if torch.isnan(model.eta.grad):
        model.eta.grad = torch.zeros_like(model.z0.grad) 
    if torch.isnan(model.decay.grad):
        model.decay.grad = torch.zeros_like(model.z0.grad) 
    if config['grad_clip']:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'], norm_type='inf')

