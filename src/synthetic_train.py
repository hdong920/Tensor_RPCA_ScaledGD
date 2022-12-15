import torch
import numpy as np
import random
import tensorly as tl
import json

from training_functions import synthetic_supervised_train, single_self_supervised_train, generate_problem
from model import TensorRPCANet

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tl.set_backend('pytorch')
out_dir = '../outputs/synthetic'

supervised_config = dict()

supervised_config['shape'] = [100, 100, 100] # shape of tensors
supervised_config['T'] = 100 # number of iterative updates of RPCA
supervised_config['softplus_factor'] = 0.01 # factor to multiple z0 and z1 by to make gradients nicer

# Network initialization
# NOTE: These are not the actual RPCA hyperparameter values. This values will be saved in json files for convenience instead of the true RPCA values. See model.py for more information.
supervised_config["z0_init"] = 0
supervised_config["z1_init"] = 0
supervised_config["eta_init"] = 0.1
supervised_config["decay_init"] = 0.8

# Logging
supervised_config['log_interval'] = 100

# Optimization parameters
supervised_config["iterations"] = 1000 # Number of backprops. Each batch contains 1 tensor
supervised_config['lr'] = 0.05
supervised_config['grad_clip'] = 100
supervised_config['scheduler_steps'] = 100
supervised_config['scheduler_decay'] = 0.7

# matrix inverse bias term
supervised_config['eps'] = 1e-7

# Fine tuning configs are very similar to supervised learning except for a few changes:
self_supervised_config = supervised_config.copy()
self_supervised_config["iterations"] = 500 
self_supervised_config['log_interval'] = 100
self_supervised_config['lr'] = 0.002
self_supervised_config['scheduler_steps'] = 100
self_supervised_config['scheduler_decay'] = 0.7

supervised_train = True #train a supervised model? Otherwise, a pretrained model will be loaded
eval_samples = 20 # number of samples to fine tune on

# Grid of scenarios to experiement
kappas = [5]
alphas = [0.0, 0.1, 0.2]
ranks = [10, 20, 30]


data_generator = lambda ranks, shape, alpha, kappa: generate_problem(ranks, shape, alpha, kappa, device)

if supervised_train:
    for k in kappas:
        supervised_config['kappa'] = k
        for a in alphas:
            supervised_config['alpha'] = a
            for r in ranks:
                supervised_config['ranks'] = [r, r, r]
                metrics_path = f'{out_dir}/supervised_kappa{k}_alpha{int(10 * a)}_rank{r}.json'
                model = TensorRPCANet(supervised_config['z0_init'], supervised_config['z1_init'], supervised_config['eta_init'], supervised_config['decay_init'], device, supervised_config['softplus_factor'])
                synthetic_supervised_train(model, data_generator, supervised_config, collect_metrics=True, metrics_path=metrics_path)


def choose_params(metrics):
    '''
    Extract z0, z1, eta, and decay from metrics.
    '''
    z0 = metrics['z0_traj'][-1]
    z1 = metrics['z1_traj'][-1]
    eta = metrics['eta_traj'][-1]
    decay = metrics['decay_traj'][-1]
    return z0, z1, eta, decay


for k in kappas:
    self_supervised_config['kappa'] = k
    for a in alphas:
        self_supervised_config['alpha'] = a
        for r in ranks:
            self_supervised_config['ranks'] = [r, r, r]

            # Loading the supervised model
            supervised_metrics_path = f'{out_dir}/supervised_kappa{k}_alpha{int(10 * a)}_rank{r}.json'
            with open(supervised_metrics_path, 'r') as f:
                metrics = json.load(f)
            z0, z1, eta, decay = choose_params(metrics)

            # Self-supervised fine tuning
            self_supervised_metrics_path = f'{out_dir}/self_supervised_kappa{k}_alpha{int(10 * a)}_rank{r}.json'    
            self_supervised_metrics = dict()
            for s in range(eval_samples):
                model = TensorRPCANet(z0, z1, eta, decay, device, self_supervised_config['softplus_factor'])
                Y, X_star = data_generator(ranks=[r, r, r], shape=self_supervised_config['shape'], alpha=a, kappa=k)
                self_supervised_metrics[s] = single_self_supervised_train(model, Y, self_supervised_config, collect_metrics=True, X_star=X_star, metrics_path=None)
                with open(self_supervised_metrics_path, 'w') as fp:
                    json.dump(self_supervised_metrics, fp) # note the first index is the results from purely supervised learning